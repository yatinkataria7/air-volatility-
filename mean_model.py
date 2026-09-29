"""Stage 5: SARIMAX mean equation on PC1, order selected via a hand-rolled AIC grid search.

pmdarima is not used -- confirmed broken against numpy>=2.0 (this machine runs 2.5.1). The grid
search below does what auto_arima does internally: fit a small set of (p,d,q) candidates by MLE and
keep the lowest-AIC one. Daily/weekly seasonality is handled by the Fourier exogenous regressors from
stage 4, not by a seasonal_order term, so the grid only searches the non-seasonal part.
"""
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX

IN_PATH = "data/processed/air_quality_features.parquet"
RESID_PATH = "data/processed/mean_model_residuals.parquet"

TARGET = "PC1"
EXOG_COLS = [c for c in []] + [
    "fourier_daily_sin_1", "fourier_daily_cos_1",
    "fourier_daily_sin_2", "fourier_daily_cos_2",
    "fourier_daily_sin_3", "fourier_daily_cos_3",
    "fourier_weekly_sin_1", "fourier_weekly_cos_1",
    "fourier_weekly_sin_2", "fourier_weekly_cos_2",
    "T",
]
TEST_WEEKS = 5  # single held-out tail window, not a walk-forward -- see PLAN.md

P_RANGE = range(0, 4)
Q_RANGE = range(0, 4)


def check_stationarity(y: pd.Series) -> int:
    stat, pvalue, *_ = adfuller(y, autolag="AIC")
    print(f"ADF test on {TARGET}: statistic={stat:.3f}, p-value={pvalue:.4f}")
    d = 0 if pvalue < 0.05 else 1
    print(f"-> using d={d}")
    return d


def grid_search(y: pd.Series, exog: pd.DataFrame, d: int):
    best = None
    results = []
    for p in P_RANGE:
        for q in Q_RANGE:
            if p == 0 and q == 0:
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = SARIMAX(
                        y, exog=exog, order=(p, d, q),
                        enforce_stationarity=False, enforce_invertibility=False,
                    )
                    fit = model.fit(disp=False, maxiter=100)
                results.append((p, d, q, fit.aic))
                if best is None or fit.aic < best[1]:
                    best = ((p, d, q), fit.aic, fit)
            except Exception as e:
                results.append((p, d, q, None))
    results_df = pd.DataFrame(results, columns=["p", "d", "q", "aic"]).sort_values("aic")
    print("\nGrid search results (top 10 by AIC):")
    print(results_df.head(10).to_string(index=False))
    return best


def main():
    df = pd.read_parquet(IN_PATH)
    y = df[TARGET]
    exog = df[EXOG_COLS]

    cutoff = df.index.max() - pd.Timedelta(weeks=TEST_WEEKS)
    train_mask = df.index <= cutoff
    y_train, y_test = y[train_mask], y[~train_mask]
    exog_train, exog_test = exog[train_mask], exog[~train_mask]
    print(f"Train: {y_train.index.min()} -> {y_train.index.max()} ({len(y_train)} rows)")
    print(f"Test:  {y_test.index.min()} -> {y_test.index.max()} ({len(y_test)} rows)")

    d = check_stationarity(y_train)
    (order, aic, fit) = grid_search(y_train, exog_train, d)
    print(f"\nBest order: {order}, AIC={aic:.1f}")
    print(fit.summary())

    in_sample_resid = fit.resid
    in_sample_resid.name = "residual"

    # Rolling one-step-ahead forecasts on the test set, parameters fixed at the training MLE estimate
    # (no re-fitting -- this is the "reduced walk-forward" agreed on, cheap because it's just Kalman
    # state updates, not re-estimation). This is what a deployed system would actually do: at each
    # hour, forecast one step ahead using everything observed so far, not commit on day 1 to an
    # 840-hour-ahead static forecast. statsmodels' `.append(..., refit=False)` extends the state space
    # with the real test observations and its `.resid` gives the true one-step-ahead prediction error
    # at each of those points, still using only the parameters estimated on the training set.
    full_fit = fit.append(endog=y_test, exog=exog_test, refit=False)
    oos_resid = full_fit.resid.loc[y_test.index]
    oos_resid.name = "residual"
    pred_mean = y_test - oos_resid

    static_ses = np.sqrt(fit.params.get("sigma2", None)) if "sigma2" in fit.params.index else None
    in_sample_sigma = np.sqrt(fit.params["sigma2"])
    pred_ci_lower = pred_mean - 1.96 * in_sample_sigma
    pred_ci_upper = pred_mean + 1.96 * in_sample_sigma

    rmse_oos = np.sqrt((oos_resid ** 2).mean())
    naive_rmse = np.sqrt(((y_test - y_test.shift(1).fillna(y_train.iloc[-1])) ** 2).mean())
    print(f"\nOut-of-sample rolling 1-step RMSE on PC1 (last {TEST_WEEKS} weeks): {rmse_oos:.3f}")
    print(f"Naive (previous-hour-value) RMSE for comparison: {naive_rmse:.3f}")

    coverage = ((y_test >= pred_ci_lower) & (y_test <= pred_ci_upper)).mean()
    print(f"Static-width 95% CI coverage on test set (before GARCH makes it dynamic): {coverage*100:.1f}%")

    out = pd.DataFrame({
        "in_sample": pd.concat([pd.Series(True, index=in_sample_resid.index),
                                 pd.Series(False, index=oos_resid.index)]),
        "residual": pd.concat([in_sample_resid, oos_resid]),
        "actual": pd.concat([y_train, y_test]),
        "fitted": pd.concat([y_train - in_sample_resid, pred_mean]),
    })
    out.to_parquet(RESID_PATH)
    print(f"\nSaved residuals to {RESID_PATH}")


if __name__ == "__main__":
    main()
