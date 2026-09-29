"""Stage 6: GARCH(1,1) and GJR-GARCH on SARIMAX residuals, dynamic CI, and honest evaluation.

GARCH parameters (omega, alpha, beta) are estimated once on the *in-sample* (training) residuals
only, exactly like the mean model. The out-of-sample conditional variance is then produced by running
the GARCH recursion forward through the test period using the actual realized one-step-ahead
residuals from stage 5 -- parameters are never re-estimated on test data, only the recursion's state
(sigma2) is updated as real residuals arrive, which is what a deployed system does at each new hour.
"""
import numpy as np
import pandas as pd
from arch import arch_model

IN_PATH = "data/processed/mean_model_residuals.parquet"
OUT_PATH = "data/processed/variance_model_output.parquet"

SCALE = 10.0  # arch works best with residuals scaled to roughly unit variance; PC1 residuals are ~0.5-1


def fit_garch(resid_train: pd.Series, dist: str, o: int = 0):
    model = arch_model(resid_train * SCALE, mean="Zero", vol="GARCH", p=1, o=o, q=1, dist=dist)
    fit = model.fit(disp="off")
    return fit


def recursive_forecast_variance(fit, resid_test: pd.Series) -> pd.Series:
    """Runs the fitted GARCH recursion forward through resid_test without re-estimating parameters."""
    params = fit.params
    omega = params["omega"]
    alpha = params["alpha[1]"]
    beta = params["beta[1]"]
    gamma = params.get("gamma[1]", 0.0)

    sigma2 = np.empty(len(resid_test))
    last_sigma2 = fit.conditional_volatility.iloc[-1] ** 2
    last_eps = fit.resid.iloc[-1]

    scaled_resid = (resid_test * SCALE).values
    prev_sigma2, prev_eps = last_sigma2, last_eps
    for i, eps in enumerate(scaled_resid):
        leverage = gamma * (prev_eps ** 2) * (1 if prev_eps < 0 else 0)
        s2 = omega + alpha * prev_eps ** 2 + leverage + beta * prev_sigma2
        sigma2[i] = s2
        prev_sigma2, prev_eps = s2, eps

    return pd.Series(np.sqrt(sigma2) / SCALE, index=resid_test.index, name="cond_vol")


def evaluate_coverage(actual, pred_mean, sigma, label: str) -> float:
    lower = pred_mean - 1.96 * sigma
    upper = pred_mean + 1.96 * sigma
    coverage = ((actual >= lower) & (actual <= upper)).mean()
    mean_width = (upper - lower).mean()
    print(f"{label}: 95% CI coverage = {coverage*100:.1f}%, mean width = {mean_width:.3f}")
    return coverage


def main():
    df = pd.read_parquet(IN_PATH)
    train = df[df["in_sample"]]
    test = df[~df["in_sample"]]

    resid_train = train["residual"]
    resid_test = test["residual"]

    print(f"Fitting GARCH(1,1) and GJR-GARCH(1,1,1) on {len(resid_train)} in-sample residuals...")
    garch_fit = fit_garch(resid_train, dist="normal", o=0)
    gjr_fit = fit_garch(resid_train, dist="normal", o=1)

    print("\nGARCH(1,1) params:")
    print(garch_fit.params.to_string())
    print(f"alpha + beta = {garch_fit.params['alpha[1]'] + garch_fit.params['beta[1]']:.4f} (persistence; <1 required for stationarity)")

    print("\nGJR-GARCH(1,1,1) params:")
    print(gjr_fit.params.to_string())
    print(f"GJR AIC={gjr_fit.aic:.1f} vs GARCH AIC={garch_fit.aic:.1f} ({'GJR preferred' if gjr_fit.aic < garch_fit.aic else 'plain GARCH preferred'})")

    garch_sigma_test = recursive_forecast_variance(garch_fit, resid_test)
    gjr_sigma_test = recursive_forecast_variance(gjr_fit, resid_test)

    static_sigma = resid_train.std()
    pred_mean_test = test["fitted"]
    actual_test = test["actual"]

    print(f"\n--- Out-of-sample 95% CI comparison (test set, {len(test)} hours) ---")
    evaluate_coverage(actual_test, pred_mean_test, static_sigma, "Static (constant-variance SARIMAX)")
    evaluate_coverage(actual_test, pred_mean_test, garch_sigma_test, "Dynamic (GARCH(1,1))         ")
    evaluate_coverage(actual_test, pred_mean_test, gjr_sigma_test, "Dynamic (GJR-GARCH(1,1,1))   ")

    # Aggregate coverage/width alone can't show whether GARCH is doing anything useful -- a constant
    # interval can match average coverage by luck. What actually demonstrates "dynamic" is: (a) sigma
    # genuinely varies over the test window, not just numerically but in a way that (b) tracks the size
    # of the shocks that actually occur.
    print(f"\nGARCH conditional vol over test window: min={garch_sigma_test.min():.3f}, "
          f"max={garch_sigma_test.max():.3f}, ratio={garch_sigma_test.max()/garch_sigma_test.min():.2f}x")
    corr = np.corrcoef(garch_sigma_test, resid_test.abs())[0, 1]
    print(f"Correlation between GARCH sigma(t) and |residual(t)| (does predicted vol track realized shock size?): {corr:.3f}")

    biggest_shock_idx = resid_test.abs().idxmax()
    window = slice(biggest_shock_idx - pd.Timedelta(hours=12), biggest_shock_idx + pd.Timedelta(hours=12))
    print(f"\nBiggest test-set shock at {biggest_shock_idx} (residual={resid_test.loc[biggest_shock_idx]:.2f}):")
    case_study = pd.DataFrame({
        "actual": actual_test.loc[window],
        "fitted": pred_mean_test.loc[window],
        "static_sigma": static_sigma,
        "garch_sigma": garch_sigma_test.loc[window],
    })
    print(case_study.round(3).to_string())

    in_sample_vol = pd.Series(garch_fit.conditional_volatility / SCALE, index=resid_train.index, name="cond_vol")
    full_vol = pd.concat([in_sample_vol, garch_sigma_test])
    full_gjr_vol = pd.concat([pd.Series(gjr_fit.conditional_volatility / SCALE, index=resid_train.index), gjr_sigma_test])

    out = df.copy()
    out["garch_sigma"] = full_vol
    out["gjr_sigma"] = full_gjr_vol
    out["static_sigma"] = static_sigma
    out.to_parquet(OUT_PATH)
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
