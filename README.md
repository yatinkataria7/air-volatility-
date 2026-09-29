# Urban Air Quality Volatility & Pollution Shock Modeling

A two-stage time-series forecasting pipeline for modeling **urban air pollution levels and time-varying uncertainty** using the UCI Air Quality dataset.

The project combines:

* **PCA** to construct a composite pollution factor
* **SARIMAX** to model expected pollution levels
* **GARCH / GJR-GARCH** to model time-varying forecast uncertainty
* **Rolling one-step-ahead forecasting** to evaluate the models under realistic deployment conditions

The key idea is that pollution forecast errors are **not constant over time**. After an atmospheric shock, uncertainty can increase substantially and then gradually decline as conditions stabilize.

---

## Key Finding

The project distinguishes between **average interval calibration** and **conditional uncertainty estimation**.

Although static, GARCH, and GJR-GARCH intervals achieve very similar overall 95% coverage, GARCH provides information about **when uncertainty is elevated**.

During the test period:

* Conditional volatility varied by **2.18×**, from 0.461 to 1.005.
* Following the largest test-set pollution shock, GARCH volatility reached its maximum during the following hour and then gradually declined.
* A static confidence interval remained unchanged throughout the same episode.

Therefore, the main contribution is not improved average coverage, but the ability to represent **time-varying uncertainty following pollution shocks**.

---

## Results

| Metric                        |             Result |
| ----------------------------- | -----------------: |
| PCA variance explained by PC1 |         **82.87%** |
| Mean model                    | **SARIMAX(2,0,3)** |
| One-step RMSE                 |          **0.676** |
| Naive RMSE                    |          **0.915** |
| Error reduction               |            **26%** |
| ADF statistic                 |          **−9.02** |
| ADF p-value                   |       **< 0.0001** |
| GARCH persistence α + β       |          **0.983** |
| GJR-GARCH γ                   |         **−0.088** |
| Conditional volatility range  |  **0.461 – 1.005** |
| Static 95% CI coverage        |          **93.8%** |
| GARCH 95% CI coverage         |          **93.9%** |
| GJR-GARCH 95% CI coverage     |          **93.7%** |

The SARIMAX model reduced one-step-ahead RMSE from 0.915 to 0.676 compared with the naive baseline, corresponding to a **26% reduction in error**.

---

# Dataset

The project uses the **UCI Air Quality Dataset**, containing hourly measurements from a multi-sensor monitoring device located on a polluted road in an Italian city.

### Dataset characteristics

* **9,357 hourly observations**
* Time range: **2004-03-10 18:00 → 2005-04-04 14:00**
* Continuous hourly time grid
* Missing values encoded using `-200`
* Multiple gas concentration measurements
* Temperature, relative humidity, and absolute humidity measurements
* 16 contiguous sensor/weather outages ranging from 1–76 hours

`NMHC(GT)` was removed because more than 90% of its observations were missing.

---

# Methodology

```text
                 UCI Air Quality Dataset
                          │
                          ▼
                ┌──────────────────┐
                │ Ingest & Cleaning │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │ Outlier Detection │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │ Dual Imputation  │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │ Features + PCA   │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │     SARIMAX      │
                │  Mean Forecast   │
                └────────┬─────────┘
                         │
                     Residuals
                         │
                         ▼
                ┌──────────────────┐
                │   GARCH / GJR    │
                │ Volatility Model │
                └────────┬─────────┘
                         │
                         ▼
              Dynamic Prediction Intervals
```

---

## 1. Data Ingestion & Cleaning

`src/ingest.py`

The first stage:

* Downloads the dataset using `ucimlrepo`
* Constructs a proper `DatetimeIndex`
* Converts the `-200` missing-value sentinel into `NaN`
* Removes `NMHC(GT)`
* Reindexes the dataset onto a continuous hourly grid
* Reports missingness for each variable

The implementation also verifies the actual timestamp format delivered by the dataset rather than relying blindly on the published description.

---

## 2. Outlier Detection

`src/outliers.py`

A major challenge is distinguishing between:

1. **Hardware glitches**, which should be removed
2. **Real pollution shocks**, which must be retained because the volatility model needs them

A conventional rolling Hampel filter was found to incorrectly flag normal daily temperature and humidity changes.

Instead, the project compares each observation against an **interpolation between its immediate neighbors**:

```text
t-1 ───────────── t ───────────── t+1
          │
          └── compare actual t
              with neighbor interpolation
```

A run-length rule is then used to avoid removing persistent real events.

---

## 3. Dual Imputation

`src/impute.py`

Different missing-gap lengths are treated differently.

### Short gaps

For gaps of **≤ 2 hours**:

**PCHIP — Piecewise Cubic Hermite Interpolation**

PCHIP is used because it preserves the shape of the surrounding observations and avoids the overshooting problem encountered with ordinary cubic splines.

### Long gaps

For gaps of **3 hours or more**:

**IterativeImputer + Bayesian Ridge**

Each variable is estimated using information from the other available channels.

Temperature, relative humidity, and absolute humidity provide useful information during gas-sensor outages.

Finally, physically non-negative variables are clipped at zero.

---

# 4. Feature Engineering & PCA

`src/features.py`

The model incorporates periodic pollution patterns using Fourier terms:

* **3 daily harmonics**
* **2 weekly harmonics**

These are included as exogenous regressors so the ARIMA component does not need to relearn predictable daily and weekly patterns.

### PCA

PCA is performed on four ground-truth pollutant concentrations:

* `CO(GT)`
* `C6H6(GT)`
* `NOx(GT)`
* `NO2(GT)`

The first principal component explains:

> **82.87% of total variance**

This PC1 becomes the composite pollution factor used by the forecasting pipeline.

Metal-oxide sensor channels are excluded because their measurements are resistance-based arbitrary units rather than concentrations. Ozone is also excluded because there is no corresponding `O3(GT)` ground-truth concentration in the selected reference set.

---

# 5. SARIMAX Mean Model

`src/mean_model.py`

The first modeling stage predicts the **expected pollution level**.

### Stationarity

An Augmented Dickey-Fuller test gives:

```text
ADF statistic = -9.02
p-value       < 0.0001
```

Therefore, PC1 is treated as stationary and:

```text
d = 0
```

### Model selection

An AIC-based grid search over:

```text
p ∈ {0,1,2,3}
q ∈ {0,1,2,3}
```

selects:

```text
SARIMAX(2,0,3)
```

with Fourier terms and temperature as exogenous variables.

### Evaluation

The model is evaluated on a **held-out five-week tail window** using rolling one-step-ahead forecasts.

The model parameters remain fixed after training while the Kalman state advances as new observations become available.

This better represents how an hourly forecasting system would operate in deployment.

---

# 6. GARCH Volatility Model

`src/variance_model.py`

The residuals from the SARIMAX model are passed into a volatility model.

Two models are evaluated:

* **GARCH(1,1)**
* **GJR-GARCH(1,1,1)**

The purpose is not to predict the pollution level itself, but to estimate:

> **How uncertain the next prediction is likely to be.**

The GARCH models are fitted only on the in-sample SARIMAX residuals.

During testing, the conditional variance is recursively updated using realized one-step-ahead residuals without re-estimating model parameters on the test set.

---

# Why GARCH?

A conventional forecasting model might assume:

```text
Prediction uncertainty = constant
```

But pollution data can behave more like:

```text
Normal conditions
      ↓
Low volatility
      ↓
Pollution shock
      ↓
High volatility
      ↓
Gradual stabilization
      ↓
Low volatility
```

GARCH captures this changing uncertainty.

For example:

```text
Expected pollution = 5

Static interval:
    4 ───────────── 6

After a shock:

Dynamic interval:
    3 ───────────────── 9
```

The expected value can remain similar while the uncertainty surrounding it changes significantly.

---

# GARCH vs Static Confidence Intervals

The project deliberately avoids claiming that GARCH improves average coverage.

| Interval  | 95% Coverage |
| --------- | -----------: |
| Static    |        93.8% |
| GARCH     |        93.9% |
| GJR-GARCH |        93.7% |

The difference is extremely small.

The more important result is the **conditional behavior of volatility**.

During the test period:

```text
Minimum σ = 0.461
Maximum σ = 1.005

Maximum / Minimum = 2.18×
```

At the largest test-set shock:

```text
Date:   2005-03-24 19:00
Actual: 4.82
Fitted: 1.57
```

GARCH volatility reached its test-period maximum during the following hour and then gradually declined, while the static interval remained unchanged.

---

# GJR-GARCH Result

The GJR-GARCH model estimates:

```text
γ = -0.088
```

This indicates an asymmetric response of volatility to positive and negative shocks.

Interestingly, the sign is opposite to the traditional financial-market leverage effect.

The project does **not** force this result to match financial-market intuition. Instead, it reports the result as a domain-specific empirical observation that may have a physical interpretation in atmospheric pollution dynamics.

---

# Reproducibility

Clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd <repository-name>

pip install -r requirements.txt
```

Then run the six pipeline stages sequentially:

```bash
python -m src.ingest
python -m src.outliers
python -m src.impute
python -m src.features
python -m src.mean_model
python -m src.variance_model
```

Each stage reads the previous stage's output from `data/` and generates the next artifact.

The first stage automatically downloads the UCI dataset, so no manual dataset download or API key is required.

### Runtime

`src.mean_model` is the most computationally expensive stage because it evaluates 15 candidate SARIMAX configurations over thousands of observations with multiple exogenous regressors.

The remaining stages execute comparatively quickly.

---

# Repository Structure

```text
.
├── src/
│   ├── ingest.py
│   ├── outliers.py
│   ├── impute.py
│   ├── features.py
│   ├── mean_model.py
│   └── variance_model.py
│
├── PLAN.md
├── REPORT.md
├── INTERVIEW_PREP.md
├── requirements.txt
└── data/
```

### Source files

| File                | Purpose                                               |
| ------------------- | ----------------------------------------------------- |
| `ingest.py`         | Dataset download, cleaning and timestamp handling     |
| `outliers.py`       | Hardware-glitch detection and real-event preservation |
| `impute.py`         | PCHIP and IterativeImputer                            |
| `features.py`       | Fourier features and PCA                              |
| `mean_model.py`     | SARIMAX model and AIC search                          |
| `variance_model.py` | GARCH/GJR-GARCH and evaluation                        |

The `data/` directory contains generated artifacts and is intentionally excluded from version control.

---

# Important Implementation Details

The project uncovered several issues that produced plausible-looking but incorrect results.

### 1. Timestamp parsing

Using the documented date format caused approximately **61% of rows to be discarded** because the delivered data used a different date representation.

### 2. Outlier detection

A standard rolling-median filter incorrectly classified normal daily weather cycles as anomalies.

### 3. Spline interpolation

Cubic spline interpolation generated physically impossible negative pollution concentrations.

### 4. Iterative imputation

Even after switching to PCHIP, Bayesian Ridge imputation generated additional negative concentration estimates. A physical non-negativity constraint was therefore applied after imputation.

### 5. Backtesting design

An 840-step static forecast prevented the GARCH model from reacting to new shocks.

Switching to rolling one-step-ahead forecasting revealed the intended volatility-clustering behavior.

---

# Limitations

This project is intended as a **methodology demonstration**, not a universal air-quality forecasting system.

### Dataset limitations

* One monitoring station
* One city
* Approximately one year of observations
* No spatial cross-section

### Validation limitations

* A single held-out tail window is used
* There is only one winter period
* Full seasonal walk-forward validation is therefore not possible

### Modeling limitations

* GARCH does not predict when a pollution shock will occur
* Correlation between predicted volatility and realized absolute residuals is only **0.101**
* The dataset contains only **9,357 observations**
* No automated unit-test suite is currently included

These limitations are explicitly treated as part of the project's methodology rather than hidden from the results.

---

# Technical Stack

```text
Python 3.12
│
├── pandas
├── NumPy
├── SciPy
├── scikit-learn
├── statsmodels
├── arch
├── ucimlrepo
└── matplotlib
```

The project uses a direct AIC grid search with `statsmodels` instead of `pmdarima`, because `pmdarima` is incompatible with NumPy ≥ 2.0 due to compiled-extension issues.

