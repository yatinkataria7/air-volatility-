Modeling
A two-part time-series pipeline on the UCI Air Quality dataset: a SARIMAX model for the expected pollution level, and a GARCH model layered on its residuals for the time-varying uncertainty around that expectation.

The point isn't just "forecast pollution." It's to produce a dynamic confidence interval — one that widens right after a real atmospheric shock and narrows again as conditions settle — because prediction errors in this system are not uniformly sized. They cluster, the same way volatility clusters in financial markets after a shock.

Expected pollution is level 5 — but a shock just hit, volatility is clustering, so the 95% interval right now is 3 to 9, not the usual 4 to 6.

Headline results
Metric	Value
PCA composite factor (PC1)	82.87% of variance across 4 gases, loadings 0.47–0.52
Mean model	SARIMAX(2,0,3) + Fourier(3 daily, 2 weekly) + temperature, AIC 17,882.0
Stationarity (ADF on PC1)	statistic −9.02, p < .0001 → stationary, d=0
One-step-ahead RMSE	0.676 vs 0.915 naive (26% error reduction)
Residual diagnostics	Ljung-Box(L1) p=0.83 · Prob(H) < .01 · kurtosis 7.16
GARCH(1,1) persistence	α=0.051, β=0.932, α+β = 0.983
GJR-GARCH asymmetry	γ = −0.088, preferred by AIC (56,181 vs 56,393)
Conditional volatility range	0.461 – 1.005 (2.18×) over the test window
95% CI coverage	Static 93.8% · GARCH 93.9% · GJR 93.7%
The honest headline
GARCH does not improve average interval coverage. Static, GARCH, and GJR all land within 0.2 points of each other (93.7–93.9%), and mean interval width barely differs. A constant-width interval can match average coverage essentially by construction, so that number proves nothing on its own — and this repo reports it that way rather than burying it.

What GARCH does buy is conditional accuracy — being right about when uncertainty is elevated:

Conditional volatility swings 2.18× across the test window (0.461 → 1.005).
At the single largest test-set shock (2005-03-24 19:00, actual 4.82 vs fitted 1.57), GARCH σ hits its test-period maximum the very next hour, then decays gradually over the following hours — textbook post-shock volatility clustering.
The static interval sits flat at 0.69 throughout that entire episode.
That distinction — marginal calibration is a tie, conditional calibration is not — is the actual finding, and it's the part worth discussing.

Run the six stages in order. Each reads the previous stage's output from data/ and writes its own:

python -m src.ingest          # fetch + clean  -> data/processed/air_quality_clean.parquet
python -m src.outliers        # glitch filter  -> air_quality_outliers_handled.parquet
python -m src.impute          # dual imputation-> air_quality_imputed.parquet
python -m src.features        # Fourier + PCA  -> air_quality_features.parquet
python -m src.mean_model      # SARIMAX        -> mean_model_residuals.parquet
python -m src.variance_model  # GARCH + report -> variance_model_output.parquet
Stage 1 downloads the dataset automatically via ucimlrepo — no manual download, no API key, no auth. Every stage prints its own diagnostics to stdout.

Runtime note: src.mean_model takes several minutes — it fits 15 candidate SARIMAX orders by maximum likelihood over 8,517 rows with 11 exogenous regressors. Every other stage is seconds.

The data
UCI Air Quality dataset (id=360) — a multi-sensor device on a road in a polluted Italian city, hourly readings.

Property	Verified value
Rows	9,357 hourly observations
Range	2004-03-10 18:00 → 2005-04-04 14:00
Time grid	Fully continuous — zero missing timestamps
Missing-value encoding	-200 sentinel
Reference analyzers missing	CO/NOx/NO2 (GT): ~17.5–18%
Sensor + weather channels missing	PT08.S*/T/RH/AH: 3.91%, in 16 contiguous outages of 1–76h
Two things here contradict the dataset's own published description, both verified directly against the data rather than assumed:

The date range. UCI's page says the data runs to February 2005. The actual final timestamp is 2005-04-04.
The date format. UCI's page describes DD/MM/YYYY. What ucimlrepo actually delivers is M/D/YYYY. Parsing with the documented format silently discards 61% of rows as unparseable.
Dropped: NMHC(GT)
Over 90% missing, in contiguous blocks — that sensor died early and never recovered. It is excluded outright rather than imputed. Imputing a mostly-dead column and then feeding it into PCA would mean PCA's reported explained variance was partly explaining a column reconstructed from the correlations of the other four — circular, and it inflates the headline number instead of measuring anything real.

Pipeline
1 · Ingest & clean — src/ingest.py
Fetches via ucimlrepo, builds a true DatetimeIndex, replaces the -200 sentinel with NaN, drops NMHC(GT), reindexes onto a continuous hourly grid, and reports per-column missingness.

2 · Outlier detection — src/outliers.py
Separates hardware glitches (discard) from real pollution events (keep — GARCH needs them).

A textbook Hampel filter compares each point to a centered rolling median. That fails badly here: the window spans both a daily peak and trough, so the steepest part of temperature's normal diurnal curve reads as anomalous. This repo instead flags deviation from each point's immediate neighbor-interpolation (t−1, t+1), which is blind to slow cycles but still catches genuine spikes. A run-length check then keeps any deviation persisting beyond 2 hours as a real event.

3 · Dual imputation — src/impute.py
Short gaps (≤2h): PCHIP (shape-preserving piecewise cubic Hermite) — cannot overshoot the range of its neighbors.
Long gaps (3h+, up to 76h): IterativeImputer, regressing each channel on the others. T/RH/AH rarely drop out simultaneously with the gas sensors, so they carry real information into the gas columns' long outages.
Post-clip: all physically non-negative columns clipped at 0.
4 · Features + PCA — src/features.py
Fourier harmonics (3 daily, 2 weekly) as exogenous regressors, so the ARIMA terms aren't spent relearning the basic commuter rhythm. Then StandardScaler + PCA over the four true reference concentrations — CO(GT), C6H6(GT), NOx(GT), NO2(GT).

Deliberately excluded from PCA: the PT08.S* metal-oxide channels (arbitrary-unit resistance readings, not concentrations — mixing units would make the composite incoherent) and any ozone channel (no O3(GT) ground truth exists; ozone is photochemically produced and often anti-correlated with NOx via titration, so it doesn't share the shared-combustion-source story PC1 is built on).

5 · Mean equation — src/mean_model.py
ADF test → PC1 is already stationary (d=0). AIC grid search over p,q ∈ 0..3 selects SARIMAX(2,0,3).

Evaluated on a single held-out 5-week tail window via rolling one-step-ahead forecasts (.append(refit=False) — parameters frozen at the training MLE, only the Kalman state advances as real observations arrive). This is what a deployed system does hourly; it is not an 840-step static forecast from one origin.

6 · Variance equation — src/variance_model.py
GARCH(1,1) and GJR-GARCH(1,1,1) fit on the SARIMAX residuals via arch. Parameters estimated on in-sample residuals only; test-window conditional variance comes from running the GARCH recursion forward through the realized one-step-ahead residuals, with no re-estimation on test data. Outputs the coverage comparison and the shock case study.

Five bugs found by verifying output, not by reading code
Each of these ran without raising an error and produced plausible-looking numbers. All were caught by checking results against ground truth or physical reality.

#	Bug	How it surfaced	Fix
1	61% of rows silently dropped	Parsed range came out 2004-01 → 2005-12, wider than possible; most rows became NaT	UCI's docs say DD/MM/YYYY, the data is M/D/YYYY. Verified against the known 2004-03-10 start.
2	Outlier filter flagged the weather	~20% of T/RH flagged, clustered at 03:00–06:00 and 14:00–17:00 — the steepest slopes of the daily curve	Rolling-median comparison → neighbor-interpolation deviation. Flags now uniform across hours.
3	Negative pollution concentrations	CO down to −3.7 mg/m³, NOx to −147 ppb	Cubic spline overshoots near sharp gap edges → switched to PCHIP.
4	Negative concentrations, again — different cause	82 more negative NOx values (to −99 ppb) survived the PCHIP fix	IterativeImputer's BayesianRidge has no non-negativity constraint → post-imputation clip.
5	Backtest design starved the variance model	GARCH σ decayed to a flat constant almost immediately, showing nothing	An 840-step static forecast gives GARCH nothing to react to → rolling one-step-ahead. This is what produced the clustering result.
Bug 4 is the instructive one: bug 3's fix was correct and verified, and the same class of error was still present from an entirely separate cause one stage later.

Scope, limitations, and what this project does not claim
Stated up front rather than left for a reader to find:

One sensor, one city, one year. No spatial cross-section — there is no second station, so nothing here demonstrates spatial spillover.
No true walk-forward cross-validation. The dataset contains exactly one winter. A walk-forward across seasons would be fiction; it could only show the model fits the one inversion period it already saw. A single held-out tail window is used instead, and labeled as such.
PC1 explains 82.9%, not the >85% often quoted for this pipeline. Reported as measured.
GARCH does not predict shocks. Correlation between predicted σ(t) and realized |residual(t)| is 0.101. GARCH forecasts how volatile the near future is given what has already happened — it never claims a shock is coming. That weak correlation is expected, not a defect.
No automated test suite. Verification here was done by checking each stage's real output against ground truth and physical constraints (documented above and in stdout), not by unit tests.
9,357 rows is small. This is a methodology demonstration on a fully checkable dataset, not a scale exercise.
One finding worth flagging
GJR-GARCH gives γ = −0.088: an unexpectedly high pollution reading raises near-term volatility more than an unexpectedly low one. That is the opposite sign from the equity-market "leverage effect" GJR-GARCH was designed around. It is physically plausible — a positive surprise may signal the onset of a persistent trapping inversion, while a negative surprise is more often a transient gust — but it is reported as a domain-specific empirical result rather than forced to match the financial convention.

Repository layout
├── src/
│   ├── ingest.py           # Stage 1 — fetch, clean, sentinel handling
│   ├── outliers.py         # Stage 2 — glitch vs. real-event separation
│   ├── impute.py           # Stage 3 — PCHIP + IterativeImputer
│   ├── features.py         # Stage 4 — Fourier terms + PCA
│   ├── mean_model.py       # Stage 5 — SARIMAX + AIC grid search
│   └── variance_model.py   # Stage 6 — GARCH / GJR-GARCH + evaluation
├── PLAN.md                 # Design decisions, made before implementation
├── REPORT.md               # Full results writeup
├── INTERVIEW_PREP.md       # Plain-language walkthrough + anticipated Q&A
└── requirements.txt        # Pinned, verified-working versions
data/ is gitignored — every artifact in it is regenerated by rerunning the pipeline, and stage 1 fetches the source data automatically.

Environment
Python 3.12.10, numpy 2.5.1, pandas 3.0.5. Exact versions in requirements.txt.

pmdarima — the conventional auto-ARIMA package, and what most tutorials for this pipeline reach for — is not used: it is incompatible with numpy ≥ 2.0 due to a binary mismatch in its compiled extensions. src/mean_model.py implements the equivalent AIC grid search directly against statsmodels, which is what auto_arima does internally anyway.
