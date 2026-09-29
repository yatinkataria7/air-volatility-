"""Stage 4: Fourier calendar features + PCA composite smog factor.

PCA runs on the four true reference gas concentrations only -- CO(GT), C6H6(GT), NOx(GT), NO2(GT) --
not the PT08.S* metal-oxide sensor proxies (arbitrary-unit resistance readings, not concentrations)
and not an ozone channel (there's no O3(GT) ground truth in this dataset, only the PT08.S5(O3) proxy;
ozone is photochemically produced and often anti-correlated with NOx via titration, so it doesn't
share the "same combustion source" story the brief's PCA rationale is built on). This is a deliberate
scope choice, not the brief's original "all 6 gas sensors" framing -- NMHC(GT) is already dropped
(stage 1), and mixing sensor-voltage units with concentration units into one PCA would make the
"composite smog factor" story incoherent.
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

IN_PATH = "data/processed/air_quality_imputed.parquet"
OUT_PATH = "data/processed/air_quality_features.parquet"

PCA_COLS = ["CO(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)"]
DAILY_HARMONICS = 3
WEEKLY_HARMONICS = 2


def add_fourier_terms(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    hour = df.index.hour.values.astype(float)
    dow_hour = df.index.dayofweek.values.astype(float) * 24 + hour

    for k in range(1, DAILY_HARMONICS + 1):
        out[f"fourier_daily_sin_{k}"] = np.sin(2 * np.pi * k * hour / 24)
        out[f"fourier_daily_cos_{k}"] = np.cos(2 * np.pi * k * hour / 24)
    for k in range(1, WEEKLY_HARMONICS + 1):
        out[f"fourier_weekly_sin_{k}"] = np.sin(2 * np.pi * k * dow_hour / 168)
        out[f"fourier_weekly_cos_{k}"] = np.cos(2 * np.pi * k * dow_hour / 168)
    return out


def run_pca(df: pd.DataFrame) -> tuple[pd.DataFrame, PCA, StandardScaler]:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[PCA_COLS])
    pca = PCA(n_components=len(PCA_COLS), random_state=0)
    components = pca.fit_transform(scaled)
    for i in range(components.shape[1]):
        df[f"PC{i+1}"] = components[:, i]
    return df, pca, scaler


def main():
    df = pd.read_parquet(IN_PATH)
    df = add_fourier_terms(df)
    df, pca, scaler = run_pca(df)

    print(f"PCA input columns: {PCA_COLS}")
    print("Explained variance ratio by component:")
    for i, ratio in enumerate(pca.explained_variance_ratio_):
        print(f"  PC{i+1}: {ratio*100:.2f}%  (cumulative: {pca.explained_variance_ratio_[:i+1].sum()*100:.2f}%)")

    print("\nPC1 loadings (which gases drive the composite smog factor):")
    loadings = pd.Series(pca.components_[0], index=PCA_COLS)
    print(loadings.round(3).to_string())

    df.to_parquet(OUT_PATH)
    print(f"\nSaved to {OUT_PATH}, shape {df.shape}")


if __name__ == "__main__":
    main()
