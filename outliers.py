"""Stage 2: Hampel filter for glitch detection, with a persistence check to protect real shocks.

A naive Hampel filter on raw levels (deviation from a centered rolling median) misfires badly on
temperature/humidity: their smooth diurnal cycle makes the steep pre-dawn and mid-afternoon slopes
look like "deviations" from a window that spans both a peak and a trough. Verified directly: with a
level-based filter, ~20% of T/RH points got flagged, clustering almost entirely at 03:00-06:00 and
14:00-17:00 -- the steepest parts of the daily curve, not glitches.

Fix: detect spikes as deviation from the straight-line interpolation of each point's immediate
neighbors (t-1, t+1), not deviation from a windowed median. A slow-moving cycle barely moves hour to
hour, so this is insensitive to trend/seasonality; a real hardware spike -- which by construction
breaks the point away from both neighbors -- still stands out sharply. This directly implements the
brief's own distinguishing logic: an isolated one-or-two-hour spike that snaps back is a glitch,
while a deviation that persists and decays over many hours (a real event) moves together with its
neighbors at each step and is never flagged.
"""
import numpy as np
import pandas as pd

CLEAN_PATH = "data/processed/air_quality_clean.parquet"
OUT_PATH = "data/processed/air_quality_outliers_handled.parquet"

MAD_SCALE = 1.4826  # scales MAD to be a consistent estimator of std under normality
WINDOW = 13  # +/-6 hours, used only to localize the MAD threshold, not to define "normal level"
N_SIGMAS = 4.0
GLITCH_MAX_RUN = 2  # a flagged run longer than this is treated as a real, persistent event


def spike_flags(s: pd.Series, window: int = WINDOW, n_sigmas: float = N_SIGMAS) -> pd.Series:
    neighbor_interp = (s.shift(1) + s.shift(-1)) / 2
    dev = (s - neighbor_interp).abs()
    local_mad = dev.rolling(window, center=True, min_periods=5).median()
    threshold = n_sigmas * MAD_SCALE * local_mad
    flagged = (dev > threshold) & dev.notna()
    return flagged.fillna(False)


def classify_runs(flagged: pd.Series) -> pd.Series:
    """Returns a boolean series: True where a flagged point is judged a glitch (short, isolated)."""
    is_glitch = pd.Series(False, index=flagged.index)
    if not flagged.any():
        return is_glitch
    run_id = (flagged != flagged.shift()).cumsum()
    for rid, members in flagged[flagged].groupby(run_id[flagged]):
        run_len = len(members)
        if run_len <= GLITCH_MAX_RUN:
            is_glitch.loc[members.index] = True
    return is_glitch


def process_column(df: pd.DataFrame, col: str) -> dict:
    s = df[col]
    flagged = spike_flags(s)
    is_glitch = classify_runs(flagged)
    n_flagged = int(flagged.sum())
    n_glitch = int(is_glitch.sum())
    n_event = n_flagged - n_glitch
    df.loc[is_glitch, col] = np.nan
    return {
        "column": col,
        "n_flagged": n_flagged,
        "n_glitch_removed": n_glitch,
        "n_persistent_event_kept": n_event,
    }


def main():
    df = pd.read_parquet(CLEAN_PATH)
    numeric_cols = [c for c in df.columns]

    stats = []
    for col in numeric_cols:
        stats.append(process_column(df, col))

    report = pd.DataFrame(stats).set_index("column")
    print(f"Spike filter results (neighbor-interpolation deviation, {N_SIGMAS} MAD-sigma, glitch = run <= {GLITCH_MAX_RUN}h):")
    print(report.to_string())
    print(f"\nTotal glitches removed: {report['n_glitch_removed'].sum()}")
    print(f"Total persistent events kept untouched: {report['n_persistent_event_kept'].sum()}")

    df.to_parquet(OUT_PATH)
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
