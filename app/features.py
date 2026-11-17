"""Ex-ante feature construction, ported 1:1 from notebook section 3.2.

Every feature of year t uses information from years < t only. Rows whose flood_frac is NaN
(the future year to be ranked) are treated as y = 0 but never leak into their own features.
"""
import numpy as np
import pandas as pd

FLOOD_THRESHOLD = 0.05
PRIOR_MEAN = 0.25          # shrinkage prior for the smoothed historical flood frequency
PRIOR_STRENGTH = 2.0
FEATS_CORE = ["logit_hist", "y_lag1"]
REQUIRED_COLUMNS = ["TB_IDN", "PV_IDN", "AP_IDN", "year", "flood_frac", "rain_anomaly_pct"]


def logit(p):
    p = np.asarray(p, dtype=float)
    return np.log(p / (1.0 - p))


def smoothed_hist_freq(n_flooded, n_years, prior_mean=PRIOR_MEAN, prior_strength=PRIOR_STRENGTH):
    """Smoothed share of earlier years in which a Tambon was flooded."""
    return (n_flooded + prior_mean * prior_strength) / (n_years + prior_strength)


def logit_hist(hist_freq):
    return logit(np.clip(hist_freq, 0.01, 0.99))


def build_features(df_panel, thr=FLOOD_THRESHOLD, prior_mean=PRIOR_MEAN, prior_strength=PRIOR_STRENGTH):
    df = df_panel.sort_values(["TB_IDN", "year"]).copy()
    df["y"] = (df["flood_frac"] > thr).astype(int)
    g = df.groupby("TB_IDN", sort=False)
    for k in (1, 2, 3):
        df[f"y_lag{k}"] = g["y"].shift(k)
    df["frac_lag1"] = g["flood_frac"].shift(1)
    df["roll_mean_3y"] = g["y"].transform(lambda s: s.shift(1).rolling(3, min_periods=2).mean())
    df["roll_std_3y"] = g["y"].transform(lambda s: s.shift(1).rolling(3, min_periods=2).std())
    n_prev, s_prev = g.cumcount(), g["y"].cumsum() - df["y"]
    df["hist_freq"] = smoothed_hist_freq(s_prev, n_prev, prior_mean, prior_strength)
    df["logit_hist"] = logit_hist(df["hist_freq"])
    df["rain_anomaly_pct_lag1"] = g["rain_anomaly_pct"].shift(1)
    for key, nm in (("AP_IDN", "ap"), ("PV_IDN", "pv")):
        rate = df.groupby([key, "year"])["y"].mean().sort_index()
        lag = rate.groupby(level=0).shift(1).rename(f"{nm}_rate_lag1").reset_index()
        df = df.merge(lag, on=[key, "year"], how="left")
    return df


def add_future_year(df_panel):
    """Append one row per Tambon for the year after the last observed year (outcome unknown)."""
    last = int(df_panel["year"].max())
    fut = df_panel[df_panel["year"] == last].copy()
    fut["year"] = last + 1
    fut["flood_frac"] = np.nan
    return pd.concat([df_panel, fut], ignore_index=True), last + 1


def validate_panel(df):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"panel is missing columns: {missing}")
    if df.duplicated(["TB_IDN", "year"]).any():
        raise ValueError("duplicate Tambon-year rows")
    if not df["flood_frac"].dropna().between(0, 1.0001).all():
        raise ValueError("flood_frac outside [0, 1]")
