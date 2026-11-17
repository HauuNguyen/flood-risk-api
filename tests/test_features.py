import numpy as np, pandas as pd
from app.features import add_future_year, build_features, smoothed_hist_freq

EX_ANTE = ["hist_freq", "logit_hist", "y_lag1", "y_lag2", "y_lag3", "frac_lag1", "roll_mean_3y", "roll_std_3y",
           "rain_anomaly_pct_lag1", "ap_rate_lag1", "pv_rate_lag1"]


def test_smoothed_frequency_formula():
    # 3 earlier years, 1 flooded, prior 0.25 with strength 2  ->  (1 + 0.5) / (3 + 2) = 0.3
    assert abs(smoothed_hist_freq(1, 3) - 0.3) < 1e-12
    df = pd.DataFrame({"TB_IDN": [1] * 4, "PV_IDN": 1, "AP_IDN": 1, "year": [2011, 2012, 2013, 2014],
                       "flood_frac": [0.10, 0.0, 0.0, 0.2], "rain_anomaly_pct": 0.0})
    f = build_features(df).set_index("year")
    assert abs(f.loc[2014, "hist_freq"] - 0.3) < 1e-12
    assert f.loc[2014, "y_lag1"] == 0 and f.loc[2013, "y_lag2"] == 1


def test_features_of_year_t_never_see_year_t(panel):
    base = build_features(panel)
    t = 2020
    changed = panel.copy()
    m = changed["year"] == t
    changed.loc[m, "flood_frac"] = 1.0 - changed.loc[m, "flood_frac"]       # alter the outcome AND rainfall of year t
    changed.loc[m, "rain_anomaly_pct"] += 50
    new = build_features(changed)
    a = base[base.year == t].sort_values("TB_IDN")[EX_ANTE].reset_index(drop=True)
    b = new[new.year == t].sort_values("TB_IDN")[EX_ANTE].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)                                       # year-t features unchanged
    a1 = base[base.year == t + 1].sort_values("TB_IDN")["y_lag1"].reset_index(drop=True)
    b1 = new[new.year == t + 1].sort_values("TB_IDN")["y_lag1"].reset_index(drop=True)
    assert not a1.equals(b1)                                                  # but the next year does react


def test_future_year_row(panel):
    ext, target = add_future_year(panel)
    assert target == 2025 and ext["flood_frac"].isna().sum() == panel["TB_IDN"].nunique()
    f = build_features(ext)
    nxt = f[f.year == 2025].set_index("TB_IDN")
    last = f[f.year == 2024].set_index("TB_IDN")
    assert (nxt["y_lag1"] == last["y"]).all()
    assert nxt["hist_freq"].notna().all()
