"""Offline pipeline: panel -> trained core model + ranking table + history + model card.

    python -m app.pipeline --panel panel_v2.parquet --out artifacts/
"""
import argparse, json, os
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import __version__
from .features import FEATS_CORE, FLOOD_THRESHOLD, add_future_year, build_features, validate_panel
from .model import fit_core

from .tiers import DISCLAIMER, TIER_BINS, TIERS


def assign_tiers(score: pd.Series):
    pct = score.rank(pct=True, method="first")
    return pct, pd.cut(pct, TIER_BINS, labels=TIERS)


def within_year_auc(df, score_col):
    aucs = [roc_auc_score(g["y"], g[score_col]) for _, g in df.groupby("year") if g["y"].nunique() > 1]
    return float(np.mean(aucs))


def final_split_evaluation(f_all):
    """Same protocol as notebook 3.4: train 2012-19, calibrate 2020-22, test 2023-24."""
    ok = lambda lo, hi: f_all[(f_all.year >= lo) & (f_all.year <= hi)].dropna(subset=FEATS_CORE + ["y"])
    tr, va, te = ok(2012, 2019), ok(2020, 2022), ok(2023, 2024)
    if min(len(tr), len(va), len(te)) == 0:
        return None
    m = fit_core(tr, va, FEATS_CORE)
    te = te.assign(core_raw=m.raw_proba(te[FEATS_CORE]), core_cal=m.predict_proba(te[FEATS_CORE]))
    return {
        "protocol": "train 2012-2019, calibrate 2020-2022, test 2023-2024",
        "test_years": [2023, 2024], "n_test_rows": int(len(te)),
        "within_year_auc": {"core_model": round(within_year_auc(te, "core_raw"), 4),
                            "climatology": round(within_year_auc(te, "hist_freq"), 4)},
        "pooled_auc": {"core_model": round(float(roc_auc_score(te["y"], te["core_raw"])), 4),
                       "climatology": round(float(roc_auc_score(te["y"], te["hist_freq"])), 4)},
    }


def train_and_export(panel_path, out_dir, flood_threshold=FLOOD_THRESHOLD):
    df = pd.read_parquet(panel_path)
    validate_panel(df)
    os.makedirs(out_dir, exist_ok=True)
    ext, target_year = add_future_year(df)
    last = target_year - 1
    f_all = build_features(ext, thr=flood_threshold)

    fit_df = f_all[(f_all.year >= 2012) & (f_all.year <= target_year - 3)].dropna(subset=FEATS_CORE + ["y"])
    cal_df = f_all[f_all.year.isin([target_year - 2, target_year - 1])].dropna(subset=FEATS_CORE + ["y"])
    model = fit_core(fit_df, cal_df, FEATS_CORE)

    nxt = f_all[f_all.year == target_year].dropna(subset=FEATS_CORE).copy()
    nxt["risk_prob"] = model.predict_proba(nxt[FEATS_CORE])
    nxt["rank_pct"], nxt["tier"] = assign_tiers(nxt["risk_prob"])
    nxt["tier"] = nxt["tier"].astype(str)
    cols = ["TB_IDN", "PV_IDN", "AP_IDN", "risk_prob", "rank_pct", "tier", "hist_freq", "y_lag1"]
    ranking = nxt[cols].sort_values("risk_prob", ascending=False).reset_index(drop=True)
    ranking["rank"] = np.arange(1, len(ranking) + 1)
    ranking.to_parquet(os.path.join(out_dir, "ranking.parquet"), index=False)

    hist = f_all[f_all.year <= last][["TB_IDN", "year", "flood_frac", "y"]]
    hist.to_parquet(os.path.join(out_dir, "history.parquet"), index=False)
    model.save(os.path.join(out_dir, "model.json"))

    card = {
        "model_version": __version__, "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "task": "One-year-ahead ranking of Tambon flood-proneness (label: flooded share of area > %.0f%%)" % (100 * flood_threshold),
        "target_year": int(target_year), "data_years": [int(df.year.min()), int(last)], "n_tambons": int(len(ranking)),
        "inputs": {"logit_hist": "logit of the smoothed share of earlier years in which the Tambon was flooded",
                   "y_lag1": "1 if the Tambon was flooded in the previous year, else 0"},
        "training_window": {"fit_years": [2012, int(target_year - 3)], "calibration_years": [int(target_year - 2), int(target_year - 1)]},
        "parameters": model.to_dict(),
        "evaluation": final_split_evaluation(f_all),
        "tiers": TIERS, "disclaimer": DISCLAIMER,
        "known_limitations": ["Skill over climatology is small and within year-to-year noise.",
                              "Scores are approximate probabilities; use them to rank.",
                              "Flood labels come from an annual satellite-derived product (AIT-GIC)."],
    }
    with open(os.path.join(out_dir, "model_card.json"), "w") as f:
        json.dump(card, f, indent=2)
    return card


def main_cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True)
    ap.add_argument("--out", default="artifacts")
    a = ap.parse_args()
    c = train_and_export(a.panel, a.out)
    print(json.dumps({k: c[k] for k in ("target_year", "n_tambons", "evaluation")}, indent=2))


if __name__ == "__main__":
    main_cli()
