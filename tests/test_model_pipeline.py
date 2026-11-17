import json
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from app.features import FEATS_CORE, build_features
from app.model import CoreModel, fit_core


def test_numpy_scoring_matches_sklearn(panel):
    f = build_features(panel)
    d = f[(f.year >= 2012)].dropna(subset=FEATS_CORE)
    tr, ca = d[d.year <= 2019], d[d.year > 2019]
    m = fit_core(tr, ca, FEATS_CORE)
    sc = StandardScaler().fit(tr[FEATS_CORE])
    lr = LogisticRegression(C=0.5, max_iter=2000).fit(sc.transform(tr[FEATS_CORE]), tr["y"])
    expected = lr.predict_proba(sc.transform(ca[FEATS_CORE]))[:, 1]
    assert np.allclose(m.raw_proba(ca[FEATS_CORE].to_numpy()), expected, atol=1e-9)


def test_json_roundtrip(tmp_path, panel):
    f = build_features(panel); d = f.dropna(subset=FEATS_CORE)
    m = fit_core(d[d.year <= 2019], d[d.year > 2019], FEATS_CORE)
    m.save(tmp_path / "m.json")
    m2 = CoreModel.load(tmp_path / "m.json")
    X = d[FEATS_CORE].to_numpy()[:50]
    assert np.allclose(m.predict_proba(X), m2.predict_proba(X))


def test_artifacts_and_tiers(artifact_dir):
    for name in ("model.json", "ranking.parquet", "history.parquet", "model_card.json"):
        assert (artifact_dir / name).exists()
    r = pd.read_parquet(artifact_dir / "ranking.parquet")
    assert r["TB_IDN"].is_unique and r["risk_prob"].between(0, 1).all()
    assert (r["rank"].to_numpy() == np.arange(1, len(r) + 1)).all()
    shares = r["tier"].value_counts(normalize=True)
    assert abs(shares["Low (bottom 50%)"] - 0.50) < 0.01 and abs(shares["Very High (top 5%)"] - 0.05) < 0.01
    card = json.loads((artifact_dir / "model_card.json").read_text())
    assert card["target_year"] == 2025 and card["evaluation"]["within_year_auc"]["climatology"] > 0.5
