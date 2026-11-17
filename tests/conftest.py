import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, pytest
from app.pipeline import train_and_export
from app.api import create_app


def make_panel(n=300, seed=7):
    rng = np.random.default_rng(seed)
    years = list(range(2011, 2025))
    place = rng.normal(0, 1, n)
    year_effect = rng.normal(-0.5, 1.3, len(years))
    frames = []
    for k, y in enumerate(years):
        p = 1 / (1 + np.exp(-(1.2 * place + year_effect[k] - 0.3)))
        fl = rng.random(n) < p
        frames.append(pd.DataFrame({
            "TB_IDN": np.arange(n), "PV_IDN": np.arange(n) % 10, "AP_IDN": np.arange(n) % 60, "year": y,
            "flood_frac": np.where(fl, rng.uniform(0.05, 0.5, n), rng.uniform(0, 0.04, n)),
            "rain_anomaly_pct": rng.normal(0, 15, n), "rain_total": 1600 + rng.normal(0, 300, n),
            "dem_mean": rng.uniform(1, 300, n)}))
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="session")
def panel():
    return make_panel()


@pytest.fixture(scope="session")
def artifact_dir(tmp_path_factory, panel):
    d = tmp_path_factory.mktemp("art")
    p = d / "panel.parquet"
    panel.to_parquet(p, index=False)
    train_and_export(str(p), str(d / "out"))
    return d / "out"


@pytest.fixture(scope="session")
def client(artifact_dir):
    from fastapi.testclient import TestClient
    with TestClient(create_app(str(artifact_dir))) as c:
        yield c
