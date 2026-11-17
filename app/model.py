"""Core model: standardised logistic regression on [logit_hist, y_lag1] + Platt scaling.

Training uses scikit-learn; the fitted parameters are exported to plain JSON and scoring is pure numpy,
so the deployed service has no pickle / scikit-learn version dependency and the model is fully transparent.
"""
import json
from dataclasses import dataclass, asdict
import numpy as np

CLIP = 1e-6


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _logit_clip(p):
    p = np.clip(p, CLIP, 1 - CLIP)
    return np.log(p / (1 - p))


@dataclass
class CoreModel:
    feats: list
    mean: list
    scale: list
    coef: list
    intercept: float
    platt_coef: float
    platt_intercept: float

    def raw_proba(self, X):
        X = np.asarray(X, dtype=float)
        z = ((X - np.array(self.mean)) / np.array(self.scale)) @ np.array(self.coef) + self.intercept
        return _sigmoid(z)

    def predict_proba(self, X):
        """Platt-calibrated score. Treat as an approximate probability; use it mainly to RANK Tambons."""
        z = _logit_clip(self.raw_proba(X))
        return _sigmoid(self.platt_coef * z + self.platt_intercept)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls.from_dict(json.load(f))


def fit_core(train_df, calib_df, feats, target="y"):
    """Fit scaler + logistic regression on train_df, Platt scaling on calib_df (both ex-ante frames)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(train_df[feats])
    lr = LogisticRegression(C=0.5, max_iter=2000).fit(sc.transform(train_df[feats]), train_df[target])
    raw = lr.predict_proba(sc.transform(calib_df[feats]))[:, 1]
    pl = LogisticRegression(C=1.0, max_iter=1000).fit(_logit_clip(raw).reshape(-1, 1), calib_df[target])
    return CoreModel(feats=list(feats), mean=sc.mean_.tolist(), scale=sc.scale_.tolist(),
                     coef=lr.coef_[0].tolist(), intercept=float(lr.intercept_[0]),
                     platt_coef=float(pl.coef_[0][0]), platt_intercept=float(pl.intercept_[0]))
