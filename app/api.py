"""FastAPI service: serves the pre-computed Tambon flood-proneness ranking and the transparent core model.

Run:  uvicorn app.api:app --host 0.0.0.0 --port 8000      (ARTIFACT_DIR selects the artifact folder)
Only numpy / pandas / fastapi are needed at serving time.
"""
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from . import __version__
from .features import logit_hist, smoothed_hist_freq
from .model import CoreModel
from .tiers import DISCLAIMER, TIERS, tier_from_pct

log = logging.getLogger("flood-risk-api")


class RankItem(BaseModel):
    rank: int
    tb_idn: str
    pv_idn: str
    ap_idn: str
    risk_prob: float
    tier: str
    hist_freq: float
    flooded_last_year: int


class HistoryItem(BaseModel):
    year: int
    flood_frac: float
    flooded: int


class TambonDetail(RankItem):
    target_year: int
    history: List[HistoryItem]
    disclaimer: str


class RankingResponse(BaseModel):
    target_year: int
    total: int
    returned: int
    items: List[RankItem]
    disclaimer: str


class ScoreRequest(BaseModel):
    history: List[int] = Field(min_length=1, max_length=60,
                               description="Flood flags of earlier years, oldest first (1 = flooded above the label threshold).")

    @field_validator("history")
    @classmethod
    def only_binary(cls, v):
        if any(x not in (0, 1) for x in v):
            raise ValueError("history must contain only 0 or 1")
        return v


class ScoreResponse(BaseModel):
    n_years: int
    hist_freq: float
    flooded_last_year: int
    risk_prob: float
    approx_rank_pct: float
    tier: str
    disclaimer: str


def load_artifacts(d: Path) -> dict:
    ranking = pd.read_parquet(d / "ranking.parquet")
    history = pd.read_parquet(d / "history.parquet")
    for df in (ranking, history):
        df["TB_IDN"] = df["TB_IDN"].astype(str)
    ranking["PV_IDN"] = ranking["PV_IDN"].astype(str)
    ranking["AP_IDN"] = ranking["AP_IDN"].astype(str)
    return {
        "model": CoreModel.load(d / "model.json"),
        "card": json.loads((d / "model_card.json").read_text()),
        "ranking": ranking,
        "by_id": ranking.set_index("TB_IDN"),
        "history": {k: g.sort_values("year") for k, g in history.groupby("TB_IDN")},
        "scores_sorted": np.sort(ranking["risk_prob"].to_numpy()),
    }


def _item(row) -> dict:
    return dict(rank=int(row["rank"]), tb_idn=str(row["TB_IDN"]), pv_idn=str(row["PV_IDN"]), ap_idn=str(row["AP_IDN"]),
                risk_prob=round(float(row["risk_prob"]), 6), tier=str(row["tier"]),
                hist_freq=round(float(row["hist_freq"]), 6), flooded_last_year=int(row["y_lag1"]))


def create_app(artifact_dir: Optional[str] = None) -> FastAPI:
    adir = Path(artifact_dir or os.getenv("ARTIFACT_DIR", "artifacts"))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.art = load_artifacts(adir)
        log.info("loaded artifacts from %s (target year %s)", adir, app.state.art["card"]["target_year"])
        yield

    app = FastAPI(title="Tambon Flood-Proneness Ranking API", version=__version__, lifespan=lifespan,
                  description="One-year-ahead ranking of flood-proneness for 6,363 Thai Tambons. " + DISCLAIMER)
    origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"], allow_headers=["*"])

    def art(request: Request) -> dict:
        return request.app.state.art

    @app.get("/health", tags=["ops"])
    def health(request: Request):
        a = art(request)
        return {"status": "ok", "model_version": a["card"]["model_version"],
                "target_year": a["card"]["target_year"], "n_tambons": int(len(a["ranking"]))}

    @app.get("/model-info", tags=["model"])
    def model_info(request: Request):
        """Model card: inputs, coefficients, training windows, evaluation and limitations."""
        return art(request)["card"]

    @app.get("/tambons/{tb_idn}", response_model=TambonDetail, tags=["ranking"])
    def tambon(tb_idn: str, request: Request):
        a = art(request)
        if tb_idn not in a["by_id"].index:
            raise HTTPException(status_code=404, detail=f"Tambon {tb_idn} not found")
        row = a["by_id"].loc[tb_idn].to_dict(); row["TB_IDN"] = tb_idn
        h = a["history"].get(tb_idn)
        hist = [] if h is None else [HistoryItem(year=int(r.year), flood_frac=round(float(r.flood_frac), 6), flooded=int(r.y))
                                     for r in h.itertuples()]
        return TambonDetail(**_item(row), target_year=a["card"]["target_year"], history=hist, disclaimer=DISCLAIMER)

    @app.get("/ranking", response_model=RankingResponse, tags=["ranking"])
    def ranking(request: Request, top: int = Query(20, ge=1, le=500), offset: int = Query(0, ge=0),
                tier: Optional[str] = Query(None, description="One of: " + " | ".join(TIERS)),
                province: Optional[str] = Query(None, description="Province code (PV_IDN)")):
        a = art(request)
        df = a["ranking"]
        if tier is not None:
            if tier not in TIERS:
                raise HTTPException(status_code=422, detail=f"tier must be one of {TIERS}")
            df = df[df["tier"] == tier]
        if province is not None:
            df = df[df["PV_IDN"] == province]
        page = df.iloc[offset:offset + top]
        return RankingResponse(target_year=a["card"]["target_year"], total=int(len(df)), returned=int(len(page)),
                               items=[_item(r) for _, r in page.iterrows()], disclaimer=DISCLAIMER)

    @app.post("/score", response_model=ScoreResponse, tags=["model"])
    def score(body: ScoreRequest, request: Request):
        """Score any flood history with the core model and place it in the current national ranking."""
        a = art(request)
        h = body.history
        hf = float(smoothed_hist_freq(sum(h), len(h)))
        x = np.array([[float(logit_hist(hf)), float(h[-1])]])
        p = float(a["model"].predict_proba(x)[0])
        pct = float(np.searchsorted(a["scores_sorted"], p, side="right") / len(a["scores_sorted"]))
        return ScoreResponse(n_years=len(h), hist_freq=round(hf, 6), flooded_last_year=int(h[-1]), risk_prob=round(p, 6),
                             approx_rank_pct=round(pct, 4), tier=tier_from_pct(pct), disclaimer=DISCLAIMER)

    return app


app = create_app()
