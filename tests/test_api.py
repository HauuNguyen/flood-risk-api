def test_health_and_model_info(client):
    r = client.get("/health").json()
    assert r["status"] == "ok" and r["target_year"] == 2025 and r["n_tambons"] == 300
    card = client.get("/model-info").json()
    assert card["parameters"]["feats"] == ["logit_hist", "y_lag1"] and "disclaimer" in card


def test_tambon_detail_and_404(client):
    top = client.get("/ranking?top=1").json()["items"][0]
    d = client.get(f"/tambons/{top['tb_idn']}").json()
    assert d["rank"] == 1 and len(d["history"]) == 14 and d["history"][0]["year"] == 2011
    assert client.get("/tambons/does-not-exist").status_code == 404


def test_ranking_filters(client):
    r = client.get("/ranking?top=500").json()
    assert r["total"] == 300 and r["returned"] == 300
    scores = [i["risk_prob"] for i in r["items"]]
    assert scores == sorted(scores, reverse=True)
    vh = client.get("/ranking?tier=Very High (top 5%)&top=100").json()
    assert vh["total"] == 15 and all(i["tier"] == "Very High (top 5%)" for i in vh["items"])
    prov = client.get("/ranking?province=3&top=100").json()
    assert prov["total"] == 30 and all(i["pv_idn"] == "3" for i in prov["items"])
    assert client.get("/ranking?tier=bogus").status_code == 422
    assert client.get("/ranking?top=0").status_code == 422
    assert client.get("/ranking?top=501").status_code == 422
    page2 = client.get("/ranking?top=5&offset=5").json()["items"]
    assert page2[0]["rank"] == 6


def test_score_endpoint(client):
    low = client.post("/score", json={"history": [0] * 10}).json()
    high = client.post("/score", json={"history": [1] * 10}).json()
    assert high["risk_prob"] > low["risk_prob"] and high["approx_rank_pct"] >= low["approx_rank_pct"]
    assert abs(low["hist_freq"] - (0 + 0.5) / (10 + 2)) < 1e-6 and low["flooded_last_year"] == 0
    assert client.post("/score", json={"history": []}).status_code == 422
    assert client.post("/score", json={"history": [0, 2]}).status_code == 422
    assert client.post("/score", json={"history": [0] * 61}).status_code == 422
    assert client.post("/score", json={}).status_code == 422
