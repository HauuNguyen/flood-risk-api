# Tambon Flood-Proneness Ranking API

One-year-ahead **ranking** of flood-proneness for 6,363 Thai Tambons, served by FastAPI.
The model is the project's *core model*: standardised logistic regression on two inputs
(`logit_hist`, `y_lag1`) with Platt scaling. Logic is ported 1:1 from notebook sections 3.2 (features),
3.4 (final split) and 3.7 (next-year map).

> Output is a ranking of long-run flood-proneness, **not** a forecast of a specific year's flood.
> Skill over the climatology baseline is small (see `/model-info`).

## Layout
```
app/features.py   ex-ante features (year t uses years < t only)
app/model.py      CoreModel: sklearn fit -> JSON parameters; numpy-only scoring
app/pipeline.py   panel -> model.json, ranking.parquet, history.parquet, model_card.json
app/api.py        FastAPI service (needs only numpy / pandas / fastapi)
scripts/train.py  CLI wrapper
tests/            pytest suite (leakage test, parity with scikit-learn, API tests)
Dockerfile        serving image (artifacts baked in)
```

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

# 1) build artifacts from your real panel (the file produced by Part 1 of the notebook)
python scripts/train.py --panel /path/to/panel_v2.parquet --out artifacts

# 2) run tests, then the API
pytest -q
uvicorn app.api:app --reload --port 8000        # docs at http://localhost:8000/docs
```
Required panel columns: `TB_IDN, PV_IDN, AP_IDN, year, flood_frac, rain_anomaly_pct`.

## Endpoints
| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness + model version, target year, number of Tambons (use for load-balancer health checks) |
| GET | `/model-info` | model card: inputs, coefficients, training windows, evaluation, limitations |
| GET | `/ranking?top=20&offset=0&tier=&province=` | ranked Tambons (top ≤ 500; filter by tier or province code) |
| GET | `/tambons/{tb_idn}` | one Tambon: rank, tier, score, flood history by year |
| POST | `/score` | body `{"history":[0,1,0,...]}` (oldest first) -> score and tier for any flood history |

Tiers are defined by **rank**: Low (bottom 50%), Moderate, High (top 20%), Very High (top 5%).
Scores take few distinct values, so Tambons can tie; order inside a tie is arbitrary.

## Yearly update
When a new year of flood labels is added to the panel, re-run `scripts/train.py` and redeploy.
The target year is always `max(year) + 1`.

## Docker
```bash
python scripts/train.py --panel /path/to/panel_v2.parquet --out artifacts   # artifacts must exist first
docker build -t flood-risk-api .
docker run -p 8000:8000 flood-risk-api
curl localhost:8000/health
```
Environment variables: `ARTIFACT_DIR` (default `artifacts`), `CORS_ORIGINS` (comma-separated, default `*`).

## Deploying to AWS

### Option A: EC2 + Docker (works in most student accounts)
1. Build the artifacts locally from the real panel (`scripts/train.py`) and run `pytest -q`.
2. EC2 console -> Launch instance: Amazon Linux 2023, t3.micro or t3.small, create a key pair (`.pem`).
   Security group: allow SSH (22) from *your IP only* and HTTP (80) from anywhere.
3. Copy the project and install Docker on the instance:
```bash
scp -i key.pem -r flood-risk-api ec2-user@<PUBLIC_IP>:~
ssh -i key.pem ec2-user@<PUBLIC_IP>
sudo dnf install -y docker && sudo systemctl enable --now docker && sudo usermod -aG docker ec2-user
exit && ssh -i key.pem ec2-user@<PUBLIC_IP>          # reconnect so the docker group applies
cd flood-risk-api && docker build -t flood-risk-api . && docker run -d --restart unless-stopped -p 80:8000 --name flood-api flood-risk-api
```
4. Check: `curl http://<PUBLIC_IP>/health` and open `http://<PUBLIC_IP>/docs`.
5. Stop the instance after the demo to avoid charges.

### Option B: Amazon ECR + AWS App Runner (managed)
```bash
aws ecr create-repository --repository-name flood-risk-api --region <REGION>
aws ecr get-login-password --region <REGION> | docker login --username AWS --password-stdin <ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com
docker build --platform linux/amd64 -t flood-risk-api .     # required on Apple-silicon Macs
docker tag flood-risk-api:latest <ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com/flood-risk-api:latest
docker push <ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com/flood-risk-api:latest
```
Then App Runner console -> Create service -> Container registry (Amazon ECR) -> select the image, port 8000,
health check path `/health`.

Pick the region allowed by your account (for example ap-southeast-1 is closest to Thailand).
Confirm with the instructor whether a public URL, a diagram or a live demo is required.

## Design notes
* Parameters are exported to JSON, so serving has no scikit-learn or pickle version dependency and the
  model can be read by humans (`/model-info`).
* A unit test checks that changing the outcome and rainfall of year t leaves every feature of year t unchanged
  (no leakage); another checks that numpy scoring equals scikit-learn.
