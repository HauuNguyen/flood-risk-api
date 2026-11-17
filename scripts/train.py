"""Thin CLI wrapper:  python scripts/train.py --panel panel_v2.parquet --out artifacts"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from app.pipeline import main_cli  # noqa: E402

if __name__ == "__main__":
    main_cli()
