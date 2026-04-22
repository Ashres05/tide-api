#!/usr/bin/env python3
"""
Load weekly/monthly artifacts and serve simulation endpoints for the frontend.

Uses only the Python standard library (no Flask required).

Usage:
  python forecast_engine_server.py --artifacts-dir ./artifacts_75k --port 8765

POST /simulate
  JSON body:
    {
      "release_calendar": [ { ... }, ... ],
      "e_score": 0.8,
      "volume_threshold": 20000,
      "enrich_product_ratios": true,
      "enrich_w2_retention": true
    }

  Simulation uses the archetypal engine: per-metric decay curves (streams / sales / songs),
  genre/label priors, Bear/Base/Bull multipliers, and optional request-body enrichment
  (product ratios, W2 retention) when provided.

  Response: JSON with keys unified_ytd and weekly_injections (per-release weekly volumes).
"""

from __future__ import annotations

import argparse
import json
import logging
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import pandas as pd

try:
    from .all_data_archetypes_simulator_ae import load_artifacts, SimulatorArtifacts
    from .marketshare_75k_simulation import (
        auto_enrich_calendar,
        auto_enrich_w2_retention,
        cluster_product_coef_from_jsonable,
        dna_lookup_from_jsonable,
        run_archetype_scenario,
    )
except ImportError:
    # Support direct script execution from within the `model` directory.
    from all_data_archetypes_simulator_ae import load_artifacts, SimulatorArtifacts
    from marketshare_75k_simulation import (
        auto_enrich_calendar,
        auto_enrich_w2_retention,
        cluster_product_coef_from_jsonable,
        dna_lookup_from_jsonable,
        run_archetype_scenario,
    )

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _df_to_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            out[c] = out[c].dt.strftime("%Y-%m-%d")
    return out.to_dict(orient="records")


class ForecastEngine:
    def __init__(self, artifacts_dir: Path, streams_dir: Path, sales_dir: Path, songs_dir: Path):
        self.artifacts_dir = Path(artifacts_dir).expanduser().resolve()
        self.streams_dir = Path(streams_dir).expanduser().resolve()
        self.sales_dir = Path(sales_dir).expanduser().resolve()
        self.songs_dir = Path(songs_dir).expanduser().resolve()
        self.df_full: pd.DataFrame = pd.DataFrame()
        self.actuals_2026: pd.DataFrame = pd.DataFrame()
        self.e_score_default: float = 0.82
        self._load()

    def _load(self) -> None:
        d = self.artifacts_dir
        self.df_full = pd.read_parquet(d / "df_full.parquet")
        self.actuals_2026 = pd.read_parquet(d / "actuals_2026.parquet")
        for col in ("Week Ending Date",):
            if col in self.df_full.columns:
                self.df_full[col] = pd.to_datetime(self.df_full[col])
            if col in self.actuals_2026.columns:
                self.actuals_2026[col] = pd.to_datetime(self.actuals_2026[col])

        # Load all 3 decay engines!
        self.artifacts_streams = load_artifacts(str(self.streams_dir))
        self.artifacts_sales = load_artifacts(str(self.sales_dir))
        self.artifacts_songs = load_artifacts(str(self.songs_dir))

        meta_path = d / "metadata.json"
        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as f:
                self.e_score_default = float(json.load(f).get("E_score", self.e_score_default))

    def simulate(self, release_calendar: List[dict], e_score: Optional[float] = None, volume_threshold: float = 20000, **kwargs):
        es = float(e_score) if e_score is not None else self.e_score_default
        cal = [dict(x) for x in release_calendar]

        df_out, tracker = run_archetype_scenario(
            cal, self.df_full, self.actuals_2026,
            self.artifacts_streams, self.artifacts_sales, self.artifacts_songs,
            e_score=es, volume_threshold=volume_threshold,
        )
        return {
            "unified_ytd": _df_to_records(df_out),
            "weekly_injections": _df_to_records(tracker),
        }

def make_handler(engine: ForecastEngine):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            logger.info("%s - %s", self.address_string(), fmt % args)

        def _send_json(self, code: int, payload: dict) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path in ("/", "/health"):
                self._send_json(
                    200,
                    {"status": "ok", "artifacts": str(engine.artifacts_dir)},
                )
                return
            self._send_json(404, {"error": "not found"})

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            if path != "/simulate":
                self._send_json(404, {"error": "not found"})
                return
            length = int(self.headers.get("Content-Length", "0") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            try:
                body = json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError as e:
                self._send_json(400, {"error": f"invalid json: {e}"})
                return
            try:
                cal = body.get("release_calendar")
                if cal is None:
                    self._send_json(400, {"error": "release_calendar is required"})
                    return
                if body.get("mode") == "log_decay":
                    self._send_json(
                        400,
                        {"error": "log_decay is removed; only archetypal simulation is supported"},
                    )
                    return
                result = engine.simulate(
                    release_calendar=cal,
                    e_score=body.get("e_score"),
                    volume_threshold=float(body.get("volume_threshold", 20000)),
                    enrich_product_ratios=bool(body.get("enrich_product_ratios", True)),
                    enrich_w2_retention=bool(body.get("enrich_w2_retention", True)),
                )
                self._send_json(200, result)
            except Exception as ex:
                logger.exception("simulate failed")
                self._send_json(500, {"error": str(ex)})

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description="75k marketshare forecast API (stdlib HTTP)")
    ap.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts_75k",
    )
    ap.add_argument("--decay-artifacts-dir", type=Path, default=Path(__file__).resolve().parent / "archetypes_artifacts" / "streams") 
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    engine = ForecastEngine(args.artifacts_dir)
    handler = make_handler(engine)
    server = HTTPServer((args.host, args.port), handler)
    logger.info("Listening on http://%s:%s (POST /simulate)", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down.")


if __name__ == "__main__":
    main()
