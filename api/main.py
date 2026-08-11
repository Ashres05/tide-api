from __future__ import annotations
from contextlib import asynccontextmanager
import sys
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Response, status
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

try:
    import model_handler
    import riaa_handler
    from api.jobs import (
        JobAlreadyRunningError,
        get_manager,
        require_api_key,
    )
except ModuleNotFoundError:
    # Allow direct execution via `python api/main.py` by adding the repo root.
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    import model_handler
    from api.jobs import (  # noqa: E402
        JobAlreadyRunningError,
        get_manager,
        require_api_key,
    )

@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Pull from S3: db, model/data/*.csv (e.g. current_data_all), artifacts_75k, archetypes."""
    from api.s3_pull import sync_serving_inputs_from_s3

    sync_serving_inputs_from_s3()
    yield


app = FastAPI(title="Tide Marketshare API", version="1.1.0", lifespan=lifespan)
# TODO: When a release is officially released but does not have a MRELG ID, give a warning to the user.

# CSVs + artifacts: see api/s3_pull.sync_serving_inputs_from_s3 (startup) and refresh_weekly.
# TODO (Phase 4): Drop the duplicate Prophet market-model fit in train_artifacts_main.
# TODO: Fix a misalignment between current's marketshare and api marketshare.
# TODO: Add a search feature for revenue.


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class ReleaseCreateBody(BaseModel):
    mrelg_id: str | None = None
    title: str | None = None
    name: str | None = Field(
        default=None,
        description="Deprecated on write: use title. Accepted as alias when title is omitted.",
    )
    artist: str
    label_name: str
    release_date: str
    genre: str
    scenario: str
    known_vols: list[float] = Field(default_factory=list)
    fw_vol: float = 0.0
    fw_streams: float = 0.0
    fw_songs: float = 0.0
    fw_sales: float = 0.0
    fy_vol: float = 0.0
    avg_historical_w1_product_ratio: float = 0.3
    product_ratio_coefficient: float = 0.3
    cluster: int | None = None


class ReleaseUpdateBody(ReleaseCreateBody):
    pass


class JobAcceptedResponse(BaseModel):
    job_id: str
    name: str
    status: str
    status_url: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dispatch(job_name: str, target) -> JobAcceptedResponse:
    """
    Submit a long-running admin task to the background job manager and return
    the 202-style acceptance envelope. 409 on duplicate-run is the signal for
    the cron to back off and try again on the next tick.
    """
    try:
        job = get_manager().submit(job_name, target)
    except JobAlreadyRunningError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    return JobAcceptedResponse(
        job_id=job.id,
        name=job.name,
        status=job.status,
        status_url=f"/v1/jobs/{job.id}",
    )


# ---------------------------------------------------------------------------
# Service endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "service": "Tide Marketshare API",
        "health": "/health",
        "docs": "/docs",
    }


@app.get("/doc")
def doc_redirect():
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Admin endpoints (async jobs)
#
# All admin endpoints are POST (state-changing), return 202 Accepted with a
# job_id, and are guarded by the X-API-Key header when TIDE_API_KEY is set.
# The actual work runs in a background thread so the HTTP request returns in
# milliseconds regardless of Snowflake/training time.
# ---------------------------------------------------------------------------

# TODO: There may be edge cases where the week end dates are misaligned among the csvs. for each csv we update, grab its individual week end date and use that as the anchor.
@app.post(
    "/v1/data/refresh_weekly",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=JobAcceptedResponse,
)
def refresh_weekly(
    force_refresh_parquets: bool = False,
    _: None = Depends(require_api_key),
) -> JobAcceptedResponse:
    """
    Weekly orchestration entrypoint. Call this once per week (the cron
    target). Runs, in order: parquet refresh (skipped if both parquets
    already exist), CSV refresh + retrain, release backfill.

    The parquet stage is skipped by default because the two queries behind it
    scan Luminate from 2018 to present and take several minutes each — the
    parquets only need to be rebuilt when the schema or methodology changes.
    Pass ?force_refresh_parquets=true to rebuild them anyway, or call
    /v1/data/refresh_model for a parquet-only refresh.
    """
    return _dispatch(
        "refresh_weekly",
        lambda: model_handler.refresh_weekly(force_refresh_parquets=force_refresh_parquets),
    )


@app.post(
    "/v1/data/refresh_data",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=JobAcceptedResponse,
)
def refresh_data(_: None = Depends(require_api_key)) -> JobAcceptedResponse:
    """
    Pull latest CSV inputs from Snowflake and retrain the forecast models.
    Prefer /refresh_weekly for the cron — this endpoint is kept for targeted
    manual refreshes (e.g. after re-running a single Snowflake query).
    """
    return _dispatch("refresh_data", model_handler.train_model)


@app.post(
    "/v1/data/refresh_model",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=JobAcceptedResponse,
)
def refresh_model(_: None = Depends(require_api_key)) -> JobAcceptedResponse:
    """
    Refresh the per-release AE and worldwide-streams parquets that feed the
    archetype decay trainer. Prefer /refresh_weekly for normal operation.
    """
    return _dispatch("refresh_model", model_handler.refresh_model)

@app.post(
    "/v1/data/reload_artifacts",
    dependencies=[Depends(require_api_key)],
)
def reload_forecast_artifacts_from_disk() -> dict:
    """
    Drop in-memory ForecastEngine caches so the next request reloads parquets/pkls
    from disk. Call after S3 sync or daily_release_sync (S3_DOWNLOAD=1) wrote new
    files without restarting uvicorn. Does not pull from S3 or retrain.
    """
    model_handler.reload_artifacts()
    return {"status": "ok", "detail": "forecast caches cleared; next API use loads from disk"}


@app.post(
    "/v1/data/export_boot_json",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=JobAcceptedResponse,
)
def export_boot_json(_: None = Depends(require_api_key)) -> JobAcceptedResponse:
    """
    Build model/data/boot.json and upload to s3://…/boot.json.
    Prefer running after roster prewarm so streaming.weekly_streams is populated.
    """
    return _dispatch("export_boot_json", model_handler.export_boot_json_snapshot)


@app.get("/v1/boot.json")
def get_boot_json():
    """
    Cold-start snapshot: streaming roster + weekly stream actuals, marketshare
    actuals/weekly, and expected releases with Actual + Forecast weekly AE
    (same shape as GET /v1/releases/{id}/weekly).

    FE should load this once at startup. Insert/edit/delete drops via
    POST/PUT/DELETE /v1/releases (live). See payload ``ui`` for the contract.
    """
    from boot_snapshot import export_boot_json_snapshot, load_boot_json_bytes

    body = load_boot_json_bytes()
    if body is None:
        try:
            export_boot_json_snapshot()
            body = load_boot_json_bytes()
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"boot.json missing and live build failed: {e}",
            ) from e
    if body is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="boot.json unavailable",
        )
    return Response(
        content=body,
        media_type="application/json",
        headers={
            "Cache-Control": "private, max-age=60, stale-while-revalidate=3600",
        },
    )


@app.post(
    "/v1/releases/backfill",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=JobAcceptedResponse,
)
def backfill_releases(_: None = Depends(require_api_key)) -> JobAcceptedResponse:
    """
    Pull any newly surfaced Luminate releases into the local SQLite store and
    refresh per-release historical metrics. Result payload (inserted/skipped/
    errors) is available on the completed job via GET /v1/jobs/{job_id}.
    """
    return _dispatch("backfill_releases", model_handler.backfill_releases)


@app.post(
    "/v1/revenue/backfill_streaming_roster",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=JobAcceptedResponse,
)
def backfill_streaming_roster(_: None = Depends(require_api_key)) -> JobAcceptedResponse:
    """
    Populate STREAMING_ROSTER_2026 from Snowflake (Album / Single / EP, no 75k gate).

    Incremental by default when the table has rows. Force a full rebuild with
    ``TIDE_STREAMING_ROSTER_FULL=1`` (``release_date >= today - 78 weeks``;
    override via ``TIDE_STREAMING_ROSTER_FULL_WEEKS``) on the API worker process.
    """
    return _dispatch(
        "backfill_streaming_roster",
        model_handler.backfill_streaming_roster,
    )


@app.post(
    "/v1/revenue/prewarm_streaming_roster",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=JobAcceptedResponse,
)
def prewarm_streaming_roster(_: None = Depends(require_api_key)) -> JobAcceptedResponse:
    """
    Batch-fill SQLite caches for STREAMING_ROSTER_2026 (daily + weekly worldwide
    streams). Incremental Snowflake per MRELG. Run after roster backfill or nightly.
    """
    return _dispatch(
        "prewarm_streaming_roster",
        model_handler.prewarm_streaming_roster_caches,
    )


@app.get("/v1/revenue/quarterly_share_and_qtd")
def quarterly_share_and_qtd(fiscal_year: int | None = None):
    """
    Fiscal-month AMG market share and QTD growth (from
    model/data/monthly_share_and_qtd.csv on S3).

    Optional filter: ``fiscal_year`` (e.g. 2026).
    """
    try:
        payload = model_handler.get_quarterly_share_and_qtd_json(fiscal_year=fiscal_year)
        return Response(content=payload, media_type="application/json")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/v1/revenue/quarterly_share_level3")
def quarterly_share_level3(
    fiscal_year: int | None = None,
    profit_center_label: str | None = None,
):
    """
    Fiscal-month level-3 profit-center share metrics (from
    model/data/monthly_share_labels.csv on S3).

    Optional filters: ``fiscal_year``, ``profit_center_label`` (e.g. ``Atlantic``).
    """
    try:
        payload = model_handler.get_quarterly_share_level3_json(
            fiscal_year=fiscal_year,
            profit_center_label=profit_center_label,
        )
        return Response(content=payload, media_type="application/json")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


def _platform_marketshare_response(
    dataset_key: str,
    *,
    period_start_date: str | None = None,
    year: int | None = None,
    label_bucket: str | None = None,
) -> Response:
    try:
        if period_start_date is not None:
            model_handler._validate_date(period_start_date)
        payload = model_handler.get_platform_marketshare_json(
            dataset_key,
            period_start_date=period_start_date,
            year=year,
            label_bucket=label_bucket,
        )
        return Response(content=payload, media_type="application/json")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/v1/revenue/labels_apple_marketshare")
def labels_apple_marketshare(
    period_start_date: str | None = None,
    year: int | None = None,
    label_bucket: str | None = None,
):
    """
    Per-label Apple US marketshare by month (from
    model/data/labels_apple_marketshare.csv on S3).

    Optional filters: ``period_start_date`` (YYYY-MM-DD), ``year``,
    ``label_bucket`` (e.g. ``Atlantic``).
    """
    return _platform_marketshare_response(
        "labels_apple_marketshare",
        period_start_date=period_start_date,
        year=year,
        label_bucket=label_bucket,
    )


@app.get("/v1/revenue/labels_spotify_marketshare")
def labels_spotify_marketshare(
    period_start_date: str | None = None,
    year: int | None = None,
    label_bucket: str | None = None,
):
    """
    Per-label Spotify US marketshare by month (from
    model/data/labels_spotify_marketshare.csv on S3).

    Optional filters: ``period_start_date`` (YYYY-MM-DD), ``year``,
    ``label_bucket`` (e.g. ``Atlantic``).
    """
    return _platform_marketshare_response(
        "labels_spotify_marketshare",
        period_start_date=period_start_date,
        year=year,
        label_bucket=label_bucket,
    )


@app.get("/v1/revenue/amg_full_apple_marketshare")
def amg_full_apple_marketshare(
    period_start_date: str | None = None,
    year: int | None = None,
    label_bucket: str | None = None,
):
    """
    AMG-full Apple US marketshare by month (from
    model/data/amg_full_apple_marketshare.csv on S3).

    Optional filters: ``period_start_date`` (YYYY-MM-DD), ``year``,
    ``label_bucket``.
    """
    return _platform_marketshare_response(
        "amg_full_apple_marketshare",
        period_start_date=period_start_date,
        year=year,
        label_bucket=label_bucket,
    )


@app.get("/v1/revenue/amg_full_spotify_marketshare")
def amg_full_spotify_marketshare(
    period_start_date: str | None = None,
    year: int | None = None,
    label_bucket: str | None = None,
):
    """
    AMG-full Spotify US marketshare by month (from
    model/data/amg_full_spotify_marketshare.csv on S3).

    Optional filters: ``period_start_date`` (YYYY-MM-DD), ``year``,
    ``label_bucket``.
    """
    return _platform_marketshare_response(
        "amg_full_spotify_marketshare",
        period_start_date=period_start_date,
        year=year,
        label_bucket=label_bucket,
    )


@app.get("/v1/revenue/releases_by_q_amg_labels")
def releases_by_q_amg_labels(
    level_3_distributor: str | None = None,
    profit_center_label: str | None = None,
    first_sale_date_from: str | None = None,
    first_sale_date_to: str | None = None,
    baseline_fiscal_year: int | None = None,
    baseline_fiscal_quarter: str | None = None,
    comparison_fiscal_year: int | None = None,
    comparison_fiscal_quarter: str | None = None,
):
    """
    US non-compilation releases for AMG level-3 labels (from
    model/data/releases_by_q_amg_labels.csv on S3).

    Flat list: optional ``level_3_distributor`` or ``profit_center_label`` (from
    level3 share UI, e.g. ``Atlantic``), ``first_sale_date_from`` / ``to``.

    Comparison mode: pass label + baseline and comparison ``fiscal_year`` /
    ``fiscal_quarter`` (exact strings from ``quarterly_share_level3``).
    """
    try:
        for d in (first_sale_date_from, first_sale_date_to):
            if d is not None:
                model_handler._validate_date(d)
        payload = model_handler.get_releases_by_q_amg_labels_json(
            level_3_distributor=level_3_distributor,
            profit_center_label=profit_center_label,
            first_sale_date_from=first_sale_date_from,
            first_sale_date_to=first_sale_date_to,
            baseline_fiscal_year=baseline_fiscal_year,
            baseline_fiscal_quarter=baseline_fiscal_quarter,
            comparison_fiscal_year=comparison_fiscal_year,
            comparison_fiscal_quarter=comparison_fiscal_quarter,
        )
        return Response(content=payload, media_type="application/json")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/v1/revenue/streaming_roster")
def list_streaming_roster():
    """
    List STREAMING_ROSTER_2026 (streaming revenue board).

    Each row includes ``LUMINATE_ARTIST_ID`` (main artist) when available —
    use with ``GET /v1/artist_art/{LUMINATE_ARTIST_ID}``.
    """
    try:
        return Response(
            content=model_handler.get_streaming_roster_2026_json(),
            media_type="application/json",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/v1/revenue/ytd_fiscal_revenue")
def ytd_fiscal_revenue(
    week_end_date: str | None = None,
    level_1_distributor: str | None = None,
    level_2_distributor: str | None = None,
    level_3_distributor: str | None = None,
):
    """
    Weekly proxy fiscal revenue by distributor label (from
    model/data/ytd_fiscal_revenue_by_label.csv, synced to S3 on refresh_weekly).

    Optional filters: week_end_date (YYYY-MM-DD), level_1/2/3_distributor.
    """
    try:
        if week_end_date is not None:
            model_handler._validate_date(week_end_date)
        payload = model_handler.get_ytd_fiscal_revenue_by_label_json(
            week_end_date=week_end_date,
            level_1_distributor=level_1_distributor,
            level_2_distributor=level_2_distributor,
            level_3_distributor=level_3_distributor,
        )
        return Response(content=payload, media_type="application/json")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


# ---------------------------------------------------------------------------
# Job status endpoints
# ---------------------------------------------------------------------------

@app.get("/v1/jobs/{job_id}")
def get_job(job_id: str):
    job = get_manager().get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")
    return job.to_dict()


@app.get("/v1/jobs")
def list_jobs(limit: int = 20):
    if limit < 1 or limit > 200:
        raise HTTPException(status_code=400, detail="limit must be in [1, 200].")
    return [j.to_dict() for j in get_manager().list(limit=limit)]


# ---------------------------------------------------------------------------
# Release CRUD + forecast endpoints (read-only / per-release, stay synchronous)
# ---------------------------------------------------------------------------

@app.get("/v1/releases")
def list_releases():
    """
    Returns a compact list of releases for dropdowns/selection.

    Response: [{"id": <int>, "album": <str>, "artist": <str>}, ...]
    """
    try:
        return Response(
            content=model_handler.get_all_releases_series_json(),
            media_type="application/json",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/releases", status_code=status.HTTP_201_CREATED)
def create_release(body: ReleaseCreateBody):
    """
    Insert a drop. Live path — does not rebuild boot.json.

    Response includes a boot-shaped ``release`` for FE local-state patch.
    After create, FE should refetch GET /v1/marketshare/weekly (server cache cleared).
    """
    try:
        from boot_snapshot import release_boot_item_by_id

        rid = model_handler.create_release(
            **model_handler.normalize_release_write_kwargs(body.model_dump())
        )
        return {
            "release_id": int(rid),
            "release": release_boot_item_by_id(int(rid)),
            "refetch": ["GET /v1/marketshare/weekly", "GET /v1/marketshare/actuals"],
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/releases/{release_id}")
def get_release(release_id: int):
    try:
        rel = model_handler.get_release(release_id)
        return rel
    except ValueError as e:
        msg = str(e)
        code = 404 if "No release found" in msg else 400
        raise HTTPException(status_code=code, detail=msg)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/v1/releases/{release_id}")
def update_release(release_id: int, body: ReleaseUpdateBody):
    """
    Edit a drop. Live path — does not rebuild boot.json.

    Returns boot-shaped ``release`` for FE patch; refetch marketshare weekly after.
    """
    try:
        from boot_snapshot import release_boot_item_by_id

        model_handler.update_release(
            id=int(release_id),
            **model_handler.normalize_release_write_kwargs(body.model_dump()),
        )
        return {
            "ok": True,
            "release_id": int(release_id),
            "release": release_boot_item_by_id(int(release_id)),
            "refetch": ["GET /v1/marketshare/weekly", "GET /v1/marketshare/actuals"],
        }
    except ValueError as e:
        msg = str(e)
        code = 404 if "No release found" in msg else 400
        raise HTTPException(status_code=code, detail=msg)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/v1/releases/{release_id}")
def delete_release(release_id: int):
    """Delete a drop. Live path — FE should drop local item and refetch marketshare."""
    try:
        model_handler.delete_release(int(release_id))
        return {
            "ok": True,
            "release_id": int(release_id),
            "refetch": ["GET /v1/marketshare/weekly", "GET /v1/marketshare/actuals"],
        }
    except ValueError as e:
        msg = str(e)
        code = 404 if "No release found" in msg else 400
        raise HTTPException(status_code=code, detail=msg)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/releases/{release_id}/weekly")
def weekly_release(release_id: int, week_ending_date: str | None = None):
    """
    Weekly US album-equivalent forecast for one marketshare release.

    Returns one row per week with ``data_type`` (``Actual`` when the week is
    backed by Snowflake metrics in ``MARKETSHARE_RELEASE_METRICS``, else
    ``Forecast``), ``streaming_equivalent``, ``product_sales``,
    ``song_sale_equivalent``, and ``total`` (sum of the three). Use ``total``
    for the plotted curve; component columns support finer-grained forecasting.
    """
    try:
        payload = model_handler.df_to_json(
            model_handler.get_release_forecasts(release_id, week_ending_date)
        )
        return Response(content=payload, media_type="application/json")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/album_art/{mrelg_id}")
def album_art(mrelg_id: str):
    """
    Stream the cover image for a given mrelg_id from s3://<bucket>/album_art/.
    Returns the raw image bytes with a long Cache-Control so Cloudflare's edge
    caches each cover for a day after the first hit. Falls back to 404 when
    no cover has been uploaded for the mrelg_id — the frontend's <AlbumArt>
    component swaps to its gradient placeholder on the load error.
    """
    import album_art as _album_art

    result = _album_art.fetch_bytes(mrelg_id)
    if result is None:
        raise HTTPException(status_code=404, detail="album art not found")
    body, content_type = result
    return Response(
        content=body,
        media_type=content_type,
        headers={
            "Cache-Control": "public, max-age=86400, immutable",
        },
    )


@app.get("/v1/artist_art/{luminate_artist_id}")
def artist_art(luminate_artist_id: str):
    """
    Stream the artist profile image for a Luminate artist id from
    ``s3://<bucket>/artist_art/{LUMINATE_ARTIST_ID}.jpeg``.

    Pair with ``LUMINATE_ARTIST_ID`` from ``GET /v1/revenue/streaming_roster``.
    Returns 404 when no image has been uploaded — frontend should use a
    placeholder on load error. Cached at the edge for one day.
    """
    import artist_art as _artist_art

    result = _artist_art.fetch_bytes(luminate_artist_id)
    if result is None:
        raise HTTPException(status_code=404, detail="artist art not found")
    body, content_type = result
    return Response(
        content=body,
        media_type=content_type,
        headers={
            "Cache-Control": "public, max-age=86400, immutable",
        },
    )


@app.get("/v1/marketshare/actuals")
def actuals_marketshare():
    try:
        payload = model_handler.df_to_json(model_handler.get_marketshare_actuals())
        return Response(content=payload, media_type="application/json")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/marketshare/weekly")
def weekly_marketshare(week_ending_date: str | None = None):
    try:
        payload = model_handler.df_to_json(
            model_handler.get_marketshare_forecasts(week_ending_date)
        )
        return Response(content=payload, media_type="application/json")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/revenue/search_global_streaming")
def search_global_streaming(
    artist: str = "",
    title: str = "",
    limit: int = 20,
):
    """
    Search the local MARKETSHARE_SEARCH_SUMMARY table for the best-matching
    Luminate releases given a free-text artist and album title. Returns a
    ranked list of candidates (each carrying its `mrelg_id`) so the front
    end can call /v1/releases/global_streaming_by_mrelg/{mrelg_id} without
    ever exposing the MRELG lookup to the user.

    Ranking: weighted blend of fuzzy text match and log-scaled daily streams
    (text-dominant by default) so popular releases bubble up but never drown
    out close text matches on smaller releases.

    Note: this static route is intentionally registered before the
    /v1/releases/{release_id} parameterized routes so FastAPI matches it as
    a literal path instead of trying to coerce 'search_global_streaming'
    into an int release_id.
    """
    try:
        payload = model_handler.search_releases_by_artist_title_json(
            artist=artist, title=title, limit=limit
        )
        return Response(content=payload, media_type="application/json")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/revenue/search_global_streaming_singles")
def search_global_streaming_singles(
    artist: str = "",
    title: str = "",
    limit: int = 20,
):
    """
    Search MARKETSHARE_SEARCH_SUMMARY_SINGLES for singles release groups given
    free-text artist and title. Same response shape as
    GET /v1/revenue/search_global_streaming. Pair with
    GET /v1/revenue/global_streaming_singles_by_mrelg/{mrelg_id}.
    """
    try:
        payload = model_handler.search_releases_by_artist_title_singles_json(
            artist=artist, title=title, limit=limit
        )
        return Response(content=payload, media_type="application/json")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/revenue/global_streaming_by_mrelg/{mrelg_id}")
def global_streaming_by_mrelg(mrelg_id: str, scenario: str = "Base"):
    """
    Returns observed + forecasted global weekly streams for a Luminate MRELG
    release group. Metadata (artist/title/release_date/genre) is resolved
    from the local MARKETSHARE_SEARCH_SUMMARY table when present and falls
    back to a direct Snowflake lookup. Pair with
    GET /v1/releases/search_global_streaming so the front end never has to
    resolve MRELG IDs by hand.

    ``scenario`` query param ("Base" / "Bear" / "Bull"; default Base) routes
    through to the worldwide_streams archetype simulation. The asymptotic
    floor is invariant under scenario; only the future weeks above the floor
    are scaled (post-fit shock).
    """
    try:
        payload = model_handler.df_to_json(
            model_handler.get_global_streaming_forecast_by_mrelg(
                mrelg_id, scenario=scenario
            )
        )
        return Response(content=payload, media_type="application/json")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/revenue/global_streaming_singles_by_mrelg/{mrelg_id}")
def global_streaming_singles_by_mrelg(mrelg_id: str, scenario: str = "Base"):
    """
    Observed + forecasted global weekly streams for a singles MRELG release group.
    Uses worldwide_streams_singles archetype artifacts. Same JSON array shape as
    GET /v1/revenue/global_streaming_by_mrelg/{mrelg_id}. Pair with
    GET /v1/revenue/search_global_streaming_singles.

    ``scenario`` ("Base" / "Bear" / "Bull"; default Base) applies the learned
    post-fit shock from the singles archetype bundle.
    """
    try:
        payload = model_handler.df_to_json(
            model_handler.get_global_streaming_forecast_singles_by_mrelg(
                mrelg_id, scenario=scenario
            )
        )
        return Response(content=payload, media_type="application/json")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/forecast/search/{mrelg_id}")
def search_catalog_eoy_forecast(
    mrelg_id: str, target_year: int = 2026, scenario: str = "Base"
):
    """
    Autoregressive catalog-decay forecast of weekly **worldwide streams** through
    the end of ``target_year`` (columns match ``catalog_streams_pruned_80k.parquet``).

    Returns JSON array of row objects: actual weeks in ``target_year`` from Snowflake
    history, then forecast weeks from the first week after the last observed point
    through ``{target_year}-12-31``.

    ``scenario`` query param ("Base" / "Bear" / "Bull"; default Base) shocks
    the worldwide_streams archetype that powers the first 78 forecast weeks.
    The catalog-decay AR continuation past week 78 has no learned
    scenario_multipliers of its own, but inherits the shock through its lag
    features because ``rolling`` is seeded with the already-shocked bridge
    weeks before AR steps begin. Visible scenario impact concentrates in
    weeks ~1–50 of the curve; the AR tail re-converges as the archetype
    decays to its dynamic floor by week ~77 (which is the same level across
    Bear / Base / Bull). Use ``/v1/revenue/global_streaming_by_mrelg/{mrelg_id}``
    for the strict 78-week archetype-only view.
    """
    try:
        df = model_handler.get_eoy_search_forecast(
            mrelg_id, target_year, scenario=scenario
        )
        payload = model_handler.df_to_json(df)
        return Response(content=payload, media_type="application/json")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/revenue/daily_streams_by_mrelg/{mrelg_id}")
def daily_streams_by_mrelg(mrelg_id: str):
    """
    Live Revenue board only — daily worldwide on-demand streams since release
    for the given Luminate MRELG release group. Reads from the cached SQLite
    table MARKETSHARE_DAILY_GLOBAL_STREAMS; lazily refreshes from Snowflake on
    the first call (or when the cache is older than DAILY_STREAMS_STALE_DAYS).

    The response is sorted ascending by report_date and intentionally excludes
    the most recent calendar day (Snowflake-side filter — Luminate data lags
    by ~1 day before settling). This endpoint is not joined into the
    forecast pipeline; it is a read-only surface for the Live Revenue board.
    """
    try:
        payload = model_handler.df_to_json(
            model_handler.get_daily_global_streams_by_mrelg(mrelg_id)
        )
        return Response(content=payload, media_type="application/json")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/revenue/catalog_revenue_2025/{mrelg_id}")
def catalog_revenue_2025(mrelg_id: str):
    """
    Live Revenue board — 2025 catalog revenue for a single Luminate MRELG
    release group. Reads from the local MARKETSHARE_REVENUE_2025 table,
    which is loaded one-shot from
    s3://parquetgarage/model/data/2025_revenue_catalog.csv. Returns
    {"catalog_revenue_2025": float | null}; null when the MRELG isn't in
    the file. Not joined into the forecast pipeline.
    """
    try:
        body = model_handler.get_catalog_revenue_2025_by_mrelg(mrelg_id)
        return {"mrelg_id": (mrelg_id or "").strip(), **body}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/releases/{release_id}/global_streaming", deprecated=True)
def global_streaming_release(release_id: int):
    """
    Deprecated: use GET /v1/releases/global_streaming_by_mrelg/{mrelg_id}
    instead. Kept temporarily so existing callers that still pass a local
    SQLite release_id continue to work during front-end migration.
    """
    try:
        payload = model_handler.df_to_json(
            model_handler.get_global_streaming_forecast(release_id)
        )
        return Response(content=payload, media_type="application/json")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------------------------------------------------------
# RIAA endpoints
# ---------------------------------------------------------------------------
@app.get("/v1/riaa/eligible_certifications")
def eligible_certifications():
    """
    Returns a list of eligible certifications from the RIAA.
    """
    try:
        payload = riaa_handler.get_eligible_certifications()
        return Response(content=payload, media_type="application/json")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/riaa/official_certifications")
def official_certifications():
    """
    Returns a list of official certifications from the RIAA.
    """
    try:
        payload = riaa_handler.get_official_certifications()
        return Response(content=payload, media_type="application/json")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
