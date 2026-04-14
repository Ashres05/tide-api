from __future__ import annotations

import json

from fastapi import FastAPI, HTTPException, Response, status
from pydantic import BaseModel, Field

import model_handler

app = FastAPI(title="Tide Marketshare API", version="1.0.0")


@app.get("/health")
def health():
    return {"status": "ok"}


class ReleaseCreateBody(BaseModel):
    mrelg_id: str | None = None
    name: str
    artist: str
    label_name: str
    release_date: str
    genre: str
    scenario: str
    known_vols: list[float] = Field(default_factory=list)
    fw_vol: float = 0.0
    fy_vol: float = 0.0
    avg_historical_w1_product_ratio: float = 0.3
    product_ratio_coefficient: float = 0.3
    cluster: int = 0
    is_released: bool = False


class ReleaseUpdateBody(ReleaseCreateBody):
    pass


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
    try:
        rid = model_handler.create_release(**body.model_dump())
        return {"release_id": int(rid)}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/releases/{release_id}")
def get_release(release_id: int):
    try:
        rel = model_handler.get_release(release_id)
        # model_handler.get_release raises ValueError when missing
        return rel
    except ValueError as e:
        msg = str(e)
        code = 404 if "No release found" in msg else 400
        raise HTTPException(status_code=code, detail=msg)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/v1/releases/{release_id}")
def update_release(release_id: int, body: ReleaseUpdateBody):
    try:
        model_handler.update_release(id=int(release_id), **body.model_dump())
        return {"ok": True}
    except ValueError as e:
        msg = str(e)
        code = 404 if "No release found" in msg else 400
        raise HTTPException(status_code=code, detail=msg)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/v1/releases/{release_id}")
def delete_release(release_id: int):
    try:
        model_handler.delete_release(int(release_id))
        return {"ok": True}
    except ValueError as e:
        msg = str(e)
        code = 404 if "No release found" in msg else 400
        raise HTTPException(status_code=code, detail=msg)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/releases/{release_id}/weekly")
def weekly_release(release_id: int, week_ending_date: str | None = None):
    try:
        payload = model_handler.df_to_json(model_handler.get_release_forecasts(release_id, week_ending_date))
        return Response(content=payload, media_type="application/json")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/marketshare/weekly")
def weekly_marketshare(week_ending_date: str | None = None):
    try:
        payload = model_handler.df_to_json(model_handler.get_marketshare_forecasts(week_ending_date))
        return Response(content=payload, media_type="application/json")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/releases/{release_id}/series")
def weekly_series(release_id: int):
    """
    Convenience endpoint returning parallel arrays for decay + label marketshare.
    """
    try:
        payload = model_handler.get_release_weekly_decay_and_marketshare_json(release_id)
        return Response(content=json.dumps(payload), media_type="application/json")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

