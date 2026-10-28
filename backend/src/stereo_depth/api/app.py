import base64
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Literal

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from stereo_depth import __version__
from stereo_depth.api import imaging
from stereo_depth.config import Settings
from stereo_depth.core import depth as core_depth
from stereo_depth.core import io as core_io
from stereo_depth.core import ply as core_ply
from stereo_depth.core import resize as core_resize
from stereo_depth.engines.base import DepthEngine, Image
from stereo_depth.engines.registry import LABELS, get_engines


class EngineStatus(BaseModel):
    name: str
    label: str
    requires_gpu: bool
    available: bool = Field(description="Dependencies and weights are present on this host")
    loaded: bool = Field(description="Weights are in memory, so the next request has no load delay")


class Health(BaseModel):
    status: str
    version: str
    gpu: bool
    engines: list[EngineStatus]


class ColorScale(BaseModel):
    """How a heatmap PNG maps colour to value, so the UI can draw its legend."""

    min: float
    max: float
    colormap: Literal["turbo"] = "turbo"
    reversed: bool = Field(description="True when the low end of the range is drawn red")


class DepthStats(BaseModel):
    min: float | None
    max: float | None
    mean: float | None
    median: float | None
    valid_fraction: float


class DepthResult(BaseModel):
    engine: str
    width: int
    height: int
    scale: float = Field(description="Output size divided by the uploaded size (<= 1)")
    unit: Literal["m", "relative"] = Field(
        description="Metres when focal_px and baseline_m are given, else unitless relative depth"
    )
    runtime_ms: float = Field(description="Time spent inside the engine")
    total_ms: float = Field(description="Time for the whole request, decoding included")
    disparity_png: str = Field(description="Colour-mapped disparity as a PNG data URL")
    disparity_scale: ColorScale
    depth_png: str = Field(description="Colour-mapped depth as a PNG data URL, near is red")
    depth_scale: ColorScale
    depth_stats: DepthStats
    ply: str | None = Field(
        default=None, description="Coloured point cloud as a binary PLY data URL, if requested"
    )


def _gpu_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def _loaded(engine: DepthEngine) -> bool:
    return bool(getattr(engine, "loaded", True))


def _servable(engine: DepthEngine, gpu: bool) -> bool:
    return gpu or not engine.requires_gpu


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    if Settings().preload_engines:
        for engine in get_engines().values():
            load = getattr(engine, "load", None)
            if load is not None and engine.is_available() and _servable(engine, _gpu_available()):
                load()
    yield


app = FastAPI(title="Stereo Depth API", version=__version__, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=Settings().cors_origins, allow_methods=["GET", "POST"]
)


def _status(engine: DepthEngine) -> EngineStatus:
    return EngineStatus(
        name=engine.name,
        label=LABELS.get(engine.name, engine.name),
        requires_gpu=engine.requires_gpu,
        available=engine.is_available(),
        loaded=_loaded(engine),
    )


@app.get("/v1/health")
def health() -> Health:
    gpu = _gpu_available()
    engines = [_status(e) for e in get_engines().values() if _servable(e, gpu)]
    return Health(status="ok", version=__version__, gpu=gpu, engines=engines)


@app.get("/v1/engines")
def engines() -> list[EngineStatus]:
    """Engines this host can run. GPU-only engines are hidden when there is no GPU."""
    gpu = _gpu_available()
    return [_status(e) for e in get_engines().values() if _servable(e, gpu)]


def _png_data_url(rgb: Image) -> str:
    return "data:image/png;base64," + base64.b64encode(core_io.encode_png(rgb)).decode()


def _stats(depth: imaging.FloatMap) -> DepthStats:
    finite = depth[np.isfinite(depth)]
    if finite.size == 0:
        return DepthStats(min=None, max=None, mean=None, median=None, valid_fraction=0.0)
    return DepthStats(
        min=float(finite.min()),
        max=float(finite.max()),
        mean=float(finite.mean()),
        median=float(np.median(finite)),
        valid_fraction=finite.size / depth.size,
    )


def _read_image(upload: UploadFile, side: str) -> Image:
    try:
        return core_io.decode_image(upload.file.read())
    except ValueError as exc:
        raise HTTPException(400, f"{side} image: {exc}") from exc


@app.post("/v1/depth")
def depth(
    left: Annotated[UploadFile, File(description="Rectified left image (PNG or JPEG)")],
    right: Annotated[UploadFile, File(description="Rectified right image, same size")],
    engine: Annotated[str, Form()] = "raft_stereo",
    focal_px: Annotated[float | None, Form(gt=0, description="Focal length at upload size")] = None,
    baseline_m: Annotated[float | None, Form(gt=0, description="Camera baseline in metres")] = None,
    include_ply: Annotated[bool, Form()] = False,
) -> DepthResult:
    """Disparity and depth for one rectified pair. Images are processed in memory, never stored."""
    start = time.perf_counter()
    if (focal_px is None) != (baseline_m is None):
        raise HTTPException(422, "give both focal_px and baseline_m, or neither")
    gpu = _gpu_available()
    chosen = get_engines().get(engine)
    if chosen is None or not _servable(chosen, gpu):
        names = [e.name for e in get_engines().values() if _servable(e, gpu)]
        raise HTTPException(404, f"unknown engine {engine!r}; choose one of {names}")
    if not chosen.is_available():
        raise HTTPException(503, f"engine {engine!r} is not ready on this host (missing weights?)")

    left_img = _read_image(left, "left")
    right_img = _read_image(right, "right")
    if left_img.shape != right_img.shape:
        raise HTTPException(400, f"images differ in size: {left_img.shape} vs {right_img.shape}")

    settings = Settings()
    left_img, scale = core_resize.downscale(left_img, settings.max_width_px)
    right_img, _ = core_resize.downscale(right_img, settings.max_width_px)
    result = chosen.predict(left_img, right_img)
    disp = result.values

    if focal_px is not None and baseline_m is not None:
        # Disparity is in downscaled pixels; the focal length is at upload size.
        depth_map = core_depth.disparity_to_depth(disp, focal_px * scale, baseline_m)
        unit: Literal["m", "relative"] = "m"
        ply_focal = focal_px * scale
    else:
        depth_map = imaging.relative_depth(disp)
        unit = "relative"
        ply_focal = float(max(disp.shape))  # about a 53 degree field of view, for viewing only

    d_lo, d_hi = imaging.robust_range(disp)
    z_lo, z_hi = imaging.robust_range(depth_map)
    ply = None
    if include_ply:
        ply_bytes = core_ply.to_ply_bytes(depth_map, left_img, ply_focal)
        ply = "data:application/octet-stream;base64," + base64.b64encode(ply_bytes).decode()

    return DepthResult(
        engine=chosen.name,
        width=disp.shape[1],
        height=disp.shape[0],
        scale=scale,
        unit=unit,
        runtime_ms=result.runtime_ms,
        total_ms=(time.perf_counter() - start) * 1000,
        disparity_png=_png_data_url(imaging.colorize_range(disp, d_lo, d_hi)),
        disparity_scale=ColorScale(min=d_lo, max=d_hi, reversed=False),
        depth_png=_png_data_url(imaging.colorize_range(depth_map, z_lo, z_hi, reverse=True)),
        depth_scale=ColorScale(min=z_lo, max=z_hi, reversed=True),
        depth_stats=_stats(depth_map),
        ply=ply,
    )
