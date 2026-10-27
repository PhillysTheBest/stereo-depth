from fastapi import FastAPI
from pydantic import BaseModel

from stereo_depth import __version__


class Health(BaseModel):
    status: str
    version: str
    gpu: bool


def _gpu_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


app = FastAPI(title="Stereo Depth API", version=__version__)


@app.get("/v1/health")
def health() -> Health:
    return Health(status="ok", version=__version__, gpu=_gpu_available())
