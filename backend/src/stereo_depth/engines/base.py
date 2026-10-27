from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

Image = npt.NDArray[np.uint8]


@dataclass(frozen=True)
class Disparity:
    """Per-pixel disparity in pixels, aligned to the left image. NaN marks invalid pixels."""

    values: npt.NDArray[np.float32]
    runtime_ms: float


class DepthEngine(ABC):
    """Every depth method implements this, so the API, benchmark and UI treat them the same."""

    name: str
    requires_gpu: bool = False

    def is_available(self) -> bool:
        return True

    @abstractmethod
    def predict(self, left: Image, right: Image) -> Disparity: ...
