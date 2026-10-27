from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass

from stereo_depth.engines.base import Image


@dataclass(frozen=True)
class StereoFrame:
    left: Image
    right: Image
    timestamp_s: float


class CameraSource(ABC):
    """Yields synchronised left/right frames, from hardware or a replayed dataset."""

    @abstractmethod
    def frames(self) -> Iterator[StereoFrame]: ...

    def close(self) -> None:
        return None
