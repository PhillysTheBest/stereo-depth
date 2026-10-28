import threading
import time
from argparse import Namespace
from pathlib import Path
from typing import Any

import numpy as np

from stereo_depth.config import Settings
from stereo_depth.engines.base import DepthEngine, Disparity, Image

# The configuration the released checkpoints were trained with (RAFT-Stereo demo.py defaults).
MODEL_ARGS = Namespace(
    hidden_dims=[128, 128, 128],
    corr_implementation="reg",
    shared_backbone=False,
    corr_levels=4,
    corr_radius=4,
    n_downsample=2,
    context_norm="batch",
    slow_fast_gru=False,
    n_gru_layers=3,
    mixed_precision=False,
)

# The encoder downsamples by 2**n_downsample and the correlation pyramid halves 4 more times.
_PAD_TO = 32


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def cuda_available() -> bool:
    if not torch_available():
        return False
    import torch

    return bool(torch.cuda.is_available())


class RAFTStereoEngine(DepthEngine):
    """Pretrained RAFT-Stereo (Lipson et al., 3DV 2021). Runs on CUDA when present, else CPU.

    On CPU the pair is resized so its longer side is at most ``raft_cpu_max_px`` and fewer GRU
    iterations run; the disparity is scaled back to the input resolution before returning.
    """

    name = "raft_stereo"
    requires_gpu = False

    def __init__(
        self,
        settings: Settings | None = None,
        checkpoint: Path | None = None,
        device: str | None = None,
    ) -> None:
        self.settings = settings or Settings()
        self.checkpoint = (
            checkpoint or Path(self.settings.weights_dir) / self.settings.raft_checkpoint
        )
        self._device = device
        self._model: Any = None
        self._lock = threading.Lock()

    @property
    def device(self) -> str:
        if self._device is None:
            self._device = "cuda" if cuda_available() else "cpu"
        return self._device

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def is_available(self) -> bool:
        return torch_available() and (self.loaded or self.checkpoint.is_file())

    def load(self, state_dict: dict[str, Any] | None = None) -> None:
        """Build the network and load weights; ``state_dict`` overrides the checkpoint file."""
        import torch

        from stereo_depth.engines.raft_stereo._vendor.raft_stereo import RAFTStereo

        with self._lock:
            if self._model is not None:
                return
            model: Any = RAFTStereo(MODEL_ARGS)  # type: ignore[no-untyped-call]
            if state_dict is None:
                state_dict = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
            # Released checkpoints were saved from nn.DataParallel, so keys start with "module.".
            state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
            self._model = model.to(self.device).eval()

    def predict(self, left: Image, right: Image) -> Disparity:
        import torch
        import torch.nn.functional as F  # noqa: N812

        if left.shape != right.shape:
            raise ValueError(f"left {left.shape} and right {right.shape} differ in size")
        if self._model is None:
            self.load()

        on_gpu = self.device == "cuda"
        max_px = self.settings.raft_gpu_max_px if on_gpu else self.settings.raft_cpu_max_px
        iters = self.settings.raft_gpu_iters if on_gpu else self.settings.raft_cpu_iters
        height, width = left.shape[:2]
        scale = min(1.0, max_px / max(height, width))

        start = time.perf_counter()
        with torch.inference_mode():
            pair = [self._to_tensor(img) for img in (left, right)]
            if scale < 1.0:
                size = (max(1, round(height * scale)), max(1, round(width * scale)))
                pair = [
                    F.interpolate(t, size=size, mode="bilinear", align_corners=False) for t in pair
                ]
            h, w = pair[0].shape[-2:]
            pad_h, pad_w = -h % _PAD_TO, -w % _PAD_TO
            pads = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
            pair = [F.pad(t, pads, mode="replicate") for t in pair]

            _, flow = self._model(pair[0], pair[1], iters=iters, test_mode=True)
            # RAFT-Stereo predicts horizontal flow from left to right, which is minus the disparity.
            disp = -flow[:, :1, pads[2] : pads[2] + h, pads[0] : pads[0] + w]
            if scale < 1.0:
                disp = F.interpolate(
                    disp, size=(height, width), mode="bilinear", align_corners=False
                )
                disp = disp * (width / w)
            values = disp[0, 0].float().cpu().numpy()
        runtime_ms = (time.perf_counter() - start) * 1000

        values = values.astype(np.float32)
        values[values <= 0] = np.nan
        return Disparity(values=values, runtime_ms=runtime_ms)

    def _to_tensor(self, img: Image) -> Any:
        import torch

        rgb = img if img.ndim == 3 else np.repeat(img[..., None], 3, axis=2)
        tensor = torch.from_numpy(np.ascontiguousarray(rgb[..., :3])).permute(2, 0, 1).float()
        return tensor[None].to(self.device)
