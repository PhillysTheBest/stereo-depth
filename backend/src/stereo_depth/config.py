from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="STEREO_")

    max_width_px: int = 1280
    data_dir: str = "data"

    # RAFT-Stereo. The CPU host runs it at reduced size with fewer refinement iterations.
    raft_checkpoint: str = "raftstereo-middlebury.pth"
    raft_hf_repo: str | None = None
    raft_cpu_max_px: int = 640
    raft_cpu_iters: int = 12
    raft_gpu_max_px: int = 1280
    raft_gpu_iters: int = 32

    # API
    preload_engines: bool = False
    cors_origins: list[str] = ["*"]

    @property
    def weights_dir(self) -> str:
        return f"{self.data_dir}/weights"
