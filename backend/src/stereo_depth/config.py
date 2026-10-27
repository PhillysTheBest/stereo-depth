from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="STEREO_")

    max_width_px: int = 1280
    data_dir: str = "data"
