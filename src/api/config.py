from pathlib import Path
from pydantic import BaseSettings, Field, validator


class Settings(BaseSettings):
    """Runtime configuration for the FastAPI layer."""

    data_root: Path = Field(default_factory=lambda: Path("data/clean/per_video"))
    raw_root: Path = Field(default_factory=lambda: Path("data/raw/single"))
    default_domain: str = "adhoc"

    class Config:
        env_prefix = "CAPSTONE_"
        env_file = ".env"
        env_file_encoding = "utf-8"

    @validator("data_root", pre=True)
    def _ensure_data_root(cls, value) -> Path:
        path = Path(value)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @validator("raw_root", pre=True)
    def _ensure_raw_root(cls, value) -> Path:
        path = Path(value)
        path.mkdir(parents=True, exist_ok=True)
        return path


settings = Settings()
