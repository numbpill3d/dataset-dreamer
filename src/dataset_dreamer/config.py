"""configuration and data models for dataset-dreamer."""
from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class SignalSource(str, Enum):
    MOON_PHASE = "moon_phase"
    SOLAR_WIND = "solar_wind"
    GIT_ACTIVITY = "git_activity"
    CRYPTO_VOLATILITY = "crypto_volatility"
    CUSTOM = "custom"


class ModelType(str, Enum):
    CTGAN = "ctgan"
    TVAE = "tvae"
    CUSTOM_FLOW = "custom_flow"


class OutputFormat(str, Enum):
    PARQUET = "parquet"
    CSV = "csv"
    JSONL = "jsonl"
    POLARS = "polars"


class SignalConfig(BaseModel):
    source: SignalSource
    weight: float = Field(default=1.0, ge=0.0, le=1.0)
    params: dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True

    @field_validator("weight")
    @classmethod
    def normalize_weight(cls, v: float) -> float:
        return max(0.0, min(1.0, v))


class TrainingConfig(BaseModel):
    model_type: ModelType = ModelType.CTGAN
    epochs: int = Field(default=300, ge=10, le=5000)
    batch_size: int = Field(default=500, ge=10, le=10000)
    lr: float = Field(default=2e-4, gt=0.0, le=0.01)
    embedding_dim: int = Field(default=128, ge=16, le=512)
    generator_dim: tuple[int, ...] = (256, 256)
    discriminator_dim: tuple[int, ...] = (256, 256)
    pac: int = Field(default=10, ge=1, le=20)
    cuda: bool = True
    verbose: bool = False


class ConditioningConfig(BaseModel):
    signals: list[SignalConfig] = Field(default_factory=list)
    latent_dim: int = Field(default=16, ge=2, le=128)
    conditioning_strategy: Literal["concat", "adaln", "cross_attn"] = "concat"
    signal_embedding_dim: int = Field(default=32, ge=8, le=256)


class GenerationConfig(BaseModel):
    num_rows: int = Field(default=1000, ge=1, le=1_000_000)
    signal_values: dict[str, float] = Field(default_factory=dict)
    temperature: float = Field(default=1.0, gt=0.0, le=5.0)
    output_format: OutputFormat = OutputFormat.PARQUET
    output_path: Path = Path("output")
    save_model: bool = True
    model_path: Path = Path("models")


class DatasetDreamerConfig(BaseModel):
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    conditioning: ConditioningConfig = Field(default_factory=ConditioningConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    seed: int = 42
    log_level: str = "INFO"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # api keys for external signals
    NOAA_API_KEY: str | None = None
    COINBASE_API_KEY: str | None = None
    COINBASE_API_SECRET: str | None = None
    BINANCE_API_KEY: str | None = None
    BINANCE_API_SECRET: str | None = None

    # paths
    DATA_DIR: Path = Path.home() / ".dataset-dreamer" / "data"
    MODEL_DIR: Path = Path.home() / ".dataset-dreamer" / "models"
    CACHE_DIR: Path = Path.home() / ".dataset-dreamer" / "cache"
    OUTPUT_DIR: Path = Path.home() / ".dataset-dreamer" / "output"

    # defaults
    DEFAULT_EPOCHS: int = 300
    DEFAULT_BATCH_SIZE: int = 500
    DEFAULT_LATENT_DIM: int = 16


settings = Settings()

# ensure directories exist
for path in [
    settings.DATA_DIR,
    settings.MODEL_DIR,
    settings.CACHE_DIR,
    settings.OUTPUT_DIR,
]:
    path.mkdir(parents=True, exist_ok=True)