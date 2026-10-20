"""main pipeline orchestrating the full workflow."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from dataset_dreamer.config import (
    DatasetDreamerConfig,
    ModelType,
    OutputFormat,
    SignalConfig,
    SignalSource,
    TrainingConfig,
    ConditioningConfig,
    GenerationConfig,
)
from dataset_dreamer.core.synthesizer import BaseSynthesizer, SynthesizerFactory
from dataset_dreamer.signals import fetch_signals, fetch_signals_sync, signals_to_vector, create_signal_metadata


class DatasetDreamer:
    def __init__(self, config: DatasetDreamerConfig = None):
        self.config = config or DatasetDreamerConfig()
        self.synthesizer: BaseSynthesizer | None = None
        self.data: pd.DataFrame | None = None
        self.signals: dict[SignalSource, Any] = {}
        self.conditioning_vector: np.ndarray | None = None

    def load_data(self, path: Path | str, **kwargs) -> pd.DataFrame:
        """load data from various formats."""
        path = Path(path)
        suffix = path.suffix.lower()

        if suffix in [".parquet", ".pq"]:
            self.data = pd.read_parquet(path, **kwargs)
        elif suffix == ".csv":
            self.data = pd.read_csv(path, **kwargs)
        elif suffix == ".jsonl":
            self.data = pd.read_json(path, lines=True, **kwargs)
        elif suffix in [".xlsx", ".xls"]:
            self.data = pd.read_excel(path, **kwargs)
        elif suffix == ".feather":
            self.data = pd.read_feather(path, **kwargs)
        else:
            # try polars for broader support
            try:
                self.data = pl.read_csv(str(path), **kwargs).to_pandas()
            except Exception:
                raise ValueError(f"unsupported file format: {suffix}")

        return self.data

    def load_data_polars(self, path: Path | str, **kwargs) -> pl.DataFrame:
        path = Path(path)
        suffix = path.suffix.lower()

        if suffix in [".parquet", ".pq"]:
            return pl.read_parquet(path, **kwargs)
        elif suffix == ".csv":
            return pl.read_csv(path, **kwargs)
        elif suffix == ".jsonl":
            return pl.read_ndjson(path, **kwargs)
        elif suffix == ".feather":
            return pl.read_ipc(path, **kwargs)
        else:
            raise ValueError(f"unsupported file format: {suffix}")

    async def fetch_signals_async(self) -> dict[SignalSource, Any]:
        self.signals = await fetch_signals(self.config.conditioning.signals)
        self.conditioning_vector = signals_to_vector(self.signals, self.config.conditioning.signals)
        return self.signals

    def fetch_signals_sync(self) -> dict[SignalSource, Any]:
        self.signals = fetch_signals_sync(self.config.conditioning.signals)
        self.conditioning_vector = signals_to_vector(self.signals, self.config.conditioning.signals)
        return self.signals

    def train(self, data: pd.DataFrame = None, conditioning: np.ndarray = None) -> BaseSynthesizer:
        data = data or self.data
        if data is None:
            raise ValueError("no data provided. call load_data() first or pass data argument.")

        cond = conditioning or self.conditioning_vector
        if cond is not None and cond.ndim == 1:
            cond = cond.reshape(1, -1)

        # tile conditioning to match data rows if needed
        if cond is not None and cond.shape[0] == 1 and len(data) > 1:
            cond = np.tile(cond, (len(data), 1))

        self.synthesizer = SynthesizerFactory.create(self.config, data)
        self.synthesizer.fit(data, cond)
        return self.synthesizer

    def generate(
        self,
        n: int = None,
        conditioning: np.ndarray = None,
        temperature: float = None,
        signal_overrides: dict[SignalSource, float] = None,
    ) -> pd.DataFrame:
        if self.synthesizer is None:
            raise RuntimeError("model not trained. call train() first or load a model.")

        n = n or self.config.generation.num_rows
        temp = temperature or self.config.generation.temperature

        # build conditioning vector
        if signal_overrides:
            # merge overrides with current signals
            cond_vec = self.conditioning_vector.copy() if self.conditioning_vector is not None else None
            if cond_vec is not None:
                for i, sig_config in enumerate(self.config.conditioning.signals):
                    if sig_config.enabled and sig_config.source in signal_overrides:
                        cond_vec[i] = signal_overrides[sig_config.source]
                cond = cond_vec.reshape(1, -1)
            else:
                cond = None
        elif conditioning is not None:
            cond = conditioning
        else:
            cond = self.conditioning_vector.reshape(1, -1) if self.conditioning_vector is not None else None

        return self.synthesizer.sample(n, cond, temp)

    def generate_streaming(
        self,
        total: int,
        batch_size: int = 1000,
        conditioning: np.ndarray = None,
        temperature: float = None,
        signal_overrides: dict[SignalSource, float] = None,
    ):
        """generate in batches for large outputs."""
        generated = 0
        while generated < total:
            batch = min(batch_size, total - generated)
            yield self.generate(batch, conditioning, temperature, signal_overrides)
            generated += batch

    def save_output(self, data: pd.DataFrame, path: Path = None, format: OutputFormat = None) -> Path:
        path = path or self.config.generation.output_path
        format = format or self.config.generation.output_format
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        stem = f"generated_{timestamp}"

        if format == OutputFormat.PARQUET:
            out_path = path / f"{stem}.parquet"
            data.to_parquet(out_path, index=False)
        elif format == OutputFormat.CSV:
            out_path = path / f"{stem}.csv"
            data.to_csv(out_path, index=False)
        elif format == OutputFormat.JSONL:
            out_path = path / f"{stem}.jsonl"
            data.to_json(out_path, orient="records", lines=True)
        elif format == OutputFormat.POLARS:
            out_path = path / f"{stem}.parquet"
            pl.from_pandas(data).write_parquet(out_path)
        else:
            raise ValueError(f"unknown format: {format}")

        # save metadata
        meta_path = path / f"{stem}_meta.json"
        meta = {
            "rows": len(data),
            "columns": list(data.columns),
            "dtypes": {c: str(data[c].dtype) for c in data.columns},
            "signal_metadata": create_signal_metadata(self.signals),
            "generation_config": self.config.generation.model_dump(),
            "conditioning_vector": self.conditioning_vector.tolist() if self.conditioning_vector is not None else None,
        }
        meta_path.write_text(json.dumps(meta, default=str, indent=2))

        return out_path

    def save_model(self, path: Path = None) -> Path:
        if self.synthesizer is None:
            raise RuntimeError("no model to save")
        path = path or self.config.generation.model_path
        path = Path(path)
        self.synthesizer.save(path)
        return path

    def load_model(self, path: Path) -> BaseSynthesizer:
        path = Path(path)
        self.synthesizer = SynthesizerFactory.load(path)
        return self.synthesizer

    def get_latent_space(self, data: pd.DataFrame = None) -> np.ndarray:
        if self.synthesizer is None:
            raise RuntimeError("model not trained")
        data = data or self.data
        if data is None:
            raise ValueError("no data provided")
        return self.synthesizer.get_latent_representation(data)

    def interpolate_latent(
        self,
        start_data: pd.DataFrame,
        end_data: pd.DataFrame,
        steps: int = 50,
        conditioning: np.ndarray = None,
    ) -> list[pd.DataFrame]:
        """interpolate in latent space between two data points."""
        if self.synthesizer is None:
            raise RuntimeError("model not trained")

        start_latent = self.synthesizer.get_latent_representation(start_data.head(1))
        end_latent = self.synthesizer.get_latent_representation(end_data.head(1))

        alphas = np.linspace(0, 1, steps)
        interpolated = []

        for alpha in alphas:
            z = start_latent * (1 - alpha) + end_latent * alpha
            # need to decode from latent - only works for custom flow
            if hasattr(self.synthesizer, 'model') and hasattr(self.synthesizer.model, 'inverse'):
                self.synthesizer.model.eval()
                with torch.no_grad():
                    z_tensor = torch.tensor(z, dtype=torch.float32)
                    cond_tensor = None
                    if conditioning is not None:
                        cond_tensor = torch.tensor(conditioning.reshape(1, -1), dtype=torch.float32)
                    x, _ = self.synthesizer.model.inverse(z_tensor, cond_tensor)
                    decoded = self.synthesizer._postprocess(x.numpy())
                    interpolated.append(decoded)

        return interpolated


def create_default_config(
    model_type: ModelType = ModelType.CTGAN,
    epochs: int = 300,
    latent_dim: int = 16,
    signals: list[SignalSource] = None,
) -> DatasetDreamerConfig:
    signals = signals or [
        SignalSource.MOON_PHASE,
        SignalSource.SOLAR_WIND,
        SignalSource.GIT_ACTIVITY,
        SignalSource.CRYPTO_VOLATILITY,
    ]

    return DatasetDreamerConfig(
        training=TrainingConfig(model_type=model_type, epochs=epochs),
        conditioning=ConditioningConfig(
            signals=[SignalConfig(source=s, weight=1.0) for s in signals],
            latent_dim=latent_dim,
        ),
        generation=GenerationConfig(num_rows=1000),
    )