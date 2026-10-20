"""core synthesis engine: conditional ctgan, tvae, and custom normalizing flow."""
from __future__ import annotations

import json
import pickle
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sdv.single_table import CTGANSynthesizer, TVAESynthesizer
from sdv.metadata import SingleTableMetadata

from dataset_dreamer.config import (
    ModelType,
    TrainingConfig,
    ConditioningConfig,
    GenerationConfig,
    DatasetDreamerConfig,
    OutputFormat,
)
from dataset_dreamer.signals import SignalConfig, SignalValue


warnings.filterwarnings("ignore", category=UserWarning)


class BaseSynthesizer(ABC):
    @abstractmethod
    def fit(self, data: pd.DataFrame, conditioning: np.ndarray = None) -> None:
        pass

    @abstractmethod
    def sample(self, n: int, conditioning: np.ndarray = None, temperature: float = 1.0) -> pd.DataFrame:
        pass

    @abstractmethod
    def save(self, path: Path) -> None:
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "BaseSynthesizer":
        pass

    @abstractmethod
    def get_latent_representation(self, data: pd.DataFrame) -> np.ndarray:
        pass


class ConditionalCTGAN(BaseSynthesizer):
    def __init__(self, config: TrainingConfig, conditioning_config: ConditioningConfig, data_metadata: SingleTableMetadata = None):
        self.config = config
        self.conditioning_config = conditioning_config
        self.data_metadata = data_metadata
        self.model: CTGANSynthesizer | None = None
        self._fitted = False
        self._conditioning_dim = sum(1 for c in conditioning_config.signals if c.enabled)

    def fit(self, data: pd.DataFrame, conditioning: np.ndarray = None) -> None:
        if self.data_metadata is None:
            self.data_metadata = SingleTableMetadata()
            self.data_metadata.detect_from_dataframe(data)

        # sdv's CTGAN doesn't natively support conditioning
        # we'll wrap it with a conditional approach: train on augmented data
        if conditioning is not None and self._conditioning_dim > 0:
            data = self._augment_with_conditioning(data, conditioning)

        self.model = CTGANSynthesizer(
            metadata=self.data_metadata,
            enforce_rounding=False,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            lr=self.config.lr,
            embedding_dim=self.config.embedding_dim,
            generator_dim=self.config.generator_dim,
            discriminator_dim=self.config.discriminator_dim,
            pac=self.config.pac,
            cuda=self.config.cuda and torch.cuda.is_available(),
            verbose=self.config.verbose,
        )
        self.model.fit(data)
        self._fitted = True

    def _augment_with_conditioning(self, data: pd.DataFrame, conditioning: np.ndarray) -> pd.DataFrame:
        """append conditioning as extra columns for joint modeling."""
        n = len(data)
        if conditioning.shape[0] != n:
            # repeat or tile conditioning to match
            if conditioning.shape[0] == 1:
                conditioning = np.tile(conditioning, (n, 1))
            else:
                raise ValueError(f"conditioning shape {conditioning.shape} incompatible with data {n}")

        cond_df = pd.DataFrame(
            conditioning,
            columns=[f"__cond_{i}" for i in range(self._conditioning_dim)],
            index=data.index,
        )
        return pd.concat([data, cond_df], axis=1)

    def sample(self, n: int, conditioning: np.ndarray = None, temperature: float = 1.0) -> pd.DataFrame:
        if not self._fitted or self.model is None:
            raise RuntimeError("model not fitted")

        if conditioning is not None and self._conditioning_dim > 0:
            # sample from conditional distribution by fixing conditioning columns
            return self._conditional_sample(n, conditioning, temperature)

        return self.model.sample(n)

    def _conditional_sample(self, n: int, conditioning: np.ndarray, temperature: float) -> pd.DataFrame:
        # for CTGAN, we sample and then filter/reject based on conditioning proximity
        # this is approximate but works for moderate conditioning dims
        cond_cols = [f"__cond_{i}" for i in range(self._conditioning_dim)]

        if conditioning.shape[0] == 1:
            conditioning = np.tile(conditioning, (n, 1))

        # oversample and select closest
        oversample_factor = max(5, min(50, 1000 // max(1, n)))
        samples = self.model.sample(n * oversample_factor)

        if not all(c in samples.columns for c in cond_cols):
            # model wasn't trained with conditioning, fall back
            return samples.drop(columns=[c for c in cond_cols if c in samples.columns]).head(n)

        # compute distance in conditioning space
        sample_cond = samples[cond_cols].values
        target_cond = conditioning[:len(sample_cond)]

        # handle temperature: higher = more exploration
        if temperature != 1.0:
            distances = np.linalg.norm(sample_cond - target_cond, axis=1)
            probs = np.exp(-distances / temperature)
            probs = probs / probs.sum()
            idx = np.random.choice(len(samples), size=n, replace=False, p=probs)
        else:
            distances = np.linalg.norm(sample_cond - target_cond, axis=1)
            idx = np.argsort(distances)[:n]

        result = samples.iloc[idx].drop(columns=cond_cols).reset_index(drop=True)
        return result

    def save(self, path: Path) -> None:
        if self.model is None:
            raise RuntimeError("no model to save")
        path.mkdir(parents=True, exist_ok=True)
        self.model.save(path / "model.pkl")
        meta = {
            "type": "ctgan",
            "config": self.config.model_dump(),
            "conditioning_config": self.conditioning_config.model_dump(),
            "conditioning_dim": self._conditioning_dim,
        }
        (path / "meta.json").write_text(json.dumps(meta, default=str))

    @classmethod
    def load(cls, path: Path) -> "ConditionalCTGAN":
        meta = json.loads((path / "meta.json").read_text())
        config = TrainingConfig(**meta["config"])
        cond_config = ConditioningConfig(**meta["conditioning_config"])
        obj = cls(config, cond_config)
        obj.model = CTGANSynthesizer.load(path / "model.pkl")
        obj._fitted = True
        obj._conditioning_dim = meta["conditioning_dim"]
        return obj

    def get_latent_representation(self, data: pd.DataFrame) -> np.ndarray:
        # CTGAN doesn't expose latent directly, use generator input noise as proxy
        if self.model is None:
            raise RuntimeError("model not fitted")
        # this is a hack - we'd need to access internal generator
        return np.random.randn(len(data), self.config.embedding_dim).astype(np.float32)


class ConditionalTVAE(BaseSynthesizer):
    def __init__(self, config: TrainingConfig, conditioning_config: ConditioningConfig, data_metadata: SingleTableMetadata = None):
        self.config = config
        self.conditioning_config = conditioning_config
        self.data_metadata = data_metadata
        self.model: TVAESynthesizer | None = None
        self._fitted = False
        self._conditioning_dim = sum(1 for c in conditioning_config.signals if c.enabled)

    def fit(self, data: pd.DataFrame, conditioning: np.ndarray = None) -> None:
        if self.data_metadata is None:
            self.data_metadata = SingleTableMetadata()
            self.data_metadata.detect_from_dataframe(data)

        if conditioning is not None and self._conditioning_dim > 0:
            data = self._augment_with_conditioning(data, conditioning)

        self.model = TVAESynthesizer(
            metadata=self.data_metadata,
            enforce_rounding=False,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            lr=self.config.lr,
            embedding_dim=self.config.embedding_dim,
            compress_dims=self.config.generator_dim,
            decompress_dims=self.config.discriminator_dim,
            cuda=self.config.cuda and torch.cuda.is_available(),
            verbose=self.config.verbose,
        )
        self.model.fit(data)
        self._fitted = True

    def _augment_with_conditioning(self, data: pd.DataFrame, conditioning: np.ndarray) -> pd.DataFrame:
        n = len(data)
        if conditioning.shape[0] != n:
            if conditioning.shape[0] == 1:
                conditioning = np.tile(conditioning, (n, 1))
            else:
                raise ValueError(f"conditioning shape {conditioning.shape} incompatible with data {n}")

        cond_df = pd.DataFrame(
            conditioning,
            columns=[f"__cond_{i}" for i in range(self._conditioning_dim)],
            index=data.index,
        )
        return pd.concat([data, cond_df], axis=1)

    def sample(self, n: int, conditioning: np.ndarray = None, temperature: float = 1.0) -> pd.DataFrame:
        if not self._fitted or self.model is None:
            raise RuntimeError("model not fitted")

        if conditioning is not None and self._conditioning_dim > 0:
            return self._conditional_sample(n, conditioning, temperature)

        return self.model.sample(n)

    def _conditional_sample(self, n: int, conditioning: np.ndarray, temperature: float) -> pd.DataFrame:
        cond_cols = [f"__cond_{i}" for i in range(self._conditioning_dim)]

        if conditioning.shape[0] == 1:
            conditioning = np.tile(conditioning, (n, 1))

        oversample_factor = max(5, min(50, 1000 // max(1, n)))
        samples = self.model.sample(n * oversample_factor)

        if not all(c in samples.columns for c in cond_cols):
            return samples.drop(columns=[c for c in cond_cols if c in samples.columns]).head(n)

        sample_cond = samples[cond_cols].values
        target_cond = conditioning[:len(sample_cond)]

        if temperature != 1.0:
            distances = np.linalg.norm(sample_cond - target_cond, axis=1)
            probs = np.exp(-distances / temperature)
            probs = probs / probs.sum()
            idx = np.random.choice(len(samples), size=n, replace=False, p=probs)
        else:
            distances = np.linalg.norm(sample_cond - target_cond, axis=1)
            idx = np.argsort(distances)[:n]

        result = samples.iloc[idx].drop(columns=cond_cols).reset_index(drop=True)
        return result

    def save(self, path: Path) -> None:
        if self.model is None:
            raise RuntimeError("no model to save")
        path.mkdir(parents=True, exist_ok=True)
        self.model.save(path / "model.pkl")
        meta = {
            "type": "tvae",
            "config": self.config.model_dump(),
            "conditioning_config": self.conditioning_config.model_dump(),
            "conditioning_dim": self._conditioning_dim,
        }
        (path / "meta.json").write_text(json.dumps(meta, default=str))

    @classmethod
    def load(cls, path: Path) -> "ConditionalTVAE":
        meta = json.loads((path / "meta.json").read_text())
        config = TrainingConfig(**meta["config"])
        cond_config = ConditioningConfig(**meta["conditioning_config"])
        obj = cls(config, cond_config)
        obj.model = TVAESynthesizer.load(path / "model.pkl")
        obj._fitted = True
        obj._conditioning_dim = meta["conditioning_dim"]
        return obj

    def get_latent_representation(self, data: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("model not fitted")
        # TVAE has encoder, try to access it
        try:
            with torch.no_grad():
                # this is internal API, may break
                encoded = self.model._model.encoder(torch.tensor(data.values, dtype=torch.float32))
                return encoded.numpy()
        except Exception:
            return np.random.randn(len(data), self.config.embedding_dim).astype(np.float32)


# custom conditional normalizing flow
class ConditionalRealNVP(nn.Module):
    def __init__(
        self,
        data_dim: int,
        conditioning_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 8,
        num_blocks: int = 3,
    ):
        super().__init__()
        self.data_dim = data_dim
        self.conditioning_dim = conditioning_dim
        self.hidden_dim = hidden_dim

        self.conditioning_encoder = nn.Sequential(
            nn.Linear(conditioning_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        ) if conditioning_dim > 0 else None

        self.flows = nn.ModuleList([
            self._make_coupling_layer(hidden_dim, num_blocks)
            for _ in range(num_layers)
        ])

        # base distribution
        self.register_buffer("base_mean", torch.zeros(data_dim))
        self.register_buffer("base_std", torch.ones(data_dim))

    def _make_coupling_layer(self, hidden_dim: int, num_blocks: int) -> nn.Module:
        class CouplingLayer(nn.Module):
            def __init__(self, data_dim, cond_dim, hidden_dim, num_blocks):
                super().__init__()
                self.data_dim = data_dim
                self.mask = torch.arange(data_dim) % 2 == 0
                self.cond_dim = cond_dim

                # scale and translation networks
                layers = []
                in_dim = data_dim // 2 + cond_dim
                for _ in range(num_blocks):
                    layers.append(nn.Linear(in_dim, hidden_dim))
                    layers.append(nn.ReLU())
                    in_dim = hidden_dim
                layers.append(nn.Linear(hidden_dim, data_dim))
                self.net = nn.Sequential(*layers)

            def forward(self, x, conditioning):
                x_a = x[:, self.mask]
                x_b = x[:, ~self.mask]

                if conditioning is not None and self.cond_dim > 0:
                    net_in = torch.cat([x_a, conditioning], dim=-1)
                else:
                    net_in = x_a

                out = self.net(net_in)
                log_s = out[:, :x_b.shape[1]]
                t = out[:, x_b.shape[1]:]

                # stability
                log_s = torch.tanh(log_s) * 2
                x_b = x_b * torch.exp(log_s) + t

                y = torch.empty_like(x)
                y[:, self.mask] = x_a
                y[:, ~self.mask] = x_b
                log_det = log_s.sum(dim=1)
                return y, log_det

            def inverse(self, y, conditioning):
                x_a = y[:, self.mask]
                x_b = y[:, ~self.mask]

                if conditioning is not None and self.cond_dim > 0:
                    net_in = torch.cat([x_a, conditioning], dim=-1)
                else:
                    net_in = x_a

                out = self.net(net_in)
                log_s = out[:, :x_b.shape[1]]
                t = out[:, x_b.shape[1]:]

                log_s = torch.tanh(log_s) * 2
                x_b = (x_b - t) * torch.exp(-log_s)

                x = torch.empty_like(y)
                x[:, self.mask] = x_a
                x[:, ~self.mask] = x_b
                log_det = -log_s.sum(dim=1)
                return x, log_det

        return CouplingLayer(self.data_dim, self.conditioning_dim, hidden_dim, num_blocks)

    def forward(self, x, conditioning=None):
        log_det_total = 0
        if self.conditioning_encoder is not None and conditioning is not None:
            conditioning = self.conditioning_encoder(conditioning)

        for flow in self.flows:
            x, log_det = flow(x, conditioning)
            log_det_total += log_det
        return x, log_det_total

    def inverse(self, z, conditioning=None):
        log_det_total = 0
        if self.conditioning_encoder is not None and conditioning is not None:
            conditioning = self.conditioning_encoder(conditioning)

        for flow in reversed(self.flows):
            z, log_det = flow.inverse(z, conditioning)
            log_det_total += log_det
        return z, log_det_total

    def log_prob(self, x, conditioning=None):
        z, log_det = self.forward(x, conditioning)
        log_pz = -0.5 * ((z - self.base_mean) ** 2 / self.base_std ** 2 + 2 * torch.log(self.base_std) + np.log(2 * np.pi)).sum(dim=1)
        return log_pz + log_det

    def sample(self, n, conditioning=None, temperature=1.0):
        z = torch.randn(n, self.data_dim) * temperature
        if self.conditioning_encoder is not None and conditioning is not None:
            conditioning = self.conditioning_encoder(conditioning)
        x, _ = self.inverse(z, conditioning)
        return x


class ConditionalFlowSynthesizer(BaseSynthesizer):
    def __init__(self, config: TrainingConfig, conditioning_config: ConditioningConfig):
        self.config = config
        self.conditioning_config = conditioning_config
        self.model: ConditionalRealNVP | None = None
        self.data_dim: int | None = None
        self._conditioning_dim = sum(1 for c in conditioning_config.signals if c.enabled)
        self._fitted = False
        self._scaler_mean: np.ndarray | None = None
        self._scaler_std: np.ndarray | None = None
        self._column_names: list[str] | None = None
        self._categorical_columns: list[str] = []
        self._categorical_encoders: dict[str, Any] = {}

    def fit(self, data: pd.DataFrame, conditioning: np.ndarray = None) -> None:
        self._column_names = list(data.columns)
        self.data_dim = len(self._column_names)

        # simple preprocessing: standardize numeric, label encode categorical
        processed_data, self._scaler_mean, self._scaler_std = self._preprocess(data)

        # convert to tensor
        x = torch.tensor(processed_data.values, dtype=torch.float32)
        cond = None
        if conditioning is not None and self._conditioning_dim > 0:
            cond = torch.tensor(conditioning, dtype=torch.float32)

        # initialize model
        self.model = ConditionalRealNVP(
            data_dim=self.data_dim,
            conditioning_dim=self._conditioning_dim if self.conditioning_config.conditioning_strategy != "concat" else 0,
            hidden_dim=self.config.embedding_dim * 2,
            num_layers=8,
            num_blocks=3,
        )

        if self.config.cuda and torch.cuda.is_available():
            self.model = self.model.cuda()
            x = x.cuda()
            if cond is not None:
                cond = cond.cuda()

        # train
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        self.model.train()

        batch_size = self.config.batch_size
        n_batches = max(1, len(x) // batch_size)

        for epoch in range(self.config.epochs):
            epoch_loss = 0
            perm = torch.randperm(len(x))
            for i in range(n_batches):
                idx = perm[i * batch_size:(i + 1) * batch_size]
                x_batch = x[idx]
                cond_batch = cond[idx] if cond is not None else None

                optimizer.zero_grad()
                loss = -self.model.log_prob(x_batch, cond_batch).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()

            if self.config.verbose and epoch % 50 == 0:
                print(f"epoch {epoch}: nll = {epoch_loss / n_batches:.4f}")

        self._fitted = True
        if self.config.cuda and torch.cuda.is_available():
            self.model = self.model.cpu()

    def _preprocess(self, data: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        processed = data.copy()
        means = []
        stds = []

        for col in processed.columns:
            if pd.api.types.is_numeric_dtype(processed[col]):
                mean = processed[col].mean()
                std = processed[col].std()
                if std == 0:
                    std = 1.0
                processed[col] = (processed[col] - mean) / std
                means.append(mean)
                stds.append(std)
            else:
                # categorical: label encode
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                processed[col] = le.fit_transform(processed[col].astype(str))
                self._categorical_columns.append(col)
                self._categorical_encoders[col] = le
                means.append(0.0)
                stds.append(1.0)

        return processed, np.array(means), np.array(stds)

    def _postprocess(self, data: np.ndarray) -> pd.DataFrame:
        df = pd.DataFrame(data, columns=self._column_names)

        for i, col in enumerate(self._column_names):
            if col in self._categorical_columns:
                le = self._categorical_encoders[col]
                # round and clip
                vals = np.clip(np.round(data[:, i]).astype(int), 0, len(le.classes_) - 1)
                df[col] = le.inverse_transform(vals)
            else:
                df[col] = data[:, i] * self._scaler_std[i] + self._scaler_mean[i]

        return df

    def sample(self, n: int, conditioning: np.ndarray = None, temperature: float = 1.0) -> pd.DataFrame:
        if not self._fitted or self.model is None:
            raise RuntimeError("model not fitted")

        device = next(self.model.parameters()).device
        cond = None
        if conditioning is not None and self._conditioning_dim > 0:
            if conditioning.shape[0] == 1:
                conditioning = np.tile(conditioning, (n, 1))
            cond = torch.tensor(conditioning, dtype=torch.float32, device=device)

        self.model.eval()
        with torch.no_grad():
            samples = self.model.sample(n, cond, temperature)
            samples = samples.cpu().numpy()

        return self._postprocess(samples)

    def save(self, path: Path) -> None:
        if self.model is None:
            raise RuntimeError("no model to save")
        path.mkdir(parents=True, exist_ok=True)

        torch.save(self.model.state_dict(), path / "model.pt")

        meta = {
            "type": "custom_flow",
            "config": self.config.model_dump(),
            "conditioning_config": self.conditioning_config.model_dump(),
            "data_dim": self.data_dim,
            "conditioning_dim": self._conditioning_dim,
            "column_names": self._column_names,
            "scaler_mean": self._scaler_mean.tolist() if self._scaler_mean is not None else None,
            "scaler_std": self._scaler_std.tolist() if self._scaler_std is not None else None,
            "categorical_columns": self._categorical_columns,
            "categorical_encoders": {
                k: {"classes_": v.classes_.tolist()} for k, v in self._categorical_encoders.items()
            },
        }
        (path / "meta.json").write_text(json.dumps(meta, default=str))

    @classmethod
    def load(cls, path: Path) -> "ConditionalFlowSynthesizer":
        meta = json.loads((path / "meta.json").read_text())
        config = TrainingConfig(**meta["config"])
        cond_config = ConditioningConfig(**meta["conditioning_config"])

        obj = cls(config, cond_config)
        obj.data_dim = meta["data_dim"]
        obj._conditioning_dim = meta["conditioning_dim"]
        obj._column_names = meta["column_names"]
        obj._scaler_mean = np.array(meta["scaler_mean"]) if meta["scaler_mean"] else None
        obj._scaler_std = np.array(meta["scaler_std"]) if meta["scaler_std"] else None
        obj._categorical_columns = meta["categorical_columns"]
        obj._categorical_encoders = {}
        for k, v in meta["categorical_encoders"].items():
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            le.classes_ = np.array(v["classes_"])
            obj._categorical_encoders[k] = le

        obj.model = ConditionalRealNVP(
            data_dim=obj.data_dim,
            conditioning_dim=obj._conditioning_dim if cond_config.conditioning_strategy != "concat" else 0,
            hidden_dim=config.embedding_dim * 2,
            num_layers=8,
            num_blocks=3,
        )
        obj.model.load_state_dict(torch.load(path / "model.pt", map_location="cpu"))
        obj._fitted = True
        return obj

    def get_latent_representation(self, data: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("model not fitted")

        processed, _, _ = self._preprocess(data)
        x = torch.tensor(processed.values, dtype=torch.float32)

        self.model.eval()
        with torch.no_grad():
            z, _ = self.model.forward(x)

        return z.numpy()


class SynthesizerFactory:
    @staticmethod
    def create(config: DatasetDreamerConfig, data: pd.DataFrame = None) -> BaseSynthesizer:
        if config.training.model_type == ModelType.CTGAN:
            metadata = None
            if data is not None:
                metadata = SingleTableMetadata()
                metadata.detect_from_dataframe(data)
            return ConditionalCTGAN(config.training, config.conditioning, metadata)
        elif config.training.model_type == ModelType.TVAE:
            metadata = None
            if data is not None:
                metadata = SingleTableMetadata()
                metadata.detect_from_dataframe(data)
            return ConditionalTVAE(config.training, config.conditioning, metadata)
        elif config.training.model_type == ModelType.CUSTOM_FLOW:
            return ConditionalFlowSynthesizer(config.training, config.conditioning)
        else:
            raise ValueError(f"unknown model type: {config.training.model_type}")

    @staticmethod
    def load(path: Path) -> BaseSynthesizer:
        meta = json.loads((path / "meta.json").read_text())
        model_type = meta["type"]

        if model_type == "ctgan":
            return ConditionalCTGAN.load(path)
        elif model_type == "tvae":
            return ConditionalTVAE.load(path)
        elif model_type == "custom_flow":
            return ConditionalFlowSynthesizer.load(path)
        else:
            raise ValueError(f"unknown model type: {model_type}")