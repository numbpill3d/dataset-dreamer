"""signal aggregation utilities."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

import numpy as np

from dataset_dreamer.config import SignalConfig, SignalSource
from dataset_dreamer.signals.fetchers import SignalAggregator, SignalValue


async def fetch_signals(configs: list[SignalConfig]) -> dict[SignalSource, SignalValue]:
    aggregator = SignalAggregator()
    enabled_sources = [c.source for c in configs if c.enabled]
    signals = await aggregator.fetch_all(enabled_sources)

    # apply weights
    for config in configs:
        if config.source in signals and config.enabled:
            signals[config.source].value *= config.weight

    return signals


def fetch_signals_sync(configs: list[SignalConfig]) -> dict[SignalSource, SignalValue]:
    aggregator = SignalAggregator()
    enabled_sources = [c.source for c in configs if c.enabled]
    return aggregator.fetch_all_sync(enabled_sources)


def signals_to_vector(signals: dict[SignalSource, SignalValue], configs: list[SignalConfig]) -> np.ndarray:
    """convert signals to conditioning vector matching config order."""
    vec = []
    for config in configs:
        if config.enabled and config.source in signals:
            vec.append(signals[config.source].value)
        else:
            vec.append(0.0)
    return np.array(vec, dtype=np.float32)


def create_signal_metadata(signals: dict[SignalSource, SignalValue]) -> dict[str, Any]:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "signals": {
            src.value: {
                "value": sig.value,
                "timestamp": sig.timestamp.isoformat(),
                "metadata": sig.metadata,
            }
            for src, sig in signals.items()
        },
    }