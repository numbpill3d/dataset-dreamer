"""external signal fetchers for conditioning."""
from __future__ import annotations

import asyncio
import hashlib
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import ephem
import httpx
import numpy as np
from git import Repo, InvalidGitRepositoryError

from dataset_dreamer.config import settings, SignalSource


@dataclass(slots=True)
class SignalValue:
    source: SignalSource
    value: float
    timestamp: datetime
    metadata: dict[str, Any]


class SignalFetcher(ABC):
    @abstractmethod
    async def fetch(self) -> SignalValue:
        pass

    @abstractmethod
    def fetch_sync(self) -> SignalValue:
        pass


class MoonPhaseFetcher(SignalFetcher):
    def __init__(self, lat: float = 0.0, lon: float = 0.0):
        self.observer = ephem.Observer()
        self.observer.lat = str(lat)
        self.observer.lon = str(lon)
        self.observer.elevation = 0

    async def fetch(self) -> SignalValue:
        return await asyncio.to_thread(self.fetch_sync)

    def fetch_sync(self) -> SignalValue:
        self.observer.date = datetime.now(timezone.utc)
        moon = ephem.Moon(self.observer)
        moon.compute(self.observer)

        # moon phase: 0.0 = new, 0.5 = full, 1.0 = next new
        phase = moon.phase / 100.0
        # normalize to [-1, 1] where 0 = new, 1 = full, -1 = new (waning)
        normalized = np.sin(2 * np.pi * phase)

        return SignalValue(
            source=SignalSource.MOON_PHASE,
            value=float(normalized),
            timestamp=datetime.now(timezone.utc),
            metadata={
                "phase_percent": moon.phase,
                "illumination": moon.phase / 100.0,
                "distance_au": float(moon.earth_distance),
                "lat": self.observer.lat,
                "lon": self.observer.lon,
            },
        )


class SolarWindFetcher(SignalFetcher):
    NOAA_URL = "https://services.swpc.noaa.gov/json/solar-wind/mag-1-day.json"
    CACHE_TTL = 300  # 5 minutes

    def __init__(self):
        self._cache: SignalValue | None = None
        self._cache_time = 0.0

    async def fetch(self) -> SignalValue:
        if time.time() - self._cache_time < self.CACHE_TTL and self._cache:
            return self._cache

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(self.NOAA_URL)
            resp.raise_for_status()
            data = resp.json()

        if not data:
            raise ValueError("no solar wind data available")

        latest = data[-1]
        # bz_gsm: north-south component in GSM coords. negative = southward = geo-effective
        bz = float(latest.get("bz_gsm", 0.0))
        bt = float(latest.get("bt", 0.0))
        speed = float(latest.get("speed", 400.0))
        density = float(latest.get("density", 5.0))

        # composite geo-effectiveness index
        # negative bz + high speed + high density = more geo-effective
        geo_eff = max(0.0, -bz / 20.0) * (speed / 400.0) * (density / 5.0)
        geo_eff = np.clip(geo_eff, 0.0, 5.0)
        normalized = np.tanh(geo_eff)  # squash to [0, 1)

        self._cache = SignalValue(
            source=SignalSource.SOLAR_WIND,
            value=float(normalized),
            timestamp=datetime.fromtimestamp(float(latest["time_tag"]) / 1000, tz=timezone.utc),
            metadata={
                "bz_gsm": bz,
                "bt": bt,
                "speed": speed,
                "density": density,
                "temperature": float(latest.get("temperature", 1e5)),
                "geo_effectiveness": float(geo_eff),
            },
        )
        self._cache_time = time.time()
        return self._cache

    def fetch_sync(self) -> SignalValue:
        return asyncio.run(self.fetch())


class GitActivityFetcher(SignalFetcher):
    def __init__(self, repo_path: Path | str = "."):
        self.repo_path = Path(repo_path).resolve()

    async def fetch(self) -> SignalValue:
        return await asyncio.to_thread(self.fetch_sync)

    def fetch_sync(self) -> SignalValue:
        try:
            repo = Repo(self.repo_path)
        except (InvalidGitRepositoryError, ValueError):
            return SignalValue(
                source=SignalSource.GIT_ACTIVITY,
                value=0.0,
                timestamp=datetime.now(timezone.utc),
                metadata={"error": "not a git repository", "path": str(self.repo_path)},
            )

        # analyze recent commits (last 30 days)
        since = datetime.now(timezone.utc).timestamp() - (30 * 86400)
        commits = list(repo.iter_commits(since=since, max_count=500))

        if not commits:
            return SignalValue(
                source=SignalSource.GIT_ACTIVITY,
                value=0.0,
                timestamp=datetime.now(timezone.utc),
                metadata={"commits_30d": 0},
            )

        # compute activity metrics
        commit_times = [c.committed_datetime.timestamp() for c in commits]
        intervals = np.diff(sorted(commit_times))

        # frequency: commits per day
        freq = len(commits) / 30.0

        # burstiness: coefficient of variation of intervals
        burstiness = float(np.std(intervals) / np.mean(intervals)) if len(intervals) > 1 and np.mean(intervals) > 0 else 0.0

        # message entropy (vocabulary diversity)
        messages = " ".join(c.message for c in commits)
        words = messages.lower().split()
        unique_words = len(set(words))
        total_words = len(words)
        lexical_diversity = unique_words / total_words if total_words > 0 else 0.0

        # files changed per commit
        files_changed = np.mean([len(c.stats.files) for c in commits])

        # composite activity score
        activity = np.tanh(
            freq / 10.0 +  # 10 commits/day = high
            burstiness / 2.0 +  # burstiness up to ~2
            lexical_diversity * 2.0 +  # diversity up to 1
            files_changed / 20.0  # 20 files/commit = high
        )

        return SignalValue(
            source=SignalSource.GIT_ACTIVITY,
            value=float(activity),
            timestamp=datetime.now(timezone.utc),
            metadata={
                "commits_30d": len(commits),
                "freq_per_day": freq,
                "burstiness": burstiness,
                "lexical_diversity": lexical_diversity,
                "avg_files_changed": float(files_changed),
                "active_branch": repo.active_branch.name,
                "head_hash": repo.head.commit.hexsha[:8],
            },
        )


class CryptoVolatilityFetcher(SignalFetcher):
    def __init__(self, symbols: list[str] = None, exchange: str = "binance"):
        self.symbols = symbols or ["BTC/USDT", "ETH/USDT"]
        self.exchange_name = exchange
        self._cache: dict[str, SignalValue] = {}
        self._cache_time = 0.0
        self.CACHE_TTL = 60  # 1 minute

    async def fetch(self) -> SignalValue:
        return await asyncio.to_thread(self.fetch_sync)

    def fetch_sync(self) -> SignalValue:
        if time.time() - self._cache_time < self.CACHE_TTL and self._cache:
            return self._aggregate(self._cache.values())

        try:
            import ccxt
            exchange = getattr(ccxt, self.exchange_name)({"enableRateLimit": True})
        except (ImportError, AttributeError):
            return self._fallback()

        values = {}
        for symbol in self.symbols:
            try:
                ohlcv = exchange.fetch_ohlcv(symbol, "1h", limit=168)  # 1 week hourly
                if len(ohlcv) < 24:
                    continue

                closes = np.array([c[4] for c in ohlcv])
                returns = np.diff(np.log(closes))

                # realized volatility (annualized)
                rv = np.std(returns) * np.sqrt(24 * 365)

                # parkinson volatility (high-low)
                highs = np.array([c[2] for c in ohlcv])
                lows = np.array([c[3] for c in ohlcv])
                pv = np.sqrt(np.mean((np.log(highs / lows)) ** 2) / (4 * np.log(2))) * np.sqrt(24 * 365)

                # volume-weighted volatility
                volumes = np.array([c[5] for c in ohlcv])
                vwv = np.average(np.abs(returns), weights=volumes[1:]) * np.sqrt(24 * 365)

                values[symbol] = SignalValue(
                    source=SignalSource.CRYPTO_VOLATILITY,
                    value=float(np.clip(rv, 0.0, 3.0)),  # cap at 300% annualized
                    timestamp=datetime.now(timezone.utc),
                    metadata={
                        "symbol": symbol,
                        "realized_vol": float(rv),
                        "parkinson_vol": float(pv),
                        "volume_weighted_vol": float(vwv),
                        "current_price": float(closes[-1]),
                        "price_change_24h": float((closes[-1] - closes[-24]) / closes[-24]) if len(closes) >= 24 else 0.0,
                    },
                )
            except Exception:
                continue

        if not values:
            return self._fallback()

        self._cache = values
        self._cache_time = time.time()
        return self._aggregate(values.values())

    def _aggregate(self, signals: list[SignalValue]) -> SignalValue:
        if not signals:
            return self._fallback()

        # weight by market cap proxy (use BTC as anchor)
        weights = {}
        for s in signals:
            sym = s.metadata.get("symbol", "")
            if "BTC" in sym:
                weights[sym] = 0.5
            elif "ETH" in sym:
                weights[sym] = 0.3
            else:
                weights[sym] = 0.2

        total_w = sum(weights.values())
        weights = {k: v / total_w for k, v in weights.items()}

        agg_value = sum(s.value * weights.get(s.metadata.get("symbol", ""), 1.0 / len(signals)) for s in signals)

        return SignalValue(
            source=SignalSource.CRYPTO_VOLATILITY,
            value=float(np.clip(agg_value, 0.0, 1.0)),
            timestamp=datetime.now(timezone.utc),
            metadata={
                "components": {s.metadata.get("symbol", f"unknown_{i}"): s.value for i, s in enumerate(signals)},
                "weights": weights,
            },
        )

    def _fallback(self) -> SignalValue:
        # deterministic pseudo-random based on time
        seed = int(time.time() / 3600)  # changes hourly
        rng = np.random.default_rng(seed)
        val = float(rng.beta(2, 5))  # skewed toward low volatility
        return SignalValue(
            source=SignalSource.CRYPTO_VOLATILITY,
            value=val,
            timestamp=datetime.now(timezone.utc),
            metadata={"fallback": True, "seed": seed},
        )


class CustomSignalFetcher(SignalFetcher):
    def __init__(self, func, source_name: str = "custom"):
        self.func = func
        self.source_name = source_name

    async def fetch(self) -> SignalValue:
        if asyncio.iscoroutinefunction(self.func):
            return await self.func()
        return await asyncio.to_thread(self.func)

    def fetch_sync(self) -> SignalValue:
        if asyncio.iscoroutinefunction(self.func):
            return asyncio.run(self.func())
        return self.func()


class SignalAggregator:
    def __init__(self, fetchers: dict[SignalSource, SignalFetcher] = None):
        self.fetchers = fetchers or self._default_fetchers()

    def _default_fetchers(self) -> dict[SignalSource, SignalFetcher]:
        return {
            SignalSource.MOON_PHASE: MoonPhaseFetcher(),
            SignalSource.SOLAR_WIND: SolarWindFetcher(),
            SignalSource.GIT_ACTIVITY: GitActivityFetcher(),
            SignalSource.CRYPTO_VOLATILITY: CryptoVolatilityFetcher(),
        }

    async def fetch_all(self, enabled: list[SignalSource] = None) -> dict[SignalSource, SignalValue]:
        enabled = enabled or list(self.fetchers.keys())
        tasks = {src: self.fetchers[src].fetch() for src in enabled if src in self.fetchers}
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)

        output = {}
        for src, result in zip(tasks.keys(), results):
            if isinstance(result, Exception):
                output[src] = SignalValue(
                    source=src,
                    value=0.0,
                    timestamp=datetime.now(timezone.utc),
                    metadata={"error": str(result)},
                )
            else:
                output[src] = result
        return output

    def fetch_all_sync(self, enabled: list[SignalSource] = None) -> dict[SignalSource, SignalValue]:
        enabled = enabled or list(self.fetchers.keys())
        output = {}
        for src in enabled:
            if src in self.fetchers:
                try:
                    output[src] = self.fetchers[src].fetch_sync()
                except Exception as e:
                    output[src] = SignalValue(
                        source=src,
                        value=0.0,
                        timestamp=datetime.now(timezone.utc),
                        metadata={"error": str(e)},
                    )
        return output

    def to_conditioning_vector(self, signals: dict[SignalSource, SignalValue], signal_configs: dict[SignalSource, Any]) -> np.ndarray:
        """convert signals to conditioning vector with weights."""
        vec = []
        for src, config in signal_configs.items():
            if src in signals and config.enabled:
                val = signals[src].value * config.weight
                vec.append(val)
            else:
                vec.append(0.0)
        return np.array(vec, dtype=np.float32)