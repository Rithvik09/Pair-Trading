"""Load config.toml: the watchlist and strategy settings the daily job trades with."""

import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path

from pairs.backtest import BacktestConfig
from pairs.paths import CONFIG_PATH

_PAPER_KEYS = {"notional", "starting_cash"}


@dataclass(frozen=True)
class Config:
    strategy: BacktestConfig = field(default_factory=BacktestConfig)
    notional: float = 10_000.0  # gross dollars per pair trade (both legs together)
    starting_cash: float = 100_000.0
    watchlist: list[tuple[str, str]] = field(default_factory=list)


def _check(section: str, got: dict, allowed: set[str]) -> None:
    unknown = sorted(set(got) - allowed)
    if unknown:
        raise ValueError(f"config.toml [{section}]: unknown setting(s) {unknown}; "
                         f"allowed: {sorted(allowed)}")


def load_config(path: Path = CONFIG_PATH) -> Config:
    raw = tomllib.loads(Path(path).read_text())
    _check("top level", raw, {"watchlist", "strategy", "paper"})
    strat = raw.get("strategy", {})
    _check("strategy", strat, {f.name for f in fields(BacktestConfig)})
    paper = raw.get("paper", {})
    _check("paper", paper, _PAPER_KEYS)
    watchlist = []
    for pair in raw.get("watchlist", []):
        if len(pair) != 2:
            raise ValueError(f"config.toml watchlist: each entry is [Y, X], got {pair}")
        watchlist.append((pair[0].strip().upper(), pair[1].strip().upper()))
    return Config(strategy=BacktestConfig(**strat), watchlist=watchlist,
                  **{k: float(v) for k, v in paper.items()})
