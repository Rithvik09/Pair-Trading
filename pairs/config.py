"""Load config.toml: the watchlist and strategy settings the daily job trades with."""

import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path

from pairs.backtest import BacktestConfig
from pairs.paths import CONFIG_PATH
from pairs.strategies import STRATEGIES

_PAPER_KEYS = {"notional", "starting_cash", "max_open_positions", "margin_rate"}
# settings every [strategies.<name>] section takes, on top of the strategy's own
_SLEEVE_DEFAULTS = {"enabled": True, "starting_cash": 100_000.0}


@dataclass(frozen=True)
class Config:
    strategy: BacktestConfig = field(default_factory=BacktestConfig)
    notional: float = 10_000.0  # gross dollars per pair trade (both legs together)
    starting_cash: float = 100_000.0
    max_open_positions: int | None = None  # None = no cap
    margin_rate: float = 0.5  # margin held per $ of stock, long and short (Reg T initial: 50%)
    watchlist: list[tuple[str, str]] = field(default_factory=list)
    # the other strategies, by name: each one's settings, defaults filled in (strategy_params)
    strategies: dict[str, dict] = field(default_factory=dict)


def _check(section: str, got: dict, allowed: set[str]) -> None:
    unknown = sorted(set(got) - allowed)
    if unknown:
        raise ValueError(f"config.toml [{section}]: unknown setting(s) {unknown}; "
                         f"allowed: {sorted(allowed)}")


def strategy_params(name: str, section: dict, cost_bps: float = BacktestConfig.cost_bps) -> dict:
    """A [strategies.<name>] section, checked, with defaults for whatever it leaves out.
    cost_bps is the pairs strategy's unless the section sets its own."""
    where = f"strategies.{name}"
    if name not in STRATEGIES:
        raise ValueError(f"config.toml [{where}]: unknown strategy; "
                         f"available: {sorted(STRATEGIES)}")
    defaults = {**_SLEEVE_DEFAULTS, "cost_bps": cost_bps, **STRATEGIES[name].DEFAULTS}
    _check(where, section, set(defaults))
    out = {**defaults, **section}
    if out["enabled"] is not True and out["enabled"] is not False:
        raise ValueError(f"config.toml [{where}]: enabled is true or false, "
                         f"got {out['enabled']!r}")
    for k, default in defaults.items():
        if isinstance(default, float):
            if isinstance(out[k], bool) or not isinstance(out[k], int | float) \
                    or out[k] < 0 or (out[k] == 0 and k != "cost_bps"):
                raise ValueError(f"config.toml [{where}]: {k} must be a number above 0, "
                                 f"got {out[k]!r}")
            out[k] = float(out[k])
    if "universe" in out:
        u = out["universe"]
        if not isinstance(u, list) or not u or not all(isinstance(t, str) and t.strip()
                                                       for t in u):
            raise ValueError(f"config.toml [{where}]: universe is a list of tickers, "
                             f"e.g. [\"SPY\", \"TLT\"], got {u!r}")
        out["universe"] = [t.strip().upper() for t in u]
    return out


def load_config(path: Path = CONFIG_PATH) -> Config:
    raw = tomllib.loads(Path(path).read_text())
    _check("top level", raw, {"watchlist", "strategy", "paper", "strategies"})
    strat = raw.get("strategy", {})
    _check("strategy", strat, {f.name for f in fields(BacktestConfig)})
    paper = raw.get("paper", {})
    _check("paper", paper, _PAPER_KEYS)
    watchlist = []
    for pair in raw.get("watchlist", []):
        if len(pair) != 2:
            raise ValueError(f"config.toml watchlist: each entry is [Y, X], got {pair}")
        watchlist.append((pair[0].strip().upper(), pair[1].strip().upper()))
    cap = paper.get("max_open_positions")
    if cap is not None and (not isinstance(cap, int) or cap < 1):
        raise ValueError(f"config.toml [paper]: max_open_positions must be a whole number "
                         f">= 1, got {cap}")
    rate = paper.get("margin_rate")
    if rate is not None and not 0 < float(rate) <= 1:
        raise ValueError(f"config.toml [paper]: margin_rate is a fraction of the stock held, "
                         f"between 0 and 1 (0.5 = 50%), got {rate}")
    paper = {k: v if k == "max_open_positions" else float(v) for k, v in paper.items()}
    strategy = BacktestConfig(**strat)
    strategies = {name: strategy_params(name, section, strategy.cost_bps)
                  for name, section in raw.get("strategies", {}).items()}
    return Config(strategy=strategy, watchlist=watchlist, strategies=strategies, **paper)
