"""Where everything lives. Stdlib only, so any module can import it cheaply."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.toml"  # watchlist + strategy settings (edit this)
UNIVERSE_PATH = ROOT / "universe.csv"  # S&P 1500 by sub-industry (pairs.cli update-universe)
OVERRIDES_PATH = ROOT / "borrow.csv"  # your broker's borrow numbers (optional)
STATE_DIR = ROOT / "state"  # committed: the paper books and their order history
BOOK_PATH = STATE_DIR / "book.json"
DATA_DIR = ROOT / "data"  # git-ignored caches: prices, IBKR borrow file


def strategy_path(name: str, state_dir: Path = STATE_DIR) -> Path:
    """Where a strategy's own paper book is kept (pairs keeps book.json)."""
    return Path(state_dir) / f"{name}.json"
