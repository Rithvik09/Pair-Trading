"""Where everything lives. Stdlib only, so any module can import it cheaply."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.toml"  # watchlist + strategy settings (edit this)
OVERRIDES_PATH = ROOT / "borrow.csv"  # your broker's borrow numbers (optional)
STATE_DIR = ROOT / "state"  # committed: the paper book and its order history
BOOK_PATH = STATE_DIR / "book.json"
DATA_DIR = ROOT / "data"  # git-ignored caches: prices, IBKR borrow file
