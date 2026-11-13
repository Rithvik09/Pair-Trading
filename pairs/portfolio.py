"""A paper book held to target weights: what every strategy other than pairs runs on.

A strategy says what share of the book each ticker should be; the book works out the whole
shares to trade, writes the orders, and tracks cash. No broker, no real money.
"""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from pairs.orders import next_session
from pairs.paper import LOG_KEEP

# A saved close that today reads less than 0.6x or more than 1.6x what it did was restated for
# a split, not a dividend (no fund pays 40% in one go; the smallest common split is 3-for-2).
SPLIT = (0.6, 1.6)


@dataclass
class WeightBook:
    path: Path
    cash: float = 100_000.0
    starting_cash: float | None = None  # what the book started with; set from cash if None
    holdings: dict[str, int] = field(default_factory=dict)  # shares per ticker, negative = short
    orders: list[dict] = field(default_factory=list)  # every order the strategy asked for
    log: list[dict] = field(default_factory=list)  # every trade and why, newest last
    equity: list[dict] = field(default_factory=list)  # the book's value after each close
    last_run_warnings: list[str] = field(default_factory=list)
    last_rebalance: str | None = None  # close the holdings were last reset to the targets at
    last_prices: dict[str, float] = field(default_factory=dict)  # per holding, at the last close
    price_dates: dict[str, str] = field(default_factory=dict)  # ...and which close that was

    def __post_init__(self):
        if self.starting_cash is None:
            self.starting_cash = self.cash

    @property
    def label(self) -> str:
        """How the strategy is named on the order sheet: state/trend.json -> 'TREND'."""
        return self.path.stem.upper()

    @classmethod
    def load(cls, path, starting_cash: float = 100_000.0) -> "WeightBook":
        path = Path(path)
        if not path.exists():
            return cls(path=path, cash=starting_cash)
        d = json.loads(path.read_text())
        return cls(path=path, cash=d["cash"], starting_cash=d.get("starting_cash", starting_cash),
                   holdings={t: int(n) for t, n in d.get("holdings", {}).items()},
                   orders=d.get("orders", []), log=d.get("log", []), equity=d.get("equity", []),
                   last_run_warnings=d.get("last_run_warnings", []),
                   last_rebalance=d.get("last_rebalance"), last_prices=d.get("last_prices", {}),
                   price_dates=d.get("price_dates", {}))

    def to_dict(self) -> dict:
        return {"cash": self.cash, "starting_cash": self.starting_cash,
                "holdings": self.holdings, "last_rebalance": self.last_rebalance,
                "last_prices": self.last_prices, "price_dates": self.price_dates,
                "orders": self.orders,
                "last_run_warnings": self.last_run_warnings, "log": self.log[-LOG_KEEP:],
                "equity": self.equity}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.to_dict(), indent=2, default=str))

    def _price(self, ticker: str, prices: dict[str, float]) -> float | None:
        """This close's price, or None if there isn't a usable one."""
        px = prices.get(ticker)
        return float(px) if px is not None and math.isfinite(px) and px > 0 else None

    def _marks(self, prices: dict[str, float]) -> dict[str, float]:
        """A price for every holding: this close's, else the last one the book saw."""
        marks = {t: self._price(t, prices) or self.last_prices.get(t) for t in self.holdings}
        return {t: px for t, px in marks.items() if px is not None}

    def credit_dividends(self, restated: dict[str, float]) -> float:
        """Pay the dividends since each holding was last priced. Returns the cash added
        (negative if the shorts owed more than the longs earned).

        Prices are dividend-adjusted: once a stock pays $2, its earlier $100 close reads $98.
        So the price alone shows a holder losing every dividend. `restated` is, per ticker,
        what the close in price_dates reads today; what it fell by from the price the book
        saved then is the dividend per share: paid to a long, charged to a short, as a broker
        would. Call it before rebalance, on the holdings that were there on the day.
        A close that reads higher than it did is a data correction, not cash, and is ignored.

        A split restates earlier closes too, by far more (SPLIT below). Then the share count
        is what changes: 100 shares at $100 become 200 at $50, a part share is paid in cash,
        and a warning says so. The book's value is the same either way.
        """
        paid = 0.0
        for t, n in list(self.holdings.items()):
            was, now = self.last_prices.get(t), self._price(t, restated)
            if was is None or now is None:
                continue
            if SPLIT[0] <= now / was <= SPLIT[1]:
                paid += n * max(was - now, 0.0)
                continue
            exact = n * was / now
            shares = round(exact) if abs(exact - round(exact)) < 0.01 else math.trunc(exact)
            self.cash += (exact - shares) * now
            self.last_prices[t] = now
            if shares:
                self.holdings[t] = shares
            else:
                del self.holdings[t]
            self.last_run_warnings.append(
                f"⚠ {self.label}: {t} looks to have split (its earlier closes now read "
                f"{now / was:.2f}x what they did), so its holding went from {n} to {shares} "
                "shares. Check that against your broker.")
        self.cash += paid
        return paid

    def accrue_borrow(self, date, fees: dict[str, float]) -> float:
        """Charge the borrow fee on the shorts held since the last recorded close: an annual
        rate per ticker (missing = 0), per calendar day (actual/360), on their value at that
        close. Taken from cash. Call it before rebalance, so the fee is on what was actually
        held. A rerun on the same close charges nothing more."""
        if not self.equity:
            return 0.0
        days = (pd.Timestamp(str(date)) - pd.Timestamp(self.equity[-1]["date"])).days
        owed = sum(abs(n) * self.last_prices.get(t, 0.0) * fees.get(t, 0.0)
                   for t, n in self.holdings.items() if n < 0) * max(days, 0) / 360
        self.cash -= owed
        return owed

    def rebalance(self, date, prices: dict[str, float], targets: dict[str, float],
                  reasons: dict[str, str], cost_bps: float = 0.0, band: float = 0.25,
                  margin_rate: float = 0.5) -> list[dict]:
        """Trade the book to `targets`: each ticker's share of the book's value at `prices`,
        negative for a short. Returns the new orders (also appended to self.orders).

        Positions are whole shares, rounded toward zero. A holding that is off its target by
        less than `band` of the target's size is left alone, so small drifts aren't churned;
        a target of 0 always closes in full, and a change of direction always trades.
        Targets that add up to more stock than the book's value can carry as margin are all
        scaled down together. A holding with no target is closed. A ticker with no price at
        this close keeps its holding and gets a warning.
        cost_bps is charged on the value of every share traded.
        """
        marks = self._marks(prices)
        equity = self.cash + sum(n * marks.get(t, 0.0) for t, n in self.holdings.items())
        gross = sum(abs(w) for w in targets.values())
        if gross * margin_rate > 1:
            targets = {t: w / (gross * margin_rate) for t, w in targets.items()}
        new = []
        for t in sorted(set(targets) | set(self.holdings)):
            w, held = targets.get(t, 0.0), self.holdings.get(t, 0)
            px = self._price(t, prices)
            if px is None:
                if w or held:
                    self.last_run_warnings.append(
                        f"⚠ {self.label}: no price for {t} at the {date} close, so its "
                        "holding was left as it is")
                continue
            want = math.trunc(w * equity / px + math.copysign(1e-9, w))  # 9.9999999 is 10
            flip = want * held < 0
            if want == held or (w != 0 and not flip
                                and abs(want - held) * px < band * abs(w) * equity):
                continue
            self.cash -= (want - held) * px + abs(want - held) * px * cost_bps / 1e4
            if want:
                self.holdings[t] = want
            else:
                del self.holdings[t]
            self.last_prices[t], self.price_dates[t] = px, str(date)
            order = {"pair": f"{self.label} {t}", "event": "rebalance", "signal_date": str(date),
                     "execute_on": next_session(date), "z": 0.0,
                     "legs": [{"ticker": t, "side": side, "qty": qty, "ref_price": px}
                              for side, qty in _legs(held, want)]}
            new.append(order)
            self.log.append({"date": str(date), "pair": order["pair"], "action": "rebalance",
                             "z": 0.0, "reason": reasons.get(t, "no longer a target: closed")})
        self.orders += new
        self.last_rebalance = str(date)
        return new

    def record_equity(self, date, prices: dict[str, float], margin_rate: float = 0.5) -> dict:
        """Snapshot the book at a close, with the same keys as PaperBook.record_equity.
        `unrealised` is the stock held at `prices` (shorts count negative), so `equity` =
        cash + unrealised, and equity - starting_cash is the book's profit or loss to date.
        One entry per date; a rerun replaces it.
        A holding whose price is missing is marked at the last price the book saw."""
        marks = self._marks(prices)
        self.last_prices = marks
        self.price_dates = {t: str(date) if self._price(t, prices) else self.price_dates.get(t)
                            for t in marks}
        held = sum(n * marks.get(t, 0.0) for t, n in self.holdings.items())
        gross = sum(abs(n) * marks.get(t, 0.0) for t, n in self.holdings.items())
        equity = self.cash + held
        snap = {"date": str(date), "cash": self.cash, "unrealised": held, "equity": equity,
                "open": len(self.holdings), "gross": gross, "margin": gross * margin_rate,
                "free": equity - gross * margin_rate}
        self.equity = sorted([e for e in self.equity if e["date"] != snap["date"]] + [snap],
                             key=lambda e: e["date"])
        return snap


def _legs(held: int, want: int) -> list[tuple[str, int]]:
    """(side, quantity) to get from `held` shares to `want`. Crossing zero takes two orders:
    close what's there, then open the other way."""
    if held > 0 > want:
        return [("SELL", held), ("SELL SHORT", -want)]
    if held < 0 < want:
        return [("BUY TO COVER", -held), ("BUY", want)]
    if want > held:
        return [("BUY TO COVER" if held < 0 else "BUY", want - held)]
    return [("SELL" if held > 0 else "SELL SHORT", held - want)]
