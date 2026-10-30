"""Arithmetic for the dashboard's sizing calculator: what a trade size and account size add
up to under the paper book's own rules. It informs a decision; it doesn't make one."""

from pairs.signals import leg_weights


def smallest_trade(y_px: float, x_px: float, beta: float) -> float:
    """Smallest trade size (both legs together) that buys at least one whole share of each
    leg, by the same rule PaperBook.step uses. Anything below it is skipped as "too small"."""
    w_y, w_x = leg_weights(beta)
    return max(y_px / w_y, x_px / abs(w_x))


def loss_rates(closed: list[dict], open_: list[tuple[float, float]]) -> list[float]:
    """P&L per dollar of trade size: closed trades, then open ones as (P&L now, size)."""
    rates = [t["pnl"] / t["notional"] for t in closed if t.get("notional")]
    return rates + [pnl / size for pnl, size in open_ if size]


def plan(account: float, notional: float, max_open: int, margin_rate: float,
         smallest: dict[str, float], loss_rates: list[float]) -> dict:
    """Totals with every slot filled: stock held, margin, what's left of the account, the
    pairs this size can't trade (largest requirement first), and the worst trade so far
    scaled to this size (None until there is one)."""
    gross = notional * max_open
    margin = gross * margin_rate
    worst = min(loss_rates, default=None)
    return {"gross": gross, "margin": margin, "free": account - margin,
            "too_small": sorted((k for k, v in smallest.items() if v > notional),
                                key=lambda k: -smallest[k]),
            "worst_rate": worst, "worst_loss": None if worst is None else worst * notional}
