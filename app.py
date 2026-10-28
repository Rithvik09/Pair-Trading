"""Pairs trading dashboard (view-only).  Run with:  uv run streamlit run app.py

The daily job (GitHub Actions -> `pairs.cli step`) is the only thing that changes the book.
The watchlist and strategy live in config.toml; edit that to change what is traded.
"""

import os
import re
from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from pairs.backtest import BacktestConfig, backtest, signal_frame
from pairs.borrow import DEFAULT_FEE, OVERRIDES_PATH, borrow_label, borrow_table, fee_label
from pairs.config import load_config
from pairs.data import UNIVERSE, load_prices
from pairs.notify import EVENT_LABELS
from pairs.paper import PaperBook, pair_key, short_legs
from pairs.paths import BOOK_PATH
from pairs.scanner import scan
from pairs.signals import trade_opens

st.set_page_config(page_title="Pairs Trader", layout="wide")
st.title("Pairs Trader")
st.caption("Tracks pairs of stocks that move together. When one runs ahead, the daily job "
           "says to short it and buy the laggard, then to close once they meet again. "
           "This app never places orders: it writes the order sheet, you place them.")

config = load_config()
live = config.strategy
GITHUB_REPO = os.environ.get("GITHUB_REPO")  # "owner/name", set in Streamlit secrets
CONFIG_EDIT_URL = (f"https://github.com/{GITHUB_REPO}/edit/main/config.toml"
                   if GITHUB_REPO else None)

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Research settings")
    st.caption("For the Scanner and Pair explorer only. The daily job uses config.toml.")
    period = st.selectbox("History", ["1y", "2y", "3y", "5y"], index=2)
    entry = st.slider("Enter when |z| ≥", 1.0, 3.5, live.entry, 0.1,
                      help="How many standard deviations apart before opening a trade.")
    exit_ = st.slider("Exit when |z| ≤", 0.0, 1.5, live.exit, 0.1,
                      help="Close once the spread has come back this close to normal.")
    stop = st.slider("Stop-loss at |z| ≥", 3.0, 6.0, live.stop, 0.25,
                     help="Bail out if the gap keeps widening — the relationship may have broken.")
    lookback = st.slider("Hedge-ratio lookback (days)", 40, 250, live.lookback, 10)
    z_window = st.slider("Z-score window (days)", 10, 90, live.z_window, 5)
    cost_bps = st.number_input("Trading cost (bps per side)", 0.0, 50.0, live.cost_bps, 0.5)

cfg = BacktestConfig(lookback=lookback, z_window=z_window, entry=entry, exit=exit_,
                     stop=stop, cost_bps=cost_bps)


def watchlist_hint(pairs: list[str]) -> None:
    """How to start trading pairs: add them to config.toml (the app itself is read-only)."""
    lines = "\n".join(f'  ["{a}", "{b}"],' for a, b in (p.split("/") for p in pairs))
    st.markdown("To trade " + ("these" if len(pairs) > 1 else "this") + ", add to the "
                "`watchlist` in config.toml"
                + (f" ([edit on GitHub]({CONFIG_EDIT_URL}))" if CONFIG_EDIT_URL else "") + ":")
    st.code(lines, language="toml")


# Caches are bounded and free-text tickers validated, so no viewer (including anyone the app
# is shared with) can make it fire unlimited Yahoo requests or fill memory.
TICKER = re.compile(r"[A-Z][A-Z0-9.\-]{0,9}")


@st.cache_data(ttl=3600, max_entries=50, show_spinner="Downloading prices…")
def prices_for(tickers: tuple[str, ...], period: str) -> pd.DataFrame:
    return load_prices(list(tickers), period=period)


@st.cache_data(ttl=3600, max_entries=20, show_spinner="Testing pairs for cointegration…")
def run_scan(groups: tuple[str, ...], period: str, max_p: float) -> pd.DataFrame:
    g = {k: UNIVERSE[k] for k in groups}
    prices = prices_for(tuple(sorted({t for v in g.values() for t in v})), period)
    return scan(prices, g, max_pvalue=max_p)


@st.cache_data(ttl=3600, max_entries=50, show_spinner="Checking short availability…")
def _borrow_cached(tickers: tuple[str, ...], overrides_mtime: float) -> dict:
    return borrow_table(list(tickers))


def borrow_for(tickers: tuple[str, ...]) -> dict:
    # keyed on borrow.csv's mtime so edits show up immediately
    mtime = OVERRIDES_PATH.stat().st_mtime if OVERRIDES_PATH.exists() else 0.0
    return _borrow_cached(tickers, mtime)


def describe(y: str, x: str, z: float, pos: int, beta: float, borrow: dict) -> str:
    if pos == -1:
        state = f"In trade: short {y} / long {x}"
    elif pos == 1:
        state = f"In trade: long {y} / short {x}"
    elif pd.isna(z):
        return "—"
    elif abs(z) >= cfg.entry:
        # Flat while diverged means the strategy won't trade: it stopped out and is waiting
        # for the gap to normalise, or the gap is already beyond the stop level.
        why = "beyond stop" if abs(z) >= cfg.stop else "cooling off after a stop-out"
        return f"Diverged, no trade ({why})"
    else:
        return "Normal"
    stuck = [t for t in short_legs((y, x), pos, beta) if borrow[t].shortable is False]
    return state + (f" ⚠ can't short {', '.join(stuck)}" if stuck else "")


def borrow_source_note(borrow: dict) -> str:
    sources = {b.source for b in borrow.values()}
    overridden = sorted(t for t, b in borrow.items() if b.source == "override")
    note = f" borrow.csv overrides: {', '.join(overridden)}." if overridden else ""
    if "default" in sources:
        return (f"IBKR borrow feed unreachable: using a {fee_label(DEFAULT_FEE)} default fee, "
                "shortability unverified. Add your broker's numbers in borrow.csv." + note)
    return "Borrow fees and availability from IBKR's short-stock file." + note


book = PaperBook.load(BOOK_PATH, starting_cash=config.starting_cash)
tab_orders, tab_book, tab_log, tab_scan, tab_pair = st.tabs(
    ["Orders", "Positions", "Decision log", "Scanner", "Pair explorer"])

# ---------------------------------------------------------------- orders
with tab_orders:
    today = date.today().isoformat()
    upcoming = [o for o in book.orders if o["execute_on"] and o["execute_on"] >= today]
    last_run = max(book.last_bar.values(), default=None)
    st.caption(f"Latest close processed: {last_run or 'none yet'}. The daily job runs after "
               "each weekday close and posts new orders to Discord.")
    if not upcoming:
        st.info("No orders to place. Nothing on the watchlist has crossed a threshold.")
    for when in sorted({o["execute_on"] for o in upcoming}):
        st.subheader(f"Place at the open, {date.fromisoformat(when):%a %b %-d}")
        for o in (o for o in upcoming if o["execute_on"] == when):
            st.markdown(f"**{o['pair']}** · {EVENT_LABELS.get(o['event'], o['event'])} · "
                        f"z {o['z']:+.2f} at the {o['signal_date']} close")
            st.dataframe(pd.DataFrame(o["legs"]), hide_index=True, width="stretch",
                         column_config={"ref_price": st.column_config.NumberColumn(
                             "ref price (signal close)", format="$%.2f")})
    if book.orders:
        st.subheader("Order history")
        hist = pd.DataFrame([
            {"execute on": o["execute_on"], "pair": o["pair"],
             "event": EVENT_LABELS.get(o["event"], o["event"]),
             "orders": "; ".join(f"{leg['side']} {leg['qty']} {leg['ticker']}"
                                 for leg in o["legs"])}
            for o in reversed(book.orders)])
        st.dataframe(hist, hide_index=True, width="stretch")

# ---------------------------------------------------------------- scanner
with tab_scan:
    c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
    groups = c1.multiselect("Sectors", list(UNIVERSE), default=list(UNIVERSE))
    max_p = c2.number_input("Max p-value", 0.001, 0.2, 0.05, 0.005, format="%.3f")
    min_corr = c3.number_input("Min return corr.", 0.0, 1.0, 0.5, 0.05)
    hl_max = c4.number_input("Max half-life (days)", 2, 250, 60)

    if groups:
        found = run_scan(tuple(groups), period, max_p)
        found = found[(found["corr"] >= min_corr) & (found["half_life"].between(1, hl_max))]
        if found.empty:
            st.info("No pairs pass these filters. Try loosening the p-value or correlation.")
        else:
            tickers = tuple(sorted(set(found["y"]) | set(found["x"])))
            px = prices_for(tickers, period)
            borrow = borrow_for(tickers)
            zs, states = [], []
            for r in found.itertuples():
                d = px[[r.y, r.x]].dropna()
                f = signal_frame(d[r.y], d[r.x], cfg)
                zs.append(f["z"].iloc[-1])
                states.append(describe(r.y, r.x, f["z"].iloc[-1], int(f["position"].iloc[-1]),
                                       r.beta, borrow))
            found = found.assign(z_now=zs, status=states)
            found["pair"] = [pair_key(p) for p in zip(found["y"], found["x"], strict=True)]
            found["borrow"] = [f"{r.y} {borrow_label(borrow[r.y])} · {r.x} {borrow_label(borrow[r.x])}"
                               for r in found.itertuples()]
            st.dataframe(
                found[["pair", "group", "z_now", "status", "borrow", "pvalue", "half_life",
                       "corr", "beta"]],
                hide_index=True, width="stretch",
                column_config={
                    "z_now": st.column_config.NumberColumn("z now", format="%+.2f"),
                    "pvalue": st.column_config.NumberColumn("p-value", format="%.4f"),
                    "half_life": st.column_config.NumberColumn("half-life (d)", format="%.1f"),
                    "corr": st.column_config.NumberColumn(format="%.2f"),
                    "beta": st.column_config.NumberColumn("hedge ratio", format="%.3f"),
                },
            )
            st.caption(borrow_source_note(borrow))
            st.caption("Testing many pairs at p < 0.05 means roughly 1 in 20 will look "
                       "cointegrated by luck — backtest before trusting one. Near-identical "
                       "funds (e.g. VOO/IVV) pass easily but their gaps are too small to trade.")
            add = st.multiselect("Pick pairs to trade", found["pair"].tolist())
            if add:
                watchlist_hint(add)

# ---------------------------------------------------------------- pair explorer
with tab_pair:
    c1, c2 = st.columns(2)
    y_t = c1.text_input("Stock Y", "KO").strip().upper()
    x_t = c2.text_input("Stock X", "PEP").strip().upper()
    bad = [t for t in (y_t, x_t) if not TICKER.fullmatch(t)]
    if bad or y_t == x_t:
        st.error("Enter two different tickers, e.g. KO and PEP.")
        st.stop()
    px = prices_for(tuple(sorted({y_t, x_t})), period)
    if y_t not in px or x_t not in px:
        st.error("Couldn't load prices for one of those tickers.")
    else:
        df = px[[y_t, x_t]].dropna()
        borrow = borrow_for(tuple(sorted({y_t, x_t})))
        res = backtest(df[y_t], df[x_t], cfg,
                       borrow_fees=(borrow[y_t].fee_rate, borrow[x_t].fee_rate))
        s = res.stats
        m = st.columns(3) + st.columns(3)
        m[0].metric("Total return", f"{s['total_return']:+.1%}")
        m[1].metric("Sharpe", f"{s['sharpe']:.2f}")
        m[2].metric("Max drawdown", f"{s['max_drawdown']:.1%}")
        m[3].metric("Trades", s["trades"])
        m[4].metric("Win rate", "—" if pd.isna(s["win_rate"]) else f"{s['win_rate']:.0%}")
        m[5].metric("Avg hold (d)", "—" if pd.isna(s["avg_days"]) else f"{s['avg_days']:.0f}")
        unborrowable = [t for t in (y_t, x_t) if borrow[t].shortable is False]
        if unborrowable:
            st.warning(f"{', '.join(unborrowable)} can't be shorted right now, so trades that "
                       "short it would be skipped in paper trading. The backtest below still "
                       "includes them, since past borrow availability isn't known.")
        st.caption(f"Borrow: {y_t} {borrow_label(borrow[y_t])}, {x_t} {borrow_label(borrow[x_t])}"
                   f" — ${s['borrow_paid']:,.0f} paid over the backtest (today's rates applied to "
                   f"all history). {borrow_source_note(borrow)}")

        fr = res.frame
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                            row_heights=[0.35, 0.35, 0.3],
                            subplot_titles=("Prices (rebased to 100)", "Spread z-score",
                                            "Strategy equity"))
        for t in (y_t, x_t):
            fig.add_trace(go.Scatter(x=df.index, y=df[t] / df[t].iloc[0] * 100, name=t), 1, 1)
        fig.add_trace(go.Scatter(x=fr.index, y=fr["z"], name="z", line={"color": "#888"}), 2, 1)
        for lvl, dash in ((cfg.entry, "dash"), (-cfg.entry, "dash"),
                          (cfg.exit, "dot"), (-cfg.exit, "dot")):
            fig.add_hline(y=lvl, line_dash=dash, line_width=1, row=2, col=1)
        opens = trade_opens(fr["position"])
        for side, color, label in ((-1, "#d62728", f"short {y_t}/buy {x_t}"),
                                   (1, "#2ca02c", f"buy {y_t}/short {x_t}")):
            pts = fr[opens & fr["position"].eq(side)]
            fig.add_trace(go.Scatter(x=pts.index, y=pts["z"], mode="markers", name=label,
                                     marker={"color": color, "size": 9, "symbol": "triangle-up"
                                                 if side == 1 else "triangle-down"}), 2, 1)
        fig.add_trace(go.Scatter(x=res.equity.index, y=res.equity, name="equity",
                                 line={"color": "#1f77b4"}), 3, 1)
        fig.update_layout(height=780, margin={"t": 40, "b": 20}, legend={"orientation": "h"})
        st.plotly_chart(fig, width="stretch")

        st.subheader("Backtest trades")
        st.dataframe(res.trades, hide_index=True, width="stretch",
                     column_config={"pnl": st.column_config.NumberColumn(format="$%.0f"),
                                    "entry_z": st.column_config.NumberColumn(format="%+.2f"),
                                    "exit_z": st.column_config.NumberColumn(format="%+.2f")})
        if (y_t, x_t) not in config.watchlist:
            watchlist_hint([pair_key((y_t, x_t))])

# ---------------------------------------------------------------- positions
with tab_book:
    s = config.strategy
    st.caption(f"Watchlist: {', '.join(pair_key(p) for p in config.watchlist) or 'empty'}. "
               f"Enter at |z| ≥ {s.entry}, exit at {s.exit}, stop at {s.stop}, "
               f"${config.notional:,.0f} per trade"
               + (f" · [edit config.toml]({CONFIG_EDIT_URL})" if CONFIG_EDIT_URL else "") + ".")
    if book.equity:
        eq = pd.DataFrame(book.equity).assign(date=lambda d: pd.to_datetime(d["date"]))
        start = book.starting_cash
        now = eq["equity"].iloc[-1]
        st.subheader("Paper book value")
        st.caption(f"After each close: what the book would be worth if every open "
                   f"position were closed at that close, after costs and borrow fees. "
                   f"\\${now:,.0f} now, {now / start - 1:+.2%} since the \\${start:,.0f} start "
                   f"on {eq['date'].iloc[0]:%b %-d, %Y}.")
        fig = go.Figure(go.Scatter(x=eq["date"], y=eq["equity"], mode="lines+markers",
                                   name="value", line={"color": "#1f77b4"}))
        fig.add_hline(y=start, line_dash="dot", line_width=1,
                      annotation_text="start", annotation_position="bottom right")
        fig.update_layout(height=320, margin={"t": 10, "b": 10}, yaxis_tickprefix="$",
                          showlegend=False)
        st.plotly_chart(fig, width="stretch")
        last = book.equity[-1]
        if "margin" in last:  # snapshots from before cash in use was tracked don't have it
            c = st.columns(3)
            c[0].metric("Stock held (long + short)", f"${last['gross']:,.0f}")
            c[1].metric("Margin it needs", f"${last['margin']:,.0f}",
                        help=f"{config.margin_rate:.0%} of the stock held (margin_rate in "
                             "config.toml). Your broker's requirement may differ.")
            c[2].metric("Free for new trades", f"${last['free']:,.0f}")
            if last["free"] < 0:
                st.warning("The open trades need more margin than the book is worth: a real "
                           "account would get a margin call.")
    unreal = 0.0
    if book.positions:
        st.subheader("Open positions")
        tick = tuple(sorted({p[k] for p in book.positions.values() for k in ("y", "x")}))
        hist = prices_for(tick, "1y").ffill()
        last = hist.iloc[-1]
        rows = []
        for k, p in book.positions.items():
            book.rebase_entry(k, hist[p["y"]], hist[p["x"]])
            owed = book.borrow_accrued(k, hist.index[-1].date())
            u = book.mark(k, last[p["y"]], last[p["x"]]) - p.get("costs", 0.0) - owed
            unreal += u
            side = -1 if p["side"] == "short_spread" else 1
            shorts = short_legs((p["y"], p["x"]), side, p["beta"])
            rows.append({"pair": k, "short": ", ".join(shorts) or "—",
                         "long": ", ".join(t for t in (p["y"], p["x"]) if t not in shorts),
                         "opened": p["entry_date"], "entry z": p["entry_z"],
                         "borrow owed": owed, "unrealised P&L": u})
        money = st.column_config.NumberColumn(format="$%.2f")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                     column_config={"unrealised P&L": money, "borrow owed": money,
                                    "entry z": st.column_config.NumberColumn(format="%+.2f")})
        st.caption("Unrealised P&L is after opening costs and borrow fees accrued so far.")

    realised = sum(t["pnl"] for t in book.closed)
    m = st.columns(3)
    m[0].metric("Cash (incl. realised P&L)", f"${book.cash:,.0f}")
    m[1].metric("Realised P&L", f"${realised:+,.0f}")
    m[2].metric("Unrealised P&L", f"${unreal:+,.0f}")
    if book.closed:
        st.subheader("Closed trades")
        st.dataframe(pd.DataFrame(book.closed), hide_index=True, width="stretch")

# ---------------------------------------------------------------- decision log
with tab_log:
    st.caption("Every decision the daily job made and why, newest first: trades opened and "
               "closed, stops, and signals it didn't act on.")
    if not book.log:
        st.info("No decisions yet. They appear after the daily job's next run.")
    else:
        log = pd.DataFrame(reversed(book.log))
        log["decision"] = [EVENT_LABELS.get(a, "not traded") for a in log["action"]]
        if not st.checkbox("Include signals that weren't traded", value=True):
            log = log[log["action"].isin(EVENT_LABELS)]
        st.dataframe(log[["date", "pair", "decision", "z", "reason"]], hide_index=True,
                     width="stretch",
                     column_config={"z": st.column_config.NumberColumn(format="%+.2f"),
                                    "reason": st.column_config.TextColumn(width="large")})
