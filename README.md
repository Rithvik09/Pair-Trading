# Pairs Trader

Finds stocks that historically move together, flags when they drift apart, and tells you
to **short the one that ran up and buy the one that fell**, then to close once they meet again.

It never places orders. After each weekday close a scheduled job works out the exact orders
(ticker, buy/sell/short, whole-share quantity, which morning) and posts them to Discord and the
dashboard. You place them with your broker. A model book tracks the same trades and P&L.

Each run posts one recap to Discord: the model book's value after that close (change on the
day and since the start), every decision with its reason ("z +2.31 ≥ +2.0: SW ran ahead of
PKG…", "…but can't borrow PKG to short it"), then the order sheet. It also shows the cash in
use: the stock held, long and short, the margin that needs (`margin_rate` in config.toml, 50%
by default as under Reg T; your broker may want more), and what's left free, with a ⚠ if the
trades need more margin than the book is worth. The dashboard has the same decision log, an
equity curve of the book, and the cash in use.

## More than one strategy

Pairs trading is the first of several strategies (the list and the order they're being built
in: [ROADMAP.md](ROADMAP.md)). Each runs as its own paper book, a "sleeve", with its own
starting cash: pairs in `state/book.json`, the others in `state/<name>.json`. Because the
books are separate, each strategy's profit or loss is its own, and the recap ends with a line
per strategy and a total:

```
**By strategy**
Pairs: $100,410 (+$410, +0.41%)
Trend: $99,800 (-$200, -0.20%)
Total: $200,210 (+$210, +0.10%) on $200,000 started
```

The dashboard's Strategies tab has the same table, each book's value over time, and each
strategy's holdings and recent trades. Strategies are switched on and sized in `config.toml`
under `[strategies.<name>]`. None of them places orders either.

**Trend following** (`pairs/strategies/trend.py`, running on paper): for each of 12 ETFs
across stocks, bonds, credit, gold, commodities, property and the dollar, go long if it rose
over the past 12 months and short if it fell. Each position is sized so every ETF brings the
same expected volatility: `vol_target` (10% a year by default) is a budget split evenly over
the ETFs, so the book itself swings well under 10%, since they don't all move together. The
stock held is capped at 1.5 times the book (`max_gross`). It rebalances at the first close of each
month and is valued every day; its orders appear in the recap as `TREND SPY` REBALANCE.
These books hold whole shares to target weights (`pairs/portfolio.py`): trading costs and the
borrow fee on shorts come out of cash, and a holding less than a quarter off its target is
left alone rather than traded. Dividends are paid into cash for longs and charged to shorts,
worked out from how far Yahoo's adjusted prices restate the previous close; a split changes
the share count instead, with a ⚠ to check it against your broker.

## How it runs (free, private)

Everything stays private: the repo (with your trade history) is a private GitHub repo, and the
dashboard is a private Streamlit app that only you and people you invite can open.

| Piece | Where | What it does |
|---|---|---|
| `.github/workflows/daily.yml` | GitHub Actions, weekdays ~4:45pm ET (+ ~11:30pm ET catch-up) | Fetches closes and updates the books in `state/`, commits them, then posts the day's recap and new orders to Discord (only after the commit succeeds, so nothing is posted twice or lost) |
| `app.py` | Streamlit Community Cloud | Read-only dashboard: Orders, Strategies (profit and loss per strategy), Positions (with the book's equity curve), Decision log, Scanner, Pair explorer |
| `config.toml` | this repo | The watchlist, the pairs settings and the other strategies' settings. Edit it to change what's traded |

### One-time setup

1. **GitHub**: create a **private** repo, push this project, and merge it to `main`
   (scheduled workflows only run from the default branch). Private repos get 2,000 free
   Actions minutes a month; the daily job uses about 80. With GitHub's default $0 spending
   limit it can't run up a bill: jobs just stop if the minutes ever run out.
2. **Discord**: in your channel, *Edit Channel → Integrations → Webhooks → New Webhook → Copy
   Webhook URL*. In GitHub: *Settings → Secrets and variables → Actions → New repository
   secret*, name `DISCORD_WEBHOOK`, paste the URL. Treat it like a password: anyone with
   it can post to the channel. Never put it in a file.
3. **First run**: *Actions → daily step → Run workflow*. It creates `state/book.json`, and
   then runs itself every weekday.
4. **Dashboard**: sign in at share.streamlit.io with GitHub and, when asked, allow access
   to **private repositories** (or later: your avatar → *Settings → Linked accounts*).
   *Create app* → this repo, branch `main`, file `app.py`; under *Advanced settings* pick
   Python 3.12 and add the secret `GITHUB_REPO = "your-name/your-repo"` (makes the "edit
   config.toml" links work). An app from a private repo is private: only you can open it,
   after signing in. To let someone else see it: the app's *Share* button → invite by email.
   Free accounts may be limited in how many private apps they can have; you need just one.
   Streamlit redeploys on every push, so it shows each day's book after the daily commit.

### Changing what's traded

Edit `config.toml` on GitHub (pencil icon) and commit. The Scanner and Pair explorer tabs show
the exact line to add for a pair. The next daily run uses the new settings.

To find pairs worth adding, run `uv run python -m pairs.cli shortlist` locally (about two
minutes). It searches every pair of S&P 1500 stocks (large, mid and small caps) within the same
GICS sub-industry, about 15,000 pairs from `universe.csv`, and keeps those that:

- are cointegrated over 3 years (p < 0.10), revert in 5–30 days (half-life), and move together
  (correlation 0.6–0.97) with a positive hedge ratio
- backtest with a Sharpe ratio ≥ 0.75, a positive total return and at least 4 trades, after
  trading costs and borrow fees
- still backtest with a Sharpe ratio ≥ 0.25 when every order is filled a day late, since you
  trade the next morning, not at the close that triggered the signal

It keeps at most 3 pairs per sub-industry and 2 per stock, and prints the `config.toml` lines.
`--universe sp500` limits it to large caps; `--universe core` uses the hand-picked groups in
`pairs/data.py`.

**Why these filters, and how much to trust them:** `uv run python -m pairs.cli validate`
(about seven minutes) replays the shortlist as if run one, two, three and four years ago,
using only the data available then, and trades its picks through the following year, which
it never saw:

| Year starting | Pairs | Made money | Avg return per pair | Portfolio Sharpe |
|---|---|---|---|---|
| Oct 2022 | 83 | 54% | −0.1% | −0.0 |
| Oct 2023 | 58 | 59% | +2.2% | 1.6 |
| Oct 2024 | 40 | 68% | +3.2% | 1.2 |
| Oct 2025 | 39 | 49% | +0.1% | 0.1 |

The previous filters lost money in 3 of these 4 years. But these settings were chosen from
~1,400 combinations *using* these same years, which flatters them. The fair test: choose using
2022–2025 only, then score the newest year. Done that way, the chosen settings **lost money**
in 2025–26 (Sharpe −0.75), and 92% of all combinations tried lost money that year. So treat
the edge as unproven: it worked in 2023–2025 and has not in the last year. The model book keeps
score on the watchlist either way, so it's a cheap way to see whether it recovers before you
commit real money. Rerun `validate` every few months; it updates the newest row.

Two checks that sound sensible made things worse or no better in every year: also requiring
cointegration over the last year on its own, and requiring a profitable last year. The stock
list is today's index members (stocks that fell out are missing), which flatters all of these
numbers a little.

**How many trades at once:** as many as the cash covers. A new trade opens only if the book's
cash (realised P&L included) still covers the margin of every open trade plus the new one
(`margin_rate` × trade size each, so $100,000 at 50% and $10,000 a trade is 20 trades).
When a signal comes in with no cash left, it's compared with the open trades by how far each
gap still has to close (|z| minus the exit level). Only open trades that would close at a
profit after costs are considered; losing trades are left to recover. If the signal has at
least 0.5 more than the one of those with the least left, that trade is closed and the new one
opened, provided closing it frees enough cash. In replays of five 6-month windows (2024-26)
this rule finished level with not swapping at all, and no variant tried did reliably better. The order sheet marks that close "swapped for a better trade". Only signals
on the latest close are compared, so a catch-up run doesn't swap on stale prices.
`[paper] max_open_positions` (off by default) adds a hard cap on the number of trades.

Refresh `universe.csv` when the indexes change with
`uv run --with lxml python -m pairs.cli update-universe` and review the diff.

## Run locally

```bash
uv sync
uv run streamlit run app.py                 # the dashboard
uv run python -m pairs.cli scan             # print cointegrated pairs in the S&P 1500
uv run python -m pairs.cli shortlist        # ...that also pass the filters and backtest
uv run python -m pairs.cli validate         # score those filters on years they never saw
uv run python -m pairs.cli step             # update the books in state/, print any new orders
DISCORD_WEBHOOK_URL=... uv run python -m pairs.cli step --notify   # ...and post them
uv run pytest                               # tests (synthetic data, no network)
```

If you change dependencies, regenerate the file Streamlit Cloud installs from (CI checks it):
`uv export --no-dev --no-hashes --no-header --format requirements-txt -o requirements.txt`

## How it works

1. **Find pairs** (`pairs/scanner.py`): within each S&P 1500 sub-industry (`universe.csv`), or each
   group in `pairs/data.py`, test every pair for cointegration (Engle-Granger). Low p-value = the
   gap between them has tended to close. `shortlist` then filters and backtests them (above).
2. **Measure the gap** (`pairs/backtest.py: signal_frame`): spread = log(Y) − β·log(X) − α with a
   rolling hedge ratio β, turned into a z-score over a rolling window.
3. **Trade it** (`pairs/signals.py`): z ≥ +2 → short Y, buy X. z ≤ −2 → buy Y, short X.
   Only while the hedge ratio is positive (the two move together); otherwise a "pair trade"
   would short or buy both legs, which is a directional bet, so it's skipped.
   Close once z is back inside ±0.5 (or has crossed through it). Stop out at |z| ≥ 4 and don't
   re-enter until the gap has normalised. The backtest and the paper book share this one rule.
4. **Backtest** with no look-ahead (today's signal earns tomorrow's move), trading costs and
   borrow fees (below).
5. **Order sheet + model book** (`pairs/paper.py`, `pairs/orders.py`): positions are whole
   shares (half the trade size per leg, rounded down) and persist in `state/book.json` along
   with every order, every decision and its reason, and the book's value after each close.
   Signals use a day's close, so orders are for the **next NYSE session's
   open** (weekends and exchange holidays skipped). Each run replays any days it missed, so a
   skipped run doesn't hide a stop-out; orders from a missed day are flagged LATE.

## Borrowing costs and short availability (`pairs/borrow.py`)

Every trade shorts one leg (with a negative hedge ratio, a short spread shorts both and a long
spread shorts neither). For each ticker the app looks up whether it can be borrowed and the fee:

1. `borrow.csv` (copy `borrow.csv.example`) — your broker's numbers. `fee_rate` is an annual
   decimal: `0.0025` = 0.25%/yr. Blank cells fall through to the next source.
2. Interactive Brokers' public short-stock file, fetched once a day. Not listed = nothing to lend.
3. Otherwise: 0.25%/yr and shortability "unverified".

- **Backtest**: the short leg pays its fee per calendar day held (actual/360). Today's fee is
  applied to all history, and trades are not removed for past unavailability.
- **Paper trading**: a trade whose short leg is known to be unborrowable is skipped
  (`unshortable GS`); unknown doesn't block. Borrow accrues actual/360 at the fee locked in
  at entry and is deducted on close; open positions show borrow owed so far.
- **Scanner**: a `borrow` column, and `⚠ can't short X` on signals you couldn't act on.

## Caveats

- The model book assumes you fill at the signal-day close; you'll actually fill at the next
  open, so real results differ by the overnight move. It also assumes you placed every order.
- Order dates skip weekends and the regular NYSE holidays, not one-off closures (e.g. a
  national day of mourning). Reference prices are Yahoo's adjusted closes.
- GitHub can start scheduled runs late, and disables schedules in repos with no activity for
  60 days (the daily commit counts as activity).
- Yahoo sometimes rate-limits cloud servers. The fetch retries; if no prices arrive at all
  the run fails, GitHub emails you, and the next run catches up. A single ticker with no data
  gets a ⚠ line in Discord.
- Scanning ~15,000 pairs produces many false positives by chance, and picking pairs by their
  own backtest flatters them: the shortlist's own Sharpe ratios (often 1–2) are far above what
  its picks earned on unseen years (−0.1% to +3.2% per pair-year). Trust `validate`, not the
  shortlist's backtest numbers.
- Small-cap pairs (S&P 600) are cheaper to get wrong: wider spreads at the open, and borrow
  can be scarce. The shortlist charges each stock's borrow fee, but not slippage.
- Pairs that worked historically can break permanently (mergers, business changes) — that's
  what the stop-loss is for.
- Prices are Yahoo adjusted closes. Borrow availability can vanish mid-trade (recalls); the
  paper book doesn't model recalls, and positions opened before borrow tracking accrue nothing.
