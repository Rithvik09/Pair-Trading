# Pairs Trader

Finds stocks that historically move together, flags when they drift apart, and tells you
to **short the one that ran up and buy the one that fell**, then to close once they meet again.

It never places orders. After each weekday close a scheduled job works out the exact orders
(ticker, buy/sell/short, whole-share quantity, which morning) and posts them to Discord and the
dashboard. You place them with your broker. A model book tracks the same trades and P&L.

## How it runs (free, private)

Everything stays private: the repo (with your trade history) is a private GitHub repo, and the
dashboard is a private Streamlit app that only you and people you invite can open.

| Piece | Where | What it does |
|---|---|---|
| `.github/workflows/daily.yml` | GitHub Actions, weekdays ~4:45pm ET | Fetches closes and updates `state/book.json`, commits it, then posts the new orders to Discord (only after the commit succeeds, so nothing is posted twice or lost) |
| `app.py` | Streamlit Community Cloud | Read-only dashboard: Orders, Positions, Scanner, Pair explorer |
| `config.toml` | this repo | The watchlist and strategy settings. Edit it to change what's traded |

### One-time setup

1. **GitHub**: create a **private** repo, push this project, and merge it to `main`
   (scheduled workflows only run from the default branch). Private repos get 2,000 free
   Actions minutes a month; the daily job uses about 80. With GitHub's default $0 spending
   limit it can't run up a bill: jobs just stop if the minutes ever run out.
2. **Discord**: in your channel, *Edit Channel → Integrations → Webhooks → New Webhook → Copy
   Webhook URL*. In GitHub: *Settings → Secrets and variables → Actions → New repository
   secret*, name `DISCORD_WEBHOOK_URL`, paste the URL. Treat it like a password: anyone with
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

## Run locally

```bash
uv sync
uv run streamlit run app.py                 # the dashboard
uv run python -m pairs.cli scan             # print cointegrated pairs in the built-in universe
uv run python -m pairs.cli step             # update state/book.json, print any new orders
DISCORD_WEBHOOK_URL=... uv run python -m pairs.cli step --notify   # ...and post them
uv run pytest                               # tests (synthetic data, no network)
```

If you change dependencies, regenerate the file Streamlit Cloud installs from (CI checks it):
`uv export --no-dev --no-hashes --no-header --format requirements-txt -o requirements.txt`

## How it works

1. **Find pairs** (`pairs/scanner.py`): within each sector group in `pairs/data.py`, test every
   pair for cointegration (Engle-Granger). Low p-value = the gap between them has tended to close.
2. **Measure the gap** (`pairs/backtest.py: signal_frame`): spread = log(Y) − β·log(X) − α with a
   rolling hedge ratio β, turned into a z-score over a rolling window.
3. **Trade it** (`pairs/signals.py`): z ≥ +2 → short Y, buy X. z ≤ −2 → buy Y, short X.
   Close once z is back inside ±0.5 (or has crossed through it). Stop out at |z| ≥ 4 and don't
   re-enter until the gap has normalised. The backtest and the paper book share this one rule.
4. **Backtest** with no look-ahead (today's signal earns tomorrow's move), trading costs and
   borrow fees (below).
5. **Order sheet + model book** (`pairs/paper.py`, `pairs/orders.py`): positions are whole
   shares (half the trade size per leg, rounded down) and persist in `state/book.json` along
   with every order. Signals use a day's close, so orders are for the **next NYSE session's
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
- Scanning hundreds of pairs at p < 0.05 produces false positives by chance. Backtest first.
- Pairs that worked historically can break permanently (mergers, business changes) — that's
  what the stop-loss is for.
- Prices are Yahoo adjusted closes. Borrow availability can vanish mid-trade (recalls); the
  paper book doesn't model recalls, and positions opened before borrow tracking accrue nothing.
