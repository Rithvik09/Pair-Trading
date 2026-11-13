# Strategies to add

Built one at a time, in this order. Each runs as its own paper book (a "sleeve") with its own
starting cash, so the daily recap and the dashboard can show profit and loss per strategy
and for all of them together. Like the pairs trader, none of them places orders.

| # | Strategy | Status | Data |
|---|---|---|---|
| 0 | Pairs trading (stat arb on cointegrated pairs) | running | daily closes |
| 1 | Trend following: long ETFs in uptrends, short those in downtrends, across asset classes | running (paper) | daily ETF closes |
| 2 | Risk parity: stocks, bonds, gold, commodities sized to equal risk, monthly rebalance | running (paper) | daily ETF closes |
| 3 | Momentum long/short: long the past year's winners, short the losers, monthly | running (paper) | universe closes |
| 4 | Low volatility long/short (quality half needs fundamentals) | running (paper) | closes (+ fundamentals for quality) |
| 5 | Stock vs. sector ETF stat arb | running (paper) | daily closes |
| 6 | Short-term reversal: buy last week's losers, short the winners | to do | daily closes |
| 7 | Discretionary long/short: your own picks, sized and tracked by the app | to do | your picks + closes |
| 8 | Merger arbitrage | needs a data source | deal announcements and terms |
| 9 | Post-earnings drift | needs a data source | earnings estimates and actuals |
| 10 | Volatility selling | needs a data source | options or VIX futures |

8 to 10 need data the free Yahoo feed doesn't provide; pick a source before building them.
