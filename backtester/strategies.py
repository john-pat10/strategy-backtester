"""Investment strategies.

Each strategy only decides how to deploy cash that has already arrived. The
deposit schedule is set separately, so the same strategy covers several cases:
`InvestImmediately` with one deposit is lump-sum investing, and with monthly
deposits it is classic dollar-cost averaging.
"""

from .engine import MIN_TRADE, Context, Order, Strategy


def drawdown(ctx: Context, ticker: str, lookback: int) -> float:
    """How far today's price is below its high over the last `lookback` days (e.g. -0.12)."""
    window = ctx.history(ticker, lookback)
    return window[-1] / window.max() - 1


class InvestImmediately(Strategy):
    """Invest every dollar the day it arrives."""

    def __init__(self, ticker, name="Invest immediately"):
        self.ticker = ticker
        self.name = name

    def on_day(self, ctx):
        if ctx.portfolio.cash >= MIN_TRADE:
            return [Order(self.ticker, ctx.portfolio.cash)]
        return []


class SpreadOut(Strategy):
    """Invest a lump sum in equal monthly pieces over `months` months.

    The uninvested part sits in cash earning interest until its turn comes.
    """

    def __init__(self, ticker, months, name=None):
        self.ticker = ticker
        self.months = months
        self.name = name or f"Spread over {months} mo"
        self._installments = 0
        self._month = None

    def on_day(self, ctx):
        month = (ctx.date.year, ctx.date.month)
        if month == self._month or self._installments >= self.months:
            return []
        self._month = month
        remaining = self.months - self._installments
        self._installments += 1
        return [Order(self.ticker, ctx.portfolio.cash / remaining)]


class BuyTheDip(Strategy):
    """Hold cash until the market is `threshold` below its recent high, then invest all of it.

    While the market stays in a dip, new deposits are invested right away.
    If the market never dips that far, the money stays in cash.
    """

    def __init__(self, ticker, threshold=0.10, lookback=252, name=None):
        self.ticker = ticker
        self.threshold = threshold
        self.lookback = lookback
        self.name = name or f"Buy the dip ({threshold:.0%})"

    def on_day(self, ctx):
        in_dip = drawdown(ctx, self.ticker, self.lookback) <= -self.threshold
        if in_dip and ctx.portfolio.cash >= MIN_TRADE:
            return [Order(self.ticker, ctx.portfolio.cash)]
        return []


class DCAWithDipReserve(Strategy):
    """DCA most of each deposit, keep the rest in cash, and spend that reserve on dips.

    Each deposit: `invest_fraction` is invested immediately; the rest joins the
    cash reserve. Each time the market falls past one of `levels` (e.g. 10%,
    20%, 30% below its recent high), an equal share of the reserve is deployed,
    so the deepest level uses whatever is left. Levels reset when the market
    makes a new high.
    """

    def __init__(self, ticker, invest_fraction=0.8, levels=(0.10, 0.20, 0.30),
                 lookback=252, name=None):
        self.ticker = ticker
        self.invest_fraction = invest_fraction
        self.levels = sorted(levels)
        self.lookback = lookback
        self.name = name or f"DCA {invest_fraction:.0%} + dip reserve"
        self._hit = set()

    def on_day(self, ctx):
        amount = ctx.flow * self.invest_fraction
        reserve = ctx.portfolio.cash - amount

        dd = drawdown(ctx, self.ticker, self.lookback)
        if dd >= 0:
            self._hit.clear()
        for level in self.levels:
            if dd <= -level and level not in self._hit:
                untriggered = len(self.levels) - len(self._hit)
                deploy = reserve / untriggered
                amount += deploy
                reserve -= deploy
                self._hit.add(level)

        return [Order(self.ticker, amount)] if amount >= MIN_TRADE else []
