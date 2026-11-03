"""Interactive dashboard for the backtester.

Run from the project root:  streamlit run app.py
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from backtester import (BuyTheDip, DCAWithDipReserve, InvestImmediately, SpreadOut,
                        head_to_head, lump_sum, monthly, rolling_backtests, run_backtest)
from backtester.data import load_cash_rate, load_prices

# Fixed color per strategy so toggling one never repaints the others.
COLORS = {
    "invest": "#2a78d6",
    "spread_3": "#eb6834",
    "spread_6": "#1baf7a",
    "spread_12": "#eda100",
    "spread_24": "#e87ba4",
    "dip": "#6250d6",
    "reserve": "#e34948",
}
NEUTRAL = "#8a8984"

st.set_page_config(page_title="Strategy Backtester", layout="wide")


@st.cache_data(show_spinner=False)
def get_prices(ticker):
    return load_prices(ticker.upper(), start="1970-01-01")


@st.cache_data(show_spinner=False)
def get_cash_rate():
    return load_cash_rate()


def build(spec, ticker):
    kind, params = spec[1], dict(spec[2])
    if kind == "invest":
        return InvestImmediately(ticker)
    if kind.startswith("spread"):
        return SpreadOut(ticker, params["months"])
    if kind == "dip":
        return BuyTheDip(ticker, params["threshold"])
    return DCAWithDipReserve(ticker, params["invest_fraction"], params["levels"])


def schedule_fn(scenario, amount):
    if scenario == "windfall":
        return lambda start, end: lump_sum(amount, start)
    return lambda start, end: monthly(amount, start, end)


@st.cache_data(show_spinner=False)
def run_rolling(ticker, specs, scenario, amount, horizon, step, cash, fee, first_start):
    prices = get_prices(ticker)
    strategies = {s[0]: (lambda s=s: build(s, ticker)) for s in specs}
    return rolling_backtests(strategies, prices, schedule_fn(scenario, amount), horizon,
                             cash, fee, step=step, first_start=first_start)


def run_single(ticker, specs, scenario, amount, horizon, cash, fee, start):
    prices = get_prices(ticker)
    end = start + pd.DateOffset(years=horizon)
    deposits = schedule_fn(scenario, amount)(start, end)
    return {s[0]: run_backtest(build(s, ticker), prices, deposits, cash, fee, start, end).daily
            for s in specs}


def money(x):
    return f"${x:,.0f}"


# ---------- Sidebar: inputs ----------

with st.sidebar.form("inputs"):
    st.header("Setup")
    ticker = st.text_input("Ticker", "SPY", help="Any Yahoo Finance ticker. Prices include dividends.")
    scenario_label = st.radio("Scenario", ["One-time windfall", "Monthly investing"])
    scenario = "windfall" if scenario_label == "One-time windfall" else "monthly"
    amount = st.number_input("Amount ($)" if scenario == "windfall" else "Amount per month ($)",
                             min_value=100, value=12_000 if scenario == "windfall" else 500, step=100)
    horizon = st.slider("Years per backtest", 1, 20, 10)
    step_label = st.radio("Test a new start date every", ["Quarter (faster)", "Month"], horizontal=True)
    cash_label = st.radio("Uninvested cash earns", ["Historical T-bill rate", "Fixed rate", "Nothing"])
    fixed_rate = st.number_input("Fixed rate (%)", 0.0, 10.0, 4.0, 0.25) if cash_label == "Fixed rate" else 0.0
    fee_bps = st.number_input("Trading cost (basis points)", 0.0, 100.0, 0.0, 1.0,
                              help="1 basis point = 0.01% of each trade")

    st.header("Strategies")
    st.caption("Invest immediately is always on: it's the baseline. "
               "With one deposit it's lump sum; with monthly deposits it's DCA.")
    spread_months = st.multiselect("Spread a lump sum over (months)", [3, 6, 12, 24],
                                   default=[3, 6, 12] if scenario == "windfall" else [])
    use_dip = st.checkbox("Buy the dip", value=scenario == "monthly")
    dip_threshold = st.slider("Dip size", 5, 30, 10, format="%d%%") / 100
    use_reserve = st.checkbox("DCA + dip reserve", value=scenario == "monthly")
    invest_fraction = st.slider("Invest right away", 0, 100, 80, 5, format="%d%%") / 100
    st.form_submit_button("Run backtest", type="primary", use_container_width=True)

baseline = "Lump sum" if scenario == "windfall" else "DCA"
specs = [(baseline, "invest", ())]
specs += [(f"Spread {m} mo", f"spread_{m}", (("months", m),)) for m in spread_months]
if use_dip:
    specs.append((f"Buy the dip {dip_threshold:.0%}", "dip", (("threshold", dip_threshold),)))
if use_reserve:
    specs.append((f"DCA {invest_fraction:.0%} + dip reserve", "reserve",
                  (("invest_fraction", invest_fraction), ("levels", (0.10, 0.20, 0.30)))))
specs = tuple(specs)
color_of = {label: COLORS[kind] for label, kind, _ in specs}
labels = [s[0] for s in specs]

# ---------- Data ----------

st.title("Strategy Backtester")

try:
    prices = get_prices(ticker)
except Exception as exc:  # bad ticker or no network
    st.error(f"Couldn't load prices for {ticker.upper()}: {exc}")
    st.stop()

if cash_label == "Historical T-bill rate":
    try:
        cash = get_cash_rate()
    except Exception as exc:
        st.warning(f"Couldn't load T-bill rates ({exc}); using 0% on cash.")
        cash = None
else:
    cash = fixed_rate / 100 or None

first_start = prices.index[0] + pd.offsets.MonthBegin(1)
years_of_data = (prices.index[-1] - prices.index[0]).days / 365.25
if years_of_data < horizon + 1:
    st.error(f"{ticker.upper()} only has {years_of_data:.0f} years of data. Choose a shorter backtest.")
    st.stop()

step = "QS" if step_label.startswith("Quarter") else "MS"
with st.spinner("Running backtests across every start date..."):
    results = run_rolling(ticker.upper(), specs, scenario, amount, horizon, step, cash,
                          fee_bps / 10_000, first_start)

starts = sorted(results["start"].unique())
deposit_text = money(amount) + (" once" if scenario == "windfall" else " per month")
st.caption(f"{ticker.upper()}, {deposit_text}, {horizon}-year backtests starting every "
           f"{'quarter' if step == 'QS' else 'month'} from {starts[0]:%b %Y} to {starts[-1]:%b %Y} "
           f"({len(starts)} periods). Data through {prices.index[-1]:%b %d, %Y}.")

# ---------- Headline: win rates ----------

challengers = [l for l in labels if l != baseline]
if challengers:
    cols = st.columns(len(challengers))
    for col, name in zip(cols, challengers):
        h = head_to_head(results, name, baseline)
        col.metric(f"{name} beat {baseline}", f"{h['a_win_rate']:.0%} of the time",
                   f"{h['median_edge']:+.1%} median ending value", delta_color="normal")
else:
    st.info("Turn on another strategy in the sidebar to compare it against the baseline.")

# ---------- Summary table ----------

summary = results.groupby("strategy").agg(
    median_final=("final_value", "median"),
    worst_final=("final_value", "min"),
    best_final=("final_value", "max"),
    median_irr=("irr", "median"),
    median_max_dd=("max_drawdown", "median"),
    avg_cash=("avg_cash_share", "mean"),
).reindex(labels)
contributed = results["contributed"].median()

st.subheader("Summary")
st.caption(f"Typical total deposited per backtest: {money(contributed)}. "
           "IRR is the annual return that accounts for when money went in.")
st.dataframe(
    summary.rename(columns={
        "median_final": "Median ending value", "worst_final": "Worst", "best_final": "Best",
        "median_irr": "Median annual return (IRR)", "median_max_dd": "Median max drawdown",
        "avg_cash": "Avg. share in cash",
    }).style.format({
        "Median ending value": money, "Worst": money, "Best": money,
        "Median annual return (IRR)": "{:.2%}", "Median max drawdown": "{:.1%}",
        "Avg. share in cash": "{:.1%}",
    }),
    use_container_width=True,
    column_config={"strategy": st.column_config.TextColumn("Strategy", width="medium")},
)

# ---------- Charts ----------

def line_chart(df, y_title, y_format, zero_line=False):
    fig = go.Figure()
    for name in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[name], name=name, mode="lines",
                                 line=dict(color=color_of.get(name, NEUTRAL), width=2),
                                 hovertemplate=f"%{{y:{y_format}}}<extra>{name}</extra>"))
    if zero_line:
        fig.add_hline(y=0, line=dict(color=NEUTRAL, width=1))
    fig.update_layout(hovermode="x unified", yaxis_title=y_title, yaxis_tickformat=y_format,
                      legend=dict(orientation="h", y=1.08, x=0), margin=dict(t=40, l=10, r=10, b=10),
                      height=420)
    fig.update_xaxes(showgrid=False)
    return fig


tab_end, tab_edge, tab_one = st.tabs(["Ending value by start date", "Edge vs baseline", "Look at one period"])

with tab_end:
    st.caption(f"Each point is one {horizon}-year backtest, plotted at the month it started.")
    wide = results.pivot(index="start", columns="strategy", values="final_value")[labels]
    st.plotly_chart(line_chart(wide, "Ending value", "$,.0f"), use_container_width=True)

with tab_edge:
    if challengers:
        st.caption(f"How much more (above 0) or less (below 0) each strategy ended with than {baseline}, "
                   "by start date. Look for where the lines rise: those are periods that started right "
                   "before a crash.")
        edge = wide[challengers].div(wide[baseline], axis=0) - 1
        st.plotly_chart(line_chart(edge, f"Difference vs {baseline}", ".0%", zero_line=True),
                        use_container_width=True)
    else:
        st.info("Turn on another strategy to see its edge over the baseline.")

with tab_one:
    pick = st.select_slider("Start date", options=starts, value=starts[len(starts) // 2],
                            format_func=lambda d: f"{d:%b %Y}")
    single = run_single(ticker.upper(), specs, scenario, amount, horizon, cash, fee_bps / 10_000, pick)
    values = pd.DataFrame({name: d["value"] for name, d in single.items()})
    deposited = next(iter(single.values()))["flow"].cumsum().rename("Total deposited")

    fig = line_chart(values, "Portfolio value", "$,.0f")
    fig.add_trace(go.Scatter(x=deposited.index, y=deposited, name="Total deposited", mode="lines",
                             line=dict(color=NEUTRAL, width=2, dash="dot"),
                             hovertemplate="%{y:$,.0f}<extra>Total deposited</extra>"))
    st.plotly_chart(fig, use_container_width=True)

    end_values = values.iloc[-1].sort_values(ascending=False)
    st.caption("Ending values: " + " · ".join(f"{n} {money(v)}" for n, v in end_values.items()))

with st.expander("Download the raw results"):
    st.download_button("Download CSV", results.to_csv(index=False), "backtest_results.csv", "text/csv")
    st.caption("One row per strategy per start date, with every metric.")

st.caption("Educational tool, not investment advice. Past performance doesn't predict future returns.")
