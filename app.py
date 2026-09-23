"""Streamlit UI: backtester + real-time trade alerts for Nifty / Sensex options.

    pip install -r requirements.txt
    streamlit run app.py
"""
import datetime as dtm
import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import live
import options_backtester as bt

CACHE = Path(".angel_cache")
STRATEGIES = {"ema_rsi_vwap": "EMA cross + above VWAP + RSI",
              "supertrend": "Supertrend flip up + above VWAP",
              "liquidity_sweep": "Liquidity sweep + trend confirmation (underlying)"}

st.set_page_config(page_title="Options Backtester", layout="wide")
st.title("Nifty / Sensex Options: Backtest and Live Alerts")
st.caption("Buy-side signals on the option premium chart. Data: Angel One SmartAPI.")


@st.cache_data(ttl=86400, show_spinner="Loading Angel One instrument master...")
def load_master():
    CACHE.mkdir(exist_ok=True)
    m = bt.get_master(CACHE)
    keep = m["name"].isin(["NIFTY", "SENSEX"]) & m["instrumenttype"].isin(["OPTIDX", "AMXIDX"])
    return m[keep].reset_index(drop=True)


def day_chart(bars, trades, day, tf, strategy):
    b = bars[bars.index.normalize() == day]
    fig = go.Figure(go.Candlestick(x=b.index, open=b["open"], high=b["high"], low=b["low"],
                                   close=b["close"], name="Premium"))
    fig.add_trace(go.Scatter(x=b.index, y=b["vwap"], name="VWAP", line=dict(width=1.2)))
    if strategy == "ema_rsi_vwap":
        fig.add_trace(go.Scatter(x=b.index, y=b["ema_f"], name="Fast EMA", line=dict(width=1)))
        fig.add_trace(go.Scatter(x=b.index, y=b["ema_s"], name="Slow EMA", line=dict(width=1)))
    sig = b[b["sig"]]
    fig.add_trace(go.Scatter(x=sig.index + pd.Timedelta(minutes=tf), y=sig["close"],
                             mode="markers", name="Signal",
                             marker=dict(symbol="diamond", size=9)))
    t = trades[trades["date"] == day.strftime("%Y-%m-%d")] if not trades.empty else trades
    for _, x in t.iterrows():
        fig.add_trace(go.Scatter(x=[x["entry_time"]], y=[x["entry"]], mode="markers",
                                 name="Entry", marker=dict(symbol="triangle-up", size=13),
                                 showlegend=False))
        fig.add_trace(go.Scatter(x=[x["exit_time"]], y=[x["exit"]], mode="markers",
                                 name=f"Exit ({x['reason']})", showlegend=False,
                                 marker=dict(symbol="x", size=11)))
        fig.add_shape(type="line", x0=x["entry_time"], x1=x["exit_time"], y0=x["sl"], y1=x["sl"],
                      line=dict(color="#d62728", dash="dot"))
        fig.add_shape(type="line", x0=x["entry_time"], x1=x["exit_time"], y0=x["target"],
                      y1=x["target"], line=dict(color="#2ca02c", dash="dot"))
    fig.update_layout(xaxis_rangeslider_visible=False, height=540,
                      margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h"))
    return fig


# ------------------------------------------------------------- sidebar ----
with st.sidebar:
    st.header("Angel One login")
    st.caption("Kept in this browser session's memory only. Never written to disk.")
    creds = dict(
        api_key=st.text_input("API key", os.environ.get("ANGEL_API_KEY", ""), type="password"),
        client_id=st.text_input("Client ID", os.environ.get("ANGEL_CLIENT_ID", "")),
        pin=st.text_input("PIN", os.environ.get("ANGEL_PIN", ""), type="password"),
        totp_secret=st.text_input("TOTP secret", os.environ.get("ANGEL_TOTP_SECRET", ""),
                                  type="password"))
    if st.button("Connect"):
        try:
            with st.spinner("Logging in..."):
                st.session_state["api"] = bt.login(creds)
            st.session_state["connected_at"] = dtm.datetime.now().strftime("%H:%M")
        except bt.BacktestError as e:
            st.session_state.pop("api", None)
            st.error(str(e))
    if "api" in st.session_state:
        st.success(f"Connected at {st.session_state['connected_at']}")
        st.caption("If a run later fails with a token error, click Connect again.")
    else:
        st.info("Not connected")

# ------------------------------------------------------------ contract ----
try:
    master = load_master()
except bt.BacktestError as e:
    st.error(str(e))
    st.stop()

st.subheader("1. Contract")
c1, c2, c3, c4 = st.columns([1, 1.6, 1.3, 1.1])
und = c1.selectbox("Underlying", ["NIFTY", "SENSEX"])
exch = "NFO" if und == "NIFTY" else "BFO"
opts = master[(master["name"] == und) & (master["instrumenttype"] == "OPTIDX")
              & (master["exch_seg"] == exch)].copy()
opts["exp"] = pd.to_datetime(opts["expiry"], format="%d%b%Y", errors="coerce")
expiries = sorted(opts["exp"].dropna().unique())
if not expiries:
    st.error("No live option contracts found in the instrument master.")
    st.stop()
exp = c2.selectbox("Expiry (live contracts)", expiries,
                   format_func=lambda x: pd.Timestamp(x).strftime("%d %b %Y (%a)"))
strikes = sorted((opts.loc[opts["exp"] == exp, "strike"].astype(float) / 100).unique())
strike = c3.selectbox("Strike (type to search)", strikes, index=len(strikes) // 2,
                      format_func=lambda x: f"{x:g}")
tf = c4.selectbox("Timeframe (min)", [1, 3, 5, 10, 15, 30, 60], index=2)
exp_str = pd.Timestamp(exp).strftime("%Y-%m-%d")


def contract(opt):
    return bt.find_contract(master, und, exp_str, float(strike), opt)


try:
    ref_row = contract("CE")
except bt.BacktestError as e:
    st.error(str(e))
    st.stop()

# ---------------------------------------------------------- parameters ----
st.subheader("2. Parameters")
P = {}
with st.expander("Strategy parameters (used by backtest and live alerts)", expanded=True):
    g1, g2, g3, g4 = st.columns(4)
    g1.markdown("**EMA + VWAP + RSI**")
    P["ema_fast"] = g1.number_input("Fast EMA", 2, 200, 9)
    P["ema_slow"] = g1.number_input("Slow EMA", 3, 400, 21)
    P["rsi_min"] = g1.number_input("Min RSI at signal", 0.0, 100.0, 55.0)
    g2.markdown("**Supertrend**")
    P["st_n"] = g2.number_input("ATR period", 2, 100, 10)
    P["st_mult"] = g2.number_input("Multiplier", 0.5, 10.0, 3.0, step=0.5)
    g3.markdown("**Liquidity sweep**")
    P["swing_n"] = int(g3.number_input("Swing strength (bars each side)", 1, 20, 3))
    P["use_pdhl"] = g3.checkbox("Include previous-day high/low", value=True)
    P["confirm_window"] = int(g3.number_input("Confirm within N bars of sweep", 1, 30, 5,
                                              help="After a sweep, wait up to N bars for the trend to confirm."))
    P["trend_ema"] = int(g3.number_input("Trend EMA (underlying)", 2, 100, 9,
                                         help="Confirmation bar must close beyond the sweep bar AND this EMA."))
    P["liq_window"] = int(g3.number_input("Liquidity filter: sweep within N bars", 1, 30, 5,
                                          help="Only for EMA / Supertrend when the liquidity filter is on."))
    g4.markdown("**Risk**")
    P["sl_atr"] = g4.number_input("Stop loss = x * ATR", 0.1, 10.0, 1.5, step=0.1,
                                  help="Liquidity strategy uses the setup low instead (min 0.5 ATR).")
    P["rr"] = g4.number_input("Target (R multiple)", 0.5, 10.0, 2.0, step=0.5)
    P["lots"] = int(g4.number_input("Lots", 1, 1000, 1))
    lot = g4.number_input("Lot size", 1, 5000, int(ref_row["lotsize"]), key=f"lot_{ref_row['token']}",
                          help="Defaults to the CURRENT lot size. Change it if you test older dates.")
    P["no_overlap"] = g4.checkbox("One trade at a time (backtest)", value=False)

with st.expander("Advanced: timing, filters, costs"):
    a1, a2, a3, a4 = st.columns(4)
    adv = dict(
        start_time=a1.time_input("First signal after", dtm.time(9, 20), step=60),
        last_entry=a1.time_input("Last entry time", dtm.time(15, 0), step=60),
        eod=a2.time_input("Square-off time", dtm.time(15, 15), step=60),
        warmup=int(a2.number_input("Warm-up bars", 0, 500, 30)),
        min_price=a3.number_input("Min premium (Rs)", 0.0, 500.0, 5.0),
        max_gap=int(a3.number_input("Max gap to next bar (min)", 1, 60, 5)),
        lookback_days=int(a4.number_input("History before start (days)", 0, 60, 6,
                                          help="Extra days fetched so indicators are warmed up.")),
        refresh=a4.checkbox("Ignore cache (re-download)", value=False))
    b1, b2, b3, b4 = st.columns(4)
    adv["slip_pct"] = b1.number_input("Slippage % per side", 0.0, 5.0, 0.25, step=0.05)
    adv["brokerage"] = b2.number_input("Brokerage per order (Rs)", 0.0, 500.0, 20.0)
    adv["stt"] = b3.number_input("STT % on sell", 0.0, 1.0, 0.15, step=0.01, format="%.3f")
    adv["txn"] = b4.number_input("Exchange charge %", 0.0, 1.0,
                                 0.03503 if und == "NIFTY" else 0.0325, step=0.001,
                                 format="%.5f", key=f"txn_{und}")
    st.caption("Verify STT and exchange charges against current rates.")

tab_bt, tab_live = st.tabs(["Backtest", "Live trades"])

# ------------------------------------------------------------ backtest ----
with tab_bt:
    t1, t2, t3 = st.columns([1, 2, 2])
    opt = t1.radio("Type", ["CE", "PE"], horizontal=True)
    strategy = t2.selectbox("Entry signal", list(STRATEGIES), format_func=STRATEGIES.get)
    liq_filter = False
    if strategy != "liquidity_sweep":
        liq_filter = t3.checkbox("Require liquidity sweep (underlying) as confirmation", value=False,
                                 help="CE needs a bullish sweep, PE a bearish sweep, within the last N bars.")
    try:
        row = contract(opt)
        st.caption(f"{row['symbol']}  |  token {row['token']}  |  {row['exch_seg']}")
    except bt.BacktestError as e:
        st.error(str(e))
        st.stop()

    today = pd.Timestamp.now(tz="Asia/Kolkata").date()
    dr = st.date_input("Pick one date, or a start and end date",
                       (today - dtm.timedelta(days=3), today), max_value=today)
    if isinstance(dr, (tuple, list)):
        d0, d1 = dr[0], (dr[1] if len(dr) > 1 else dr[0])
    else:
        d0 = d1 = dr

    if st.button("Run backtest", type="primary"):
        if "api" not in st.session_state:
            st.error("Connect to Angel One in the sidebar first.")
        else:
            logs = []
            a = bt.default_args(underlying=und, expiry=exp_str, strike=float(strike), type=opt,
                                tf=int(tf), start=str(d0), end=str(d1), lot=int(lot),
                                strategy=strategy, liq_filter=liq_filter, **P, **adv)
            try:
                with st.spinner("Fetching candles and running the backtest (first run can take a minute)..."):
                    res = bt.run_backtest(a, st.session_state["api"], master, CACHE,
                                          log=lambda ds, m: logs.append((ds or "-", m)))
                st.session_state["out"] = dict(res=res, logs=logs, a=a)
            except bt.RateLimitError:
                st.error("Angel One is rate-limiting the candle requests (it sometimes does this even "
                         "below its documented limits). Wait a minute and run again; data already "
                         "downloaded is cached.")
            except bt.BacktestError as e:
                st.error(str(e))
            except Exception as e:  # API/network problems
                st.error(f"Unexpected error: {e}. If your session expired, click Connect again.")

    out = st.session_state.get("out")
    if out:
        res, logs, a = out["res"], out["logs"], out["a"]
        r = res["trades"]
        st.divider()
        st.subheader("Results")
        st.caption(f"{res['row']['symbol']}  |  lot {res['lot']} x {a.lots}  |  {a.tf}-min  |  "
                   f"{STRATEGIES[a.strategy]}{' + liquidity filter' if a.liq_filter else ''}  |  "
                   f"{a.start} to {a.end or a.start}")
        with st.expander("Day-by-day status (market closed, no signals, ...)", expanded=r.empty):
            st.dataframe(pd.DataFrame(logs, columns=["Date", "Status"]), hide_index=True)

        if r.empty:
            st.warning("No trades for this selection.")
        else:
            s = bt.stats(r)
            m = st.columns(6)
            m[0].metric("Trades", s["trades"])
            m[1].metric("Win rate", f"{s['win_rate']:.1f}%")
            m[2].metric("Net P&L (Rs)", f"{s['net']:,.0f}")
            m[3].metric("Profit factor", f"{s['profit_factor']:.2f}")
            m[4].metric("Avg R", f"{s['avg_R']:.2f}")
            m[5].metric("Max drawdown (Rs)", f"{s['max_dd']:,.0f}")

            show = r.copy()
            for c in ("signal_time", "entry_time", "exit_time"):
                show[c] = show[c].map(lambda x: x.strftime("%H:%M"))
            st.dataframe(show, hide_index=True)
            st.download_button("Download trades (CSV)", r.to_csv(index=False).encode(),
                               "trades.csv", "text/csv")
            st.line_chart(pd.DataFrame({"Cumulative net P&L (Rs)": r["net"].cumsum().values},
                                       index=pd.RangeIndex(1, len(r) + 1, name="Trade #")))

        bars = res["bars"]
        if bars is not None:
            days = [d for d in sorted(bars.index.normalize().unique())
                    if pd.Timestamp(a.start) <= d <= pd.Timestamp(a.end or a.start)]
            if days:
                st.subheader("Chart inspector")
                sel = st.selectbox("Day", days, index=len(days) - 1,
                                   format_func=lambda d: pd.Timestamp(d).strftime("%d %b %Y"))
                st.plotly_chart(day_chart(bars, r, pd.Timestamp(sel), a.tf, a.strategy))
                st.caption("Diamond = signal (bar close), triangle = entry, x = exit, "
                           "red/green dotted = stop loss / target.")

# ---------------------------------------------------------------- live ----
@st.cache_resource
def monitor_store():
    return {}                    # survives page refreshes / new browser tabs


MON = monitor_store()


@st.cache_data
def beep_wav():
    import io
    import math
    import struct
    import wave
    sr = 22050
    frames = b"".join(struct.pack("<h", int(11000 * math.sin(2 * math.pi * 880 * i / sr)))
                      for i in range(int(sr * 0.4)))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(frames)
    return buf.getvalue()


def live_chart(mon, key):
    a, b = mon.args[key], mon.bars.get(key)
    if b is None or b.empty:
        return None
    b = b[b.index.normalize() == mon.clock().normalize()]
    if b.empty:
        return None
    tfd = pd.Timedelta(minutes=a.tf)
    fig = go.Figure(go.Candlestick(x=b.index, open=b["open"], high=b["high"], low=b["low"],
                                   close=b["close"], name="Premium"))
    fig.add_trace(go.Scatter(x=b.index, y=b["vwap"], name="VWAP", line=dict(width=1.2)))
    if a.strategy == "ema_rsi_vwap":
        fig.add_trace(go.Scatter(x=b.index, y=b["ema_f"], name="Fast EMA", line=dict(width=1)))
        fig.add_trace(go.Scatter(x=b.index, y=b["ema_s"], name="Slow EMA", line=dict(width=1)))
    sig = b[b["sig"]]
    fig.add_trace(go.Scatter(x=sig.index + tfd, y=sig["close"], mode="markers", name="Signal",
                             marker=dict(symbol="diamond", size=9)))
    last_x = b.index[-1] + tfd
    for t in [x for x in list(mon.trades) if x["key"] == key]:
        x1 = t.get("exit_ts") or last_x
        if t.get("exit_ts"):
            fig.add_trace(go.Scatter(x=[x1], y=[t["exit"]], mode="markers", showlegend=False,
                                     name=t["status"], marker=dict(symbol="x", size=11)))
        fig.add_trace(go.Scatter(x=[t["entry_ts"]], y=[t["entry"]], mode="markers", showlegend=False,
                                 name="Entry", marker=dict(symbol="triangle-up", size=13)))
        fig.add_shape(type="line", x0=t["entry_ts"], x1=x1, y0=t["sl"], y1=t["sl"],
                      line=dict(color="#d62728", dash="dot"))
        fig.add_shape(type="line", x0=t["entry_ts"], x1=x1, y0=t["target"], y1=t["target"],
                      line=dict(color="#2ca02c", dash="dot"))
    fig.update_layout(xaxis_rangeslider_visible=False, height=480,
                      margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h"))
    return fig


ICONS = {"BUY": "\U0001F7E2", "TARGET": "\u2705", "SL": "\U0001F534"}


def live_panel():
    mon = MON.get("monitor")
    if mon is None:
        st.info("Monitor not started. Pick contracts and strategies above, then press Start.")
        return
    st.markdown(f"**{'RUNNING' if mon.running else 'STOPPED'}**  |  {mon.status}  |  "
                f"last poll: {mon.last_poll or '-'}")

    # ---- new alerts: toast + optional beep
    alerts = list(mon.alerts)
    seen = st.session_state.get("seen_alerts")
    if seen is None or seen > len(alerts):
        seen = len(alerts)                       # don't replay old alerts on refresh
    new = alerts[seen:]
    for al in new:
        st.toast(al["text"], icon=ICONS.get(al["kind"], "\u23F9\uFE0F"))
    if new:
        st.session_state["seen_alerts"] = len(alerts)
        if st.session_state.get("live_sound", True):
            try:
                st.audio(beep_wav(), format="audio/wav", autoplay=True)
            except TypeError:                    # older Streamlit without autoplay
                pass
    else:
        st.session_state["seen_alerts"] = len(alerts)

    trades = list(mon.trades)
    open_tr = [t for t in trades if t["status"] == "OPEN"]
    closed = [t for t in trades if t["status"] != "OPEN"]

    m = st.columns(4)
    m[0].metric("Signals today", len(trades))
    m[1].metric("Open trades", len(open_tr))
    m[2].metric("Closed", len(closed))
    pnl = sum(t["points"] * t["a"].lot * t["a"].lots for t in closed)
    m[3].metric("Closed P&L (Rs, before costs)", f"{pnl:,.0f}")

    # ---- open trades as live cards
    for t in open_tr:
        with st.container(border=True):
            c = st.columns([1.6, 1, 1, 1, 1, 1.2])
            c[0].markdown(f"**{t['side']}  |  {t['strategy']}**  \n{t['symbol']}  \nsince {t['time']}")
            c[1].metric("Entry", f"{t['entry']:.2f}")
            c[2].metric("Stop loss", f"{t['sl']:.2f}")
            c[3].metric("Target", f"{t['target']:.2f}")
            lp = mon.last_price(t["side"])
            c[4].metric("Last price", f"{lp:.2f}" if lp else "-")
            if lp:
                pts = lp - t["entry"]
                c[5].metric("Unrealised", f"{pts:+.2f} pts", f"Rs {pts * t['a'].lot * t['a'].lots:,.0f}")

    # ---- latest alerts, newest first
    if alerts:
        st.markdown("**Latest alerts**")
        for al in reversed(alerts[-6:]):
            msg = f"{al['time']}  {al['text']}"
            {"BUY": st.success, "TARGET": st.success, "SL": st.error}.get(al["kind"], st.warning)(msg)
    else:
        st.caption("No signals yet. New trades appear here the moment a signal confirms.")

    if trades:
        cols = ["id", "time", "symbol", "strategy", "entry", "sl", "target", "status", "exit",
                "points", "exit_time"]
        st.markdown("**All trades today**")
        st.dataframe(pd.DataFrame([{k: t[k] for k in cols} for t in trades]), hide_index=True)

    keys = list(mon.args)
    if keys:
        st.markdown("**Live chart**")
        sel = st.selectbox("Contract / strategy", keys, key="live_chart_sel",
                           format_func=lambda k: f"{k[0]}  |  {k[1]}")
        fig = live_chart(mon, sel)
        if fig is not None:
            st.plotly_chart(fig)
            st.caption("Chart shows completed bars. Diamond = signal, triangle = entry, "
                       "red/green dotted = SL / target.")
        else:
            st.caption("Waiting for today's first completed bars...")
    with st.expander("Event log"):
        ev = list(mon.events)[:80]
        st.dataframe(pd.DataFrame(ev, columns=["Time", "Level", "Message"]), hide_index=True)


if hasattr(st, "fragment"):                      # auto-refresh every 5 s (Streamlit >= 1.37)
    live_panel = st.fragment(run_every="5s")(live_panel)

with tab_live:
    st.caption("Alerts only: this never places orders. It watches the strike selected above, rebuilds the "
               "same signals on completed bars, and shows each entry / SL / target here, then the exit "
               "(SL, target or square-off).")
    l1, l2 = st.columns(2)
    types = l1.multiselect("Contracts to watch", ["CE", "PE"], default=["CE", "PE"])
    live_strats = l2.multiselect("Strategies", list(STRATEGIES), default=list(STRATEGIES),
                                 format_func=STRATEGIES.get)
    o1, o2, o3 = st.columns(3)
    live_liq = o1.checkbox("Also require a liquidity sweep for EMA / Supertrend", value=False)
    poll = int(o2.number_input("Check every (seconds)", 5, 60, 10))
    o3.checkbox("Beep on new alert", value=True, key="live_sound",
                help="Browsers may block sound until you have interacted with the page.")

    k1, k2, _ = st.columns([1, 1, 4])
    if k1.button("Start monitoring", type="primary"):
        mon = MON.get("monitor")
        if mon is not None and mon.running:
            st.warning("Already running. Stop it first to change settings.")
        elif "api" not in st.session_state:
            st.error("Connect to Angel One in the sidebar first.")
        elif not types or not live_strats:
            st.error("Pick at least one contract type and one strategy.")
        else:
            try:
                base = bt.default_args(underlying=und, expiry=exp_str, strike=float(strike), tf=int(tf),
                                       lot=int(lot), **{**P, "no_overlap": True}, **adv)
                variants = [dict(strategy=s, liq_filter=(live_liq and s != "liquidity_sweep"))
                            for s in live_strats]
                mon = live.LiveMonitor(st.session_state["api"], master, base, types, variants,
                                       str(CACHE), poll, relogin=lambda: bt.login(creds))
                mon.start()
                MON["monitor"] = mon
                st.session_state["seen_alerts"] = 0
            except bt.BacktestError as e:
                st.error(str(e))
    if k2.button("Stop"):
        mon = MON.get("monitor")
        if mon is not None:
            mon.stop()
    st.caption("Keep the computer awake during market hours. The monitor keeps running on the server "
               "even if you refresh the page. For headless use:  python live.py --help")
    live_panel()
