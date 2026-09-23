#!/usr/bin/env python3
"""
Nifty / Sensex options BUY-signal backtester  -  data source: Angel One SmartAPI
- Custom contract (underlying, expiry, strike, CE/PE), timeframe, and date / date range
- Signals: indicators on the option premium chart, or liquidity sweep + trend confirmation on the
  underlying index
  (CE = bullish sweep, PE = bearish sweep). All at bar CLOSE (no lookahead).
  --liq-filter adds a liquidity-sweep confirmation to the EMA / Supertrend strategies.
- Entry = open of the next 1-min bar after the signal bar closes
- SL / Target are resolved on 1-min data, not on the resampled bars
- If SL and target both fall inside one 1-min bar, SL is assumed first (conservative)
- Prints "Market closed" for weekends / holidays (checked against the index's daily candles)

SETUP
    pip install smartapi-python pyotp requests pandas numpy
    export ANGEL_API_KEY=...  ANGEL_CLIENT_ID=...  ANGEL_PIN=...  ANGEL_TOTP_SECRET=...
    (or just use the UI:  streamlit run app.py)

EXAMPLE
    python options_backtester.py --underlying NIFTY --expiry 2026-09-22 \
        --strike 25000 --type CE --tf 5 --start 2026-09-14 --end 2026-09-18

LIMITS OF THE ANGEL ONE API
- Only LIVE contracts are in the instrument master; expired contracts can't be fetched.
  Candles are cached in --cache-dir, so run it while a contract is live to keep its history.
- 1-min history: max 30 days per request (auto-chunked); ~3 requests/sec (auto-throttled).
"""
import argparse
import json
import os
import random
import sys
import threading
import time as _time
from collections import deque
from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

TICK = 0.05


class BacktestError(Exception):
    """User-facing error (bad input, login failure, missing data)."""


class RateLimitError(BacktestError):
    """Angel One answered 'Access denied because of exceeding access rate'."""


class AuthError(BacktestError):
    """Angel One session / token is no longer valid (log in again)."""


# ---- process-wide limiter for getCandleData (Angel One: 3/sec, 180/min, 5000/hour) ----
RATE_LIMIT = True
_rl_lock, _rl_calls = threading.Lock(), deque()


def _rate_wait(min_gap=0.45, per_min=150):
    """Block until another getCandleData call is allowed. Shared by all threads."""
    if not RATE_LIMIT:
        return
    with _rl_lock:
        while True:
            now = _time.time()
            while _rl_calls and now - _rl_calls[0] > 60:
                _rl_calls.popleft()
            wait = 0.0
            if _rl_calls:
                wait = max(wait, min_gap - (now - _rl_calls[-1]))
            if len(_rl_calls) >= per_min:
                wait = max(wait, 60 - (now - _rl_calls[0]) + 0.05)
            if wait <= 0:
                break
            _time.sleep(wait)
        _rl_calls.append(_time.time())


MASTER_URLS = [
    "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json",
    "https://margincalculator.angelbroking.com/OpenAPI_Files/files/OpenAPIScripMaster.json",
]
OHLCV = ["open", "high", "low", "close", "volume"]


# ------------------------------------------------------------ Angel One ----
def login(creds=None):
    """creds = dict(api_key, client_id, pin, totp_secret); falls back to env vars."""
    try:
        from SmartApi import SmartConnect
        import pyotp
    except ImportError:
        raise BacktestError("Run: pip install smartapi-python pyotp requests pandas numpy")
    env = {"api_key": "ANGEL_API_KEY", "client_id": "ANGEL_CLIENT_ID",
           "pin": "ANGEL_PIN", "totp_secret": "ANGEL_TOTP_SECRET"}
    creds = {k: ((creds or {}).get(k) or os.environ.get(v) or "").strip() for k, v in env.items()}
    missing = [env[k] for k, v in creds.items() if not v]
    if missing:
        raise BacktestError("Missing credentials: " + ", ".join(missing))
    api = SmartConnect(api_key=creds["api_key"])
    try:
        r = api.generateSession(creds["client_id"], creds["pin"],
                                pyotp.TOTP(creds["totp_secret"]).now())
    except Exception as e:
        raise BacktestError(f"Angel One login error: {e}")
    if not r or not r.get("status"):
        raise BacktestError(f"Angel One login failed: {r.get('message') if r else r}")
    return api


def get_master(cache_dir):
    p = cache_dir / "scrip_master.json"
    data = None
    if p.exists() and _time.time() - p.stat().st_mtime < 86400:
        data = json.loads(p.read_text())
    else:
        for u in MASTER_URLS:
            try:
                r = requests.get(u, timeout=180)
                r.raise_for_status()
                data = r.json()
                p.write_text(json.dumps(data))
                break
            except Exception:
                continue
        if data is None and p.exists():
            data = json.loads(p.read_text())          # stale copy is better than none
    if data is None:
        raise BacktestError("Could not download the Angel One instrument master.")
    return pd.DataFrame(data)


def find_contract(master, underlying, expiry, strike, opt):
    exch = "NFO" if underlying == "NIFTY" else "BFO"
    base = master[(master["name"] == underlying) & (master["instrumenttype"] == "OPTIDX")
                  & (master["exch_seg"] == exch)]
    exp = pd.Timestamp(expiry).strftime("%d%b%Y").upper()
    m = base[(base["expiry"].str.upper() == exp)
             & np.isclose(base["strike"].astype(float), strike * 100)   # master strike x100
             & base["symbol"].str.endswith(opt)]
    if m.empty:
        ex = pd.to_datetime(base["expiry"], format="%d%b%Y", errors="coerce").dropna().unique()
        ex = ", ".join(str(x)[:10] for x in sorted(ex)[:8])
        raise BacktestError(f"Contract not listed (only live contracts exist). Check expiry/strike.\n"
                 f"Upcoming expiries: {ex}")
    return m.iloc[0]


def _candles(api, exch, token, interval, frm, to, strict=False, retries=5):
    p = dict(exchange=exch, symboltoken=str(token), interval=interval,
             fromdate=frm.strftime("%Y-%m-%d %H:%M"), todate=to.strftime("%Y-%m-%d %H:%M"))
    r, msg, kind = None, "", "other"
    for k in range(retries):
        _rate_wait()
        try:
            r = api.getCandleData(p)
        except Exception as e:
            r = {"status": False, "message": str(e)}
        if r and r.get("status"):
            return r.get("data") or []
        msg = str((r or {}).get("message", r))
        low = msg.lower()
        if any(x in low for x in ("invalid token", "ag8001", "ab1010", "session expired")):
            kind = "auth"
            break                                        # retrying won't help; log in again
        kind = "rate" if ("access rate" in low or "couldn't parse" in low) else "other"
        if k < retries - 1:                              # exponential backoff + jitter
            _time.sleep(min(2 ** k, 8) + random.random() * 0.5)
    err = f"candle fetch failed {p['fromdate']} -> {p['todate']}: {msg}"
    if strict:
        raise {"auth": AuthError, "rate": RateLimitError}.get(kind, BacktestError)(err)
    print(f"  warning: {err}")
    return []


def to_df(rows):
    if not rows:
        return pd.DataFrame(columns=OHLCV, index=pd.DatetimeIndex([], name="dt"))
    df = pd.DataFrame(rows, columns=["dt"] + OHLCV)
    s = pd.to_datetime(df["dt"])
    df["dt"] = s.dt.tz_localize(None) if s.dt.tz is not None else s   # keep IST wall time
    df[OHLCV] = df[OHLCV].astype(float)
    return df.set_index("dt").sort_index()


def fetch_range(api, exch, token, interval, d0, d1, step_days, strict=False):
    rows, cur = [], d0
    while cur <= d1:
        nxt = min(cur + pd.Timedelta(days=step_days - 1), d1)
        rows += _candles(api, exch, token, interval, cur + pd.Timedelta(hours=9, minutes=15),
                         nxt + pd.Timedelta(hours=15, minutes=30), strict=strict)
        cur = nxt + pd.Timedelta(days=1)
    df = to_df(rows)
    return df[~df.index.duplicated(keep="last")]


def cached_1m(api, exch, token, d0, d1, cache_dir, refresh):
    """1-min candles with a local CSV cache (+ JSON of the fully-fetched date range)."""
    today = pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None).normalize()
    csv, meta = cache_dir / f"{exch}_{token}.csv", cache_dir / f"{exch}_{token}.json"
    have, cov = to_df([]), None
    if not refresh and csv.exists() and meta.exists():
        have = pd.read_csv(csv, index_col=0, parse_dates=True)
        m = json.loads(meta.read_text())
        cov = (pd.Timestamp(m["from"]), pd.Timestamp(m["to"]))
    pick = lambda df: df[(df.index >= d0) & (df.index < d1 + pd.Timedelta(days=1))]
    if cov and d0 >= cov[0] and d1 <= cov[1]:
        return pick(have)

    new = fetch_range(api, exch, token, "ONE_MINUTE", d0, d1, 25, strict=True)
    done_to = min(d1, today - pd.Timedelta(days=1))    # today's data is still incomplete
    if new.empty or done_to < d0:
        return new
    if cov and d0 <= cov[1] + pd.Timedelta(days=1) and done_to >= cov[0] - pd.Timedelta(days=1):
        df = pd.concat([have, new])
        df = df[~df.index.duplicated(keep="last")].sort_index()
        c0, c1 = min(cov[0], d0), max(cov[1], done_to)
    else:
        df, c0, c1 = new, d0, done_to
    df.to_csv(csv)
    meta.write_text(json.dumps({"from": str(c0.date()), "to": str(c1.date())}))
    return pick(df)


def index_row(master, underlying):
    seg = "NSE" if underlying == "NIFTY" else "BSE"
    r = master[(master["name"] == underlying) & (master["instrumenttype"] == "AMXIDX")
               & (master["exch_seg"] == seg)]
    return None if r.empty else r.iloc[0]


def trading_days(api, master, underlying, d0, d1):
    """Days the market was open, from the index's daily candles. None if unavailable."""
    r = index_row(master, underlying)
    if r is None:
        return None
    df = fetch_range(api, r["exch_seg"], r["token"], "ONE_DAY", d0, d1, 400)
    return set(df.index.normalize()) if not df.empty else None


def resample(df, tf):
    """Session-anchored bars (each day starts 09:15). Bar timestamp = bar start."""
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    out = []
    for _, g in df.groupby(df.index.normalize()):
        r = g.resample(f"{tf}min", offset="9h15min").agg(agg).dropna(subset=["open"])
        out.append(r)
    return pd.concat(out)


# ----------------------------------------------------------- indicators ----
def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rma(s, n):
    return s.ewm(alpha=1 / n, adjust=False).mean()


def rsi(c, n=14):
    d = c.diff()
    up, dn = rma(d.clip(lower=0), n), rma((-d).clip(lower=0), n)
    return 100 - 100 / (1 + up / dn)


def atr(df, n=14):
    pc = df["close"].shift()
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(),
                    (df["low"] - pc).abs()], axis=1).max(axis=1)
    return rma(tr, n)


def vwap(df):
    day = df.index.normalize()
    tp = (df["high"] + df["low"] + df["close"]) / 3
    return (tp * df["volume"]).groupby(day).cumsum() / df["volume"].groupby(day).cumsum()


def supertrend(df, n=10, m=3.0):
    N = len(df)
    a = atr(df, n).values
    hl2 = ((df["high"] + df["low"]) / 2).values
    c = df["close"].values
    ub, lb = hl2 + m * a, hl2 - m * a
    fu, fl, trend = ub.copy(), lb.copy(), np.zeros(N)
    if N == 0:
        return trend
    trend[0] = 1
    for i in range(1, N):
        fu[i] = ub[i] if (ub[i] < fu[i - 1] or c[i - 1] > fu[i - 1]) else fu[i - 1]
        fl[i] = lb[i] if (lb[i] > fl[i - 1] or c[i - 1] < fl[i - 1]) else fl[i - 1]
        if trend[i - 1] == 1:
            trend[i] = -1 if c[i] < fl[i] else 1
        else:
            trend[i] = 1 if c[i] > fu[i] else -1
    return trend


def liquidity_sweeps(ib, n=3, use_pdhl=True, max_levels=12):
    """Liquidity sweeps on the UNDERLYING index bars. Returns (bull, bear) bool Series.
    Liquidity = confirmed swing lows/highs (n bars each side, same day) + previous-day low/high.
    Bullish sweep : bar trades below a resting low, then closes back above it on a green bar.
    Bearish sweep : bar trades above a resting high, then closes back below it on a red bar.
    A swing is only 'known' n bars after it forms, and a swept level is consumed (no lookahead)."""
    N = len(ib)
    o, h, l, c = (ib[k].values for k in ("open", "high", "low", "close"))
    day = ib.index.normalize()
    dh, dl = ib["high"].groupby(day).max(), ib["low"].groupby(day).min()
    prev = dict(zip(dh.index[1:], zip(dh.values[:-1], dl.values[:-1])))
    bull, bear = np.zeros(N, bool), np.zeros(N, bool)
    lows, highs = [], []
    for j in range(N):
        if use_pdhl and (j == 0 or day[j] != day[j - 1]) and day[j] in prev:
            highs.append(prev[day[j]][0]); lows.append(prev[day[j]][1])
        for L in lows[:]:                                   # sweeps of resting lows
            if l[j] < L:
                lows.remove(L)
                if c[j] > L and c[j] > o[j]:
                    bull[j] = True
        for H in highs[:]:                                  # sweeps of resting highs
            if h[j] > H:
                highs.remove(H)
                if c[j] < H and c[j] < o[j]:
                    bear[j] = True
        p = j - n                                           # pivot confirmed at bar j
        if p >= n and day[p - n] == day[j]:
            if l[p] < l[p - n:p].min() and l[p] <= l[p + 1:j + 1].min():
                lows.append(l[p])
            if h[p] > h[p - n:p].max() and h[p] >= h[p + 1:j + 1].max():
                highs.append(h[p])
        lows, highs = lows[-max_levels:], highs[-max_levels:]
    return pd.Series(bull, index=ib.index), pd.Series(bear, index=ib.index)


def liquidity_confirmed(ib, bull, bear, window=5, ema_n=9):
    """Sweep -> trend confirmation on the UNDERLYING. Returns (conf_bull, conf_bear):
    Series holding the SWEEP bar's timestamp on the bar where the entry is confirmed (NaT elsewhere).
    Bullish: after a bullish sweep bar S, the first bar (within `window` bars, same day) that
      CLOSES above S's high AND above the fast EMA (trend turned up) confirms the trade.
      If a bar closes below S's low first, the setup is void.
    Bearish is the mirror image. Entry is at the very next open after the confirming bar closes."""
    N = len(ib)
    c, h, l = ib["close"].values, ib["high"].values, ib["low"].values
    e = ema(ib["close"], ema_n).values
    day = ib.index.normalize().values
    cb = np.full(N, np.datetime64("NaT"), dtype="datetime64[ns]")
    cs = cb.copy()
    bu, be = bull.values, bear.values
    pb = ps = None                                      # pending (sweep idx, high, low)
    for j in range(N):
        if pb is not None:
            i, hi, lo = pb
            if day[j] != day[i] or j - i > window or c[j] < lo:
                pb = None
            elif c[j] > hi and c[j] > e[j]:
                cb[j] = ib.index[i].to_datetime64(); pb = None
        if ps is not None:
            i, hi, lo = ps
            if day[j] != day[i] or j - i > window or c[j] > hi:
                ps = None
            elif c[j] < lo and c[j] < e[j]:
                cs[j] = ib.index[i].to_datetime64(); ps = None
        if bu[j]:
            pb = (j, h[j], l[j])
        if be[j]:
            ps = (j, h[j], l[j])
    return pd.Series(cb, index=ib.index), pd.Series(cs, index=ib.index)


def recent_sweep(s, k):
    """True if a sweep happened on this bar or the previous k-1 bars of the SAME day."""
    d = s.index.normalize().values
    out = s.copy()
    for i in range(1, k):
        same = np.r_[np.zeros(i, bool), d[i:] == d[:-i]]
        out |= s.shift(i, fill_value=False) & same
    return out


def add_signals(b, a, sweeps=None):
    b = b.copy()
    b["ema_f"], b["ema_s"] = ema(b["close"], a.ema_fast), ema(b["close"], a.ema_slow)
    b["rsi"], b["atr"], b["vwap"] = rsi(b["close"]), atr(b, a.atr_n), vwap(b)
    if a.strategy == "ema_rsi_vwap":
        cross = (b["ema_f"] > b["ema_s"]) & (b["ema_f"].shift() <= b["ema_s"].shift())
        sig = cross & (b["close"] > b["vwap"]) & (b["rsi"] >= a.rsi_min)
    elif a.strategy == "liquidity_sweep":   # signal comes from the underlying's sweeps
        conf = sweeps[2] if a.type == "CE" else sweeps[3]   # CE: bullish, PE: bearish
        b["sweep_ts"] = conf.reindex(b.index)
        sig = b["sweep_ts"].notna()
    else:  # supertrend flip to up, above VWAP
        st = pd.Series(supertrend(b, a.st_n, a.st_mult), index=b.index)
        sig = (st == 1) & (st.shift() == -1) & (b["close"] > b["vwap"])
    if a.liq_filter and a.strategy != "liquidity_sweep":   # confluence with liquidity
        s = sweeps[0] if a.type == "CE" else sweeps[1]
        sig &= recent_sweep(s, a.liq_window).reindex(b.index, fill_value=False).astype(bool)
    sig &= (np.arange(len(b)) >= a.warmup)          # indicator warm-up
    sig &= (b["volume"] > 0) & (b["close"] >= a.min_price)  # liquidity filter
    b["sig"] = sig.fillna(False)
    return b


# ------------------------------------------------------------ execution ----
def build_bars(a, con, idx=None):
    """Option 1-min candles (+ underlying 1-min candles if needed) -> signal bars."""
    sweeps = None
    if a.strategy == "liquidity_sweep" or a.liq_filter:
        if idx is None or not len(idx):
            raise BacktestError("No index candles; cannot compute liquidity levels.")
        ib = resample(idx, a.tf)
        bull, bear = liquidity_sweeps(ib, a.swing_n, a.use_pdhl)
        cb, cs = liquidity_confirmed(ib, bull, bear, a.confirm_window, a.trend_ema)
        sweeps = (bull, bear, cb, cs)
    return add_signals(resample(con, a.tf), a, sweeps)


def rt(x, up):
    """Round to tick: up=True -> ceil, else floor."""
    v = np.ceil(x / TICK - 1e-9) if up else np.floor(x / TICK + 1e-9)
    return round(float(v) * TICK, 2)


def slip(p, a):
    return max(TICK, p * a.slip_pct / 100)


def costs(entry, exitp, qty, a):
    buy, sell = entry * qty, exitp * qty
    brok = 2 * a.brokerage
    txn = (buy + sell) * a.txn / 100
    stt = sell * a.stt / 100
    sebi = (buy + sell) * 10 / 1e7
    stamp = buy * 0.003 / 100
    gst = 0.18 * (brok + txn + sebi)
    return brok + txn + stt + sebi + stamp + gst


def plan_levels(entry_raw, atr_val, a, bar_low=None):
    """Entry (with slippage), stop loss and target. Returns None if the setup is unusable."""
    entry = rt(entry_raw + slip(entry_raw, a), True)
    risk = a.sl_atr * atr_val
    if bar_low is not None and not pd.isna(bar_low):   # sweep trades: SL under the setup's low
        risk = max(entry - bar_low, 0.5 * atr_val)
    sl = max(rt(entry - risk, False), TICK)
    if entry - sl < TICK:
        return None
    return entry, sl, rt(entry + a.rr * (entry - sl), True)


def run_trade(base, pos, atr_val, a, bar_low=None):
    """base = contract's base-resolution bars for the day; pos = entry bar index."""
    ts = base.index
    o, h, l, c = (base[k].values for k in ("open", "high", "low", "close"))
    plan = plan_levels(o[pos], atr_val, a, bar_low)
    if plan is None:
        return None
    entry, sl, tgt = plan

    ex, why, j = None, None, len(ts) - 1
    for j in range(pos, len(ts)):
        if ts[j].time() >= a.eod:
            ex, why = o[j], "EOD"; break
        if j > pos:
            if o[j] <= sl:
                ex, why = o[j], "SL-gap"; break
            if o[j] >= tgt:
                ex, why = o[j], "TARGET-gap"; break
        if l[j] <= sl:                       # SL checked first (conservative)
            ex, why = sl, "SL"; break
        if h[j] >= tgt + TICK:               # must trade through target to fill
            ex, why = tgt, "TARGET"; break
    if ex is None:
        ex, why = c[j], "EOD(last bar)"
    limit_fill = why.startswith("TARGET")
    xf = ex if limit_fill else max(rt(ex - slip(ex, a), False), 0.0)

    qty = a.lot * a.lots
    gross = (xf - entry) * qty
    cost = costs(entry, xf, qty, a)
    return dict(entry_time=ts[pos], entry=entry, sl=sl, target=tgt,
                exit_time=ts[j], exit=round(xf, 2), reason=why,
                points=round(xf - entry, 2), gross=round(gross, 2),
                costs=round(cost, 2), net=round(gross - cost, 2),
                R=round((xf - entry) / (entry - sl), 2))


# ----------------------------------------------------------------- main ----
def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--underlying", required=True, choices=["NIFTY", "SENSEX"])
    p.add_argument("--expiry", required=True, help="YYYY-MM-DD (must be a live contract)")
    p.add_argument("--strike", required=True, type=float)
    p.add_argument("--type", required=True, choices=["CE", "PE"])
    p.add_argument("--tf", type=int, default=5, help="timeframe in minutes")
    p.add_argument("--start", required=True)
    p.add_argument("--end", default=None)
    p.add_argument("--lot", type=int, default=None,
                   help="lot size (default: CURRENT lot from Angel One; override for old dates)")
    p.add_argument("--lots", type=int, default=1)
    p.add_argument("--lookback-days", dest="lookback_days", type=int, default=6,
                   help="extra calendar days fetched before --start for indicator warm-up")
    p.add_argument("--cache-dir", dest="cache_dir", default=".angel_cache")
    p.add_argument("--refresh", action="store_true", help="ignore cached candles")
    p.add_argument("--strategy", default="ema_rsi_vwap", choices=["ema_rsi_vwap", "supertrend", "liquidity_sweep"])
    p.add_argument("--ema-fast", dest="ema_fast", type=int, default=9)
    p.add_argument("--ema-slow", dest="ema_slow", type=int, default=21)
    p.add_argument("--rsi-min", dest="rsi_min", type=float, default=55)
    p.add_argument("--atr-n", dest="atr_n", type=int, default=14)
    p.add_argument("--st-n", dest="st_n", type=int, default=10)
    p.add_argument("--st-mult", dest="st_mult", type=float, default=3.0)
    p.add_argument("--swing-n", dest="swing_n", type=int, default=3,
                   help="liquidity_sweep: bars each side to confirm a swing high/low")
    p.add_argument("--no-pdhl", dest="use_pdhl", action="store_false",
                   help="liquidity_sweep: ignore previous-day high/low levels")
    p.add_argument("--confirm-window", dest="confirm_window", type=int, default=5,
                   help="liquidity_sweep: bars allowed between the sweep and the trend confirmation")
    p.add_argument("--trend-ema", dest="trend_ema", type=int, default=9,
                   help="liquidity_sweep: confirmation bar must close beyond this EMA of the underlying")
    p.add_argument("--liq-filter", dest="liq_filter", action="store_true",
                   help="ema_rsi_vwap / supertrend: also require a liquidity sweep on the underlying")
    p.add_argument("--liq-window", dest="liq_window", type=int, default=5,
                   help="sweep must be within the last N bars (incl. signal bar)")
    p.add_argument("--sl-atr", dest="sl_atr", type=float, default=1.5, help="SL = k * ATR")
    p.add_argument("--rr", type=float, default=2.0, help="target = rr * risk")
    p.add_argument("--warmup", type=int, default=30, help="min bars before signals")
    p.add_argument("--min-price", dest="min_price", type=float, default=5.0)
    p.add_argument("--start-time", dest="start_time", default="09:20")
    p.add_argument("--last-entry", dest="last_entry", default="15:00")
    p.add_argument("--eod", default="15:15", help="square-off time")
    p.add_argument("--max-gap", dest="max_gap", type=int, default=5,
                   help="skip if next bar is > N min after signal (illiquid)")
    p.add_argument("--no-overlap", dest="no_overlap", action="store_true")
    # cost model - VERIFY these against current exchange / broker rates
    p.add_argument("--slip-pct", dest="slip_pct", type=float, default=0.25)
    p.add_argument("--brokerage", type=float, default=20.0, help="per order")
    p.add_argument("--stt", type=float, default=0.15, help="%% on sell premium")
    p.add_argument("--txn", type=float, default=None, help="exchange charge %%")
    p.add_argument("--out", default=None, help="save trades to CSV")
    return p


def finalize(a):
    for k in ("start_time", "last_entry", "eod"):
        v = getattr(a, k)
        if isinstance(v, str):
            setattr(a, k, time.fromisoformat(v))
    if a.txn is None:
        a.txn = {"NIFTY": 0.03503, "SENSEX": 0.0325}[a.underlying]
    return a


def parse(argv=None):
    return finalize(build_parser().parse_args(argv))


def default_args(**overrides):
    """Namespace with all defaults; used by the UI."""
    a = build_parser().parse_args(["--underlying", "NIFTY", "--expiry", "2000-01-01",
                                   "--strike", "0", "--type", "CE", "--start", "2000-01-01"])
    a.__dict__.update(overrides)
    return finalize(a)


def stats(r):
    w, ls = r[r.net > 0].net.sum(), -r[r.net <= 0].net.sum()
    eq = r.net.cumsum()
    return dict(trades=len(r), wins=int((r.net > 0).sum()), win_rate=(r.net > 0).mean() * 100,
                net=r.net.sum(), costs=r.costs.sum(), avg_R=r.R.mean(),
                profit_factor=(w / ls if ls else float("inf")),
                max_dd=(eq - eq.cummax()).min())


def summary(r):
    s = stats(r)
    print("\n===== SUMMARY =====")
    print(f"Trades        : {s['trades']}   Wins: {s['wins']} ({s['win_rate']:.1f}%)")
    print(f"Net P&L (Rs)  : {s['net']:,.2f}   Costs: {s['costs']:,.2f}")
    print(f"Avg R         : {s['avg_R']:.2f}   Profit factor: {s['profit_factor']:.2f}")
    print(f"Max drawdown  : {s['max_dd']:,.2f}")


def run_backtest(a, api, master, cache_dir, log=None):
    """Core engine. log(date_str_or_None, message). Returns dict(row, trades, bars, con)."""
    log = log or (lambda ds, msg: print(f"{ds}: {msg}" if ds else msg))
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    row = find_contract(master, a.underlying, a.expiry, a.strike, a.type)
    a.lot = a.lot or int(row["lotsize"])

    expiry = pd.Timestamp(a.expiry).normalize()
    start = pd.Timestamp(a.start).normalize()
    end = pd.Timestamp(a.end or a.start).normalize()
    if end < start:
        raise BacktestError("End date is before start date.")
    today = pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None).normalize()

    days = trading_days(api, master, a.underlying, start, min(end, today))
    con = cached_1m(api, row["exch_seg"], row["token"],
                    start - pd.Timedelta(days=a.lookback_days), min(end, today),
                    cache, a.refresh)
    con = con[~con.index.duplicated(keep="last")].between_time("09:15", "15:29")
    if days is None:
        log(None, "warning: index calendar unavailable; using contract data to detect closed days")
        days = set(con.index.normalize())
    tfd = pd.Timedelta(minutes=a.tf)
    idx = None
    if a.strategy == "liquidity_sweep" or a.liq_filter:   # levels are read off the underlying
        ir = index_row(master, a.underlying)
        if ir is None:
            raise BacktestError("Index instrument not found in master (needed for liquidity levels).")
        idx = cached_1m(api, ir["exch_seg"], ir["token"],
                        start - pd.Timedelta(days=a.lookback_days), min(end, today), cache, a.refresh)
        idx = idx[~idx.index.duplicated(keep="last")].between_time("09:15", "15:29")
        idx = idx[idx.index.normalize() <= end]
    bars = (build_bars(a, con[con.index.normalize() <= end], idx) if len(con) else None)

    trades = []
    for d in pd.date_range(start, end):
        ds = d.strftime("%Y-%m-%d")
        if d > today:
            log(ds, "date is in the future"); continue
        if d not in days:
            log(ds, "Market closed"); continue
        if d > expiry:
            log(ds, "contract already expired"); continue
        base = con[con.index.normalize() == d]
        if base.empty:
            log(ds, "no data for this contract"); continue

        db = bars[bars.index.normalize() == d]
        bar_end = db.index + tfd
        ok = db["sig"].values & np.array([t >= a.start_time for t in bar_end.time])
        if not ok.any():
            log(ds, "no buy signals"); continue

        busy_until, n = None, 0
        for i in np.where(ok)[0]:
            be = bar_end[i]
            pos = base.index.searchsorted(be)
            if pos >= len(base):
                continue
            et = base.index[pos]
            if et - be > pd.Timedelta(minutes=a.max_gap) or et.time() > a.last_entry:
                continue
            if a.no_overlap and busy_until is not None and et < busy_until:
                continue
            bl = None
            if a.strategy == "liquidity_sweep":      # SL under the lowest low: sweep -> confirmation
                bl = bars.loc[db["sweep_ts"].iloc[i]:db.index[i], "low"].min()
            t = run_trade(base, pos, db["atr"].iloc[i], a, bl)
            if t is None:
                continue
            busy_until = t["exit_time"]
            n += 1
            trades.append(dict(date=ds, signal_time=be, side=f"{a.strike:g}{a.type}", **t))
        log(ds, f"{n} trade(s)" if n else "signals found but none tradable (gap / cutoff / overlap)")
    return dict(row=row, trades=pd.DataFrame(trades), bars=bars, con=con, lot=a.lot)


def main():
    a = parse()
    try:
        api = login()
        cache = Path(a.cache_dir)
        cache.mkdir(parents=True, exist_ok=True)
        res = run_backtest(a, api, get_master(cache), cache)
    except BacktestError as e:
        sys.exit(str(e))
    r = res["trades"]
    print(f"\nContract: {res['row']['symbol']} | lot {res['lot']}")
    if r.empty:
        print("No trades.")
        return
    show = r.copy()
    for c in ("signal_time", "entry_time", "exit_time"):
        show[c] = show[c].map(lambda x: x.strftime("%H:%M"))
    print("\n" + show.to_string(index=False))
    summary(r)
    if a.out:
        r.to_csv(a.out, index=False)
        print(f"\nSaved: {a.out}")


if __name__ == "__main__":
    main()
