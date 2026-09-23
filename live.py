#!/usr/bin/env python3
"""
Real-time signal monitor for a specified Nifty / Sensex strike  (ALERTS ONLY - places NO orders).

Every few seconds it pulls today's 1-min candles from Angel One, rebuilds the SAME signals as the
backtester (on completed bars only) and, when a new BUY signal appears, records an alert with
entry / stop loss / target. It then watches the trade and records the exit (SL, target or
square-off time). The Streamlit app (app.py, "Live trades" tab) displays all of this on the web page.

Headless use (prints alerts to the console):
    export ANGEL_API_KEY=... ANGEL_CLIENT_ID=... ANGEL_PIN=... ANGEL_TOTP_SECRET=...
    python live.py --underlying NIFTY --expiry 2026-09-22 --strike 25000 --types CE PE \
        --strategies ema_rsi_vwap supertrend liquidity_sweep --tf 5
"""
import argparse
import copy
import sys
import threading
import time as _time
from collections import deque
from datetime import time

import numpy as np
import pandas as pd
from pathlib import Path

import options_backtester as bt

SHORT = {"ema_rsi_vwap": "EMA-VWAP", "supertrend": "SUPERTREND", "liquidity_sweep": "LIQ-SWEEP"}


def now_ist():
    return pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None)


def variant_label(v):
    return SHORT[v["strategy"]] + ("+LIQ" if v.get("liq_filter") else "")


# -------------------------------------------------------------- monitor ----
class LiveMonitor:
    def __init__(self, api, master, base_args, types, variants, cache_dir=".angel_cache",
                 poll_sec=10, relogin=None, clock=now_ist):
        self.api, self.master = api, master
        self.types, self.variants = list(types), list(variants)
        self.cache, self.poll_sec, self.relogin, self.clock = Path(cache_dir), poll_sec, relogin, clock
        self.cache.mkdir(parents=True, exist_ok=True)
        self.base = base_args
        self.rows = {t: bt.find_contract(master, base_args.underlying, base_args.expiry,
                                         base_args.strike, t) for t in self.types}
        self.args = {}
        for t in self.types:
            for v in self.variants:
                a = copy.copy(base_args)
                a.type, a.strategy, a.liq_filter = t, v["strategy"], v.get("liq_filter", False)
                a.lot = a.lot or int(self.rows[t]["lotsize"])
                self.args[(t, variant_label(v))] = a
        self.need_index = any(a.strategy == "liquidity_sweep" or a.liq_filter
                              for a in self.args.values())
        self.irow = bt.index_row(master, base_args.underlying)
        self.events, self.trades, self.alerts, self.bars = deque(maxlen=500), [], [], {}
        self.status, self.last_poll = "not started", None
        self._stop, self._thread = threading.Event(), None
        self._hist, self._hist_idx, self._today = {}, None, {}
        self._seen, self._got, self._ltp_last = {}, {}, {}
        self._fail, self._rl, self._n = 0, 0, 0

    # ---- helpers
    def log(self, level, msg):
        self.events.appendleft((self.clock().strftime("%H:%M:%S"), level, msg))
        print(f"[{level}] {msg}", flush=True)

    def _throttle(self):
        _time.sleep(0.2)                                  # ltpData is a separate, looser limit

    def _fetch_today(self, row, now, cutoff):
        """Today's 1-min candles. At most ONE API call per instrument per completed minute."""
        key = row["token"]
        got = self._got.get(key)
        if got is not None and got[0] == cutoff:
            return got[1]
        d0 = now.normalize()
        rows = bt._candles(self.api, row["exch_seg"], key, "ONE_MINUTE",
                           d0 + pd.Timedelta(hours=9, minutes=15), now + pd.Timedelta(minutes=1),
                           strict=True)                   # backoff + limiter live in bt._candles
        df = bt.to_df(rows)
        df = df[~df.index.duplicated(keep="last")]
        self._got[key] = (cutoff, df)
        return df

    @staticmethod
    def _merge(hist, today, cutoff):
        df = pd.concat([hist, today]) if len(hist) else today
        df = df[~df.index.duplicated(keep="last")].sort_index()
        df = df[df.index + pd.Timedelta(minutes=1) <= cutoff]     # completed 1-min candles only
        return df.between_time("09:15", "15:29")

    def _ltp(self, row):
        try:
            r = self.api.ltpData(row["exch_seg"], row["symbol"], str(row["token"]))
            self._throttle()
            return float(r["data"]["ltp"])
        except Exception:
            return None

    def _notify(self, kind, text, tr):
        """Record an alert for the web page (kind: BUY / TARGET / SL / SQUARE-OFF)."""
        self.alerts.append(dict(id=len(self.alerts) + 1, time=self.clock().strftime("%H:%M:%S"),
                                kind=kind, trade_id=tr["id"], text=text))
        self.log("ALERT", text)

    def last_price(self, t):
        """Latest price of the CE / PE (last LTP poll, else newest 1-min candle)."""
        if self._ltp_last.get(t) is not None:
            return self._ltp_last[t]
        ot = self._today.get(t)
        return float(ot["close"].iloc[-1]) if ot is not None and len(ot) else None

    def _title(self, a, row, label):
        exp = pd.Timestamp(a.expiry).strftime("%d%b").upper()
        return f"{a.underlying} {a.strike:g} {a.type} {exp} [{label} {a.tf}m]"

    # ---- lifecycle
    def prepare(self):
        self.status = "loading history"
        today = self.clock().normalize()
        if today > pd.Timestamp(self.base.expiry):
            raise bt.BacktestError("This contract has already expired.")
        d0, d1 = today - pd.Timedelta(days=self.base.lookback_days), today - pd.Timedelta(days=1)
        for t, row in self.rows.items():
            self._hist[t] = (bt.cached_1m(self.api, row["exch_seg"], row["token"], d0, d1,
                                          self.cache, False) if d1 >= d0 else bt.to_df([]))
        if self.need_index:
            if self.irow is None:
                raise bt.BacktestError("Index instrument not found (needed for liquidity levels).")
            self._hist_idx = (bt.cached_1m(self.api, self.irow["exch_seg"], self.irow["token"],
                                           d0, d1, self.cache, False) if d1 >= d0 else bt.to_df([]))
        self.log("INFO", f"History loaded. Watching {len(self.args)} strategy/contract combos: "
                         + ", ".join(f"{t} {l}" for (t, l) in self.args))

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        self.status = "stopped"

    @property
    def running(self):
        return bool(self._thread and self._thread.is_alive())

    def _run(self):
        for attempt in range(6):                          # history load; retry if rate-limited
            try:
                self.prepare()
                break
            except bt.RateLimitError as e:
                self.status = "Angel One rate-limited the history download - retrying"
                self.log("WARN", f"{e} (attempt {attempt + 1}/6, waiting)")
                if self._stop.wait(10 * (attempt + 1)):
                    return
            except Exception as e:
                self.status = "error"
                self.log("ERROR", f"Startup failed: {e}")
                return
        else:
            self.status = "error"
            self.log("ERROR", "Startup failed: Angel One kept rate-limiting. Stop and start again in a minute.")
            return
        while not self._stop.is_set():
            try:
                self.step()
                self._fail = self._rl = 0
            except bt.RateLimitError as e:                # Angel One throttled us: just try again
                self._rl += 1
                self.status = "Angel One rate-limited the request - retrying"
                if self._rl in (1, 5) or self._rl % 30 == 0:
                    self.log("WARN", f"rate-limited (x{self._rl}); will retry automatically. {e}")
            except bt.AuthError as e:
                self.log("ERROR", f"session expired: {e}")
                self._relogin()
            except Exception as e:
                self._fail += 1
                self.log("ERROR", f"poll failed ({self._fail}): {e}")
                if self._fail >= 3:
                    self._relogin()
            if self.status.startswith("Market closed"):
                break
            now = self.clock()                            # wake shortly after each minute closes
            to_next_min = (now.floor("min") + pd.Timedelta(seconds=66) - now).total_seconds()
            self._stop.wait(min(self.poll_sec, to_next_min) if to_next_min > 1 else self.poll_sec)
        self.log("INFO", "Monitor stopped.")

    def _relogin(self):
        if not self.relogin:
            return
        try:
            self.api = self.relogin()
            self._fail = 0
            self.log("INFO", "Re-logged in to Angel One.")
        except Exception as e:
            self.log("ERROR", f"re-login failed: {e}")

    # ---- one poll
    def step(self):
        now = self.clock()
        self.last_poll = now.strftime("%H:%M:%S")
        if now.weekday() >= 5:
            self.status = "Market closed (weekend)"; return
        if now.time() < time(9, 15):
            self.status = "Waiting for market open (09:15)"; return
        if now.time() >= time(15, 31):
            self._handle_eod(now)
            self.status = "Market closed for today"; return

        cutoff = (now - pd.Timedelta(seconds=5)).floor("min")
        if not self._seen:                                # first poll: don't alert on history
            self._seen = {k: cutoff for k in self.args}
        self.status = f"Live - {now.strftime('%H:%M:%S')}"

        try:
            # Candles change only once a minute, so we fetch at most once per minute per
            # instrument (the rest of the polls only check LTP for open trades).
            if any(v < cutoff for v in self._seen.values()):
                idx_all = None
                if self.need_index:
                    it = self._fetch_today(self.irow, now, cutoff)
                    idx_all = self._merge(self._hist_idx, it, cutoff)
                for t, row in self.rows.items():
                    ot = self._fetch_today(row, now, cutoff)
                    self._today[t] = ot
                    con = self._merge(self._hist[t], ot, cutoff)
                    for v in self.variants:
                        label = variant_label(v)
                        a, key = self.args[(t, label)], (t, label)
                        if self._seen[key] >= cutoff or not len(con):
                            continue
                        if (a.strategy == "liquidity_sweep" or a.liq_filter) and \
                                (idx_all is None or not len(idx_all)):
                            self.log("WARN", "No index candles yet; skipping liquidity strategies.")
                            continue
                        self._scan(a, label, row, con, idx_all, self._seen[key], cutoff, now)
                        self._seen[key] = cutoff
        finally:                                          # exits must be watched even if a fetch fails
            self._check_exits(now)
            self._handle_eod(now)

    def _scan(self, a, label, row, con, idx_all, seen, cutoff, now):
        b = bt.build_bars(a, con, idx_all)
        self.bars[(a.type, label)] = b               # for the web chart
        tfd = pd.Timedelta(minutes=a.tf)
        be = b.index + tfd
        ok = (b["sig"].values & np.asarray(be > seen) & np.asarray(be <= cutoff)
              & np.array([x >= a.start_time for x in be.time]))
        for i in np.where(ok)[0]:
            if now - be[i] > pd.Timedelta(minutes=max(2 * a.tf, 3)):
                self.log("INFO", f"{label} {a.type}: stale signal from {be[i].strftime('%H:%M')} ignored")
                continue
            if now.time() > a.last_entry or now.time() >= a.eod:
                self.log("INFO", f"{label} {a.type}: signal after last-entry time, ignored")
                continue
            if a.no_overlap and any(x["status"] == "OPEN" and x["key"] == (a.type, label)
                                    for x in self.trades):
                self.log("INFO", f"{label} {a.type}: signal skipped (trade already open)")
                continue
            ltp = self._ltp(row)
            if ltp is None:
                ltp = float(con["close"].iloc[-1])
            bl = None
            if a.strategy == "liquidity_sweep":
                bl = b.loc[b["sweep_ts"].iloc[i]:b.index[i], "low"].min()
            plan = bt.plan_levels(ltp, float(b["atr"].iloc[i]), a, bl)
            if plan is None:
                continue
            entry, sl, tgt = plan
            self._n += 1
            tr = dict(id=self._n, time=now.strftime("%H:%M:%S"), entry_ts=now, key=(a.type, label),
                      symbol=row["symbol"], side=a.type, strategy=label, entry=entry, sl=sl,
                      target=tgt, status="OPEN", exit=None, points=None, exit_time=None, a=a, row=row)
            self.trades.append(tr)
            self._notify("BUY", f"BUY {self._title(a, row, label)} Entry~{entry:.2f} SL {sl:.2f} "
                                f"TGT {tgt:.2f} Lot {a.lot} @{now.strftime('%H:%M')}", tr)

    def _close(self, tr, status, price, now):
        tr.update(status=status, exit=round(price, 2), exit_time=now.strftime("%H:%M:%S"), exit_ts=now,
                  points=round(price - tr["entry"], 2))
        a = tr["a"]
        self._notify(status.split()[0], f"EXIT {self._title(a, tr['row'], tr['strategy'])} {status} "
                                        f"@{price:.2f} (entry {tr['entry']:.2f}, {tr['points']:+.2f} pts)", tr)

    def _check_exits(self, now):
        ltps = {}
        for tr in self.trades:
            if tr["status"] != "OPEN":
                continue
            a, row = tr["a"], tr["row"]
            ot = self._today.get(a.type)
            hit = None
            if ot is not None and len(ot):                 # candles that started after the entry minute
                for ts, c in ot[ot.index > tr["entry_ts"].floor("min")].iterrows():
                    if c["low"] <= tr["sl"]:
                        hit = ("SL HIT", tr["sl"]); break
                    if c["high"] >= tr["target"] + bt.TICK:
                        hit = ("TARGET HIT", tr["target"]); break
            if hit is None:
                if a.type not in ltps:
                    ltps[a.type] = self._ltp(row)
                    self._ltp_last[a.type] = ltps[a.type] if ltps[a.type] is not None else self._ltp_last.get(a.type)
                ltp = ltps[a.type]
                if ltp is not None:
                    if ltp <= tr["sl"]:
                        hit = ("SL HIT", ltp)
                    elif ltp >= tr["target"]:
                        hit = ("TARGET HIT", ltp)
            if hit:
                self._close(tr, hit[0], hit[1], now)

    def _handle_eod(self, now):
        if now.time() < self.base.eod:
            return
        for tr in self.trades:
            if tr["status"] == "OPEN":
                p = self._ltp(tr["row"]) or tr["entry"]
                self._close(tr, "SQUARE-OFF TIME", p, now)


# ------------------------------------------------------------------ CLI ----
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--underlying", required=True, choices=["NIFTY", "SENSEX"])
    p.add_argument("--expiry", required=True)
    p.add_argument("--strike", required=True, type=float)
    p.add_argument("--types", nargs="+", default=["CE", "PE"], choices=["CE", "PE"])
    p.add_argument("--strategies", nargs="+", default=["ema_rsi_vwap"],
                   choices=list(SHORT))
    p.add_argument("--liq-filter", action="store_true",
                   help="add a liquidity-sweep confirmation to EMA / Supertrend")
    p.add_argument("--tf", type=int, default=5)
    p.add_argument("--lots", type=int, default=1)
    p.add_argument("--poll", type=int, default=10, help="seconds between polls")
    p.add_argument("--cache-dir", default=".angel_cache")
    c = p.parse_args()

    try:
        api = bt.login()
        master = bt.get_master(Path(c.cache_dir))
        base = bt.default_args(underlying=c.underlying, expiry=c.expiry, strike=c.strike, tf=c.tf,
                               lots=c.lots, no_overlap=True)
        variants = [dict(strategy=s, liq_filter=(c.liq_filter and s != "liquidity_sweep"))
                    for s in c.strategies]
        mon = LiveMonitor(api, master, base, c.types, variants, c.cache_dir, c.poll,
                          relogin=bt.login)
    except bt.BacktestError as e:
        sys.exit(str(e))
    print("Monitoring... alerts print below. Ctrl+C to stop.")
    mon.start()
    try:
        while mon.running:
            _time.sleep(1)
    except KeyboardInterrupt:
        mon.stop()


if __name__ == "__main__":
    main()
