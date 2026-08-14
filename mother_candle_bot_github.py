"""
Mother Candle + FRVP Strategy Alert Bot -- XAU/USD, 5-minute candles
(GitHub Actions version -- runs ONCE per invocation, triggered on a schedule
by GitHub Actions rather than looping forever like the Colab version did.)

STRATEGY: same as the Colab final version --
  Entry: mother candle -> candles 2 & 3 fully contained (wicks included) ->
    first candle from candle 4 onward whose CLOSE breaks beyond the mother
    candle's high/low triggers entry (wicks alone don't invalidate afterward).
  Bias filter (FRVP): price < VAL -> long only; price > VAH -> short only;
    between VAH/VAL -> no trade.
  No time-of-day restriction -- checks and alerts any time it's run.
  SL: exactly at the mother candle's opposite boundary (no buffer).
  Target: 1R main target (already a win), then locks 2R, closes at 3R (max).

CONFIG: reads TWELVE_DATA_API_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID from
environment variables (set as GitHub Secrets -- see setup instructions),
instead of being hardcoded in the file. This keeps your keys out of the
public/private repo's visible code.

NOTE ON DUPLICATE ALERTS: because this version runs fresh each time (no
memory between runs), it cannot remember "I already alerted on this mother
candle" the way the Colab version could. To avoid re-alerting on the same
pattern every 5 minutes while it's still valid, this version only alerts
when the LATEST candle is the exact trigger candle (checked via
is_latest_candle_entry) -- so a given pattern should only ever alert once,
on the run immediately after it triggers.
"""

import requests
import os
import sys
import pandas as pd
import numpy as np

# ---- CONFIG (from environment / GitHub Secrets) ----
TWELVE_DATA_API_KEY = os.environ.get("TWELVE_DATA_API_KEY")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

SYMBOL = "XAU/USD"
INTERVAL = "5min"
PIP = 0.1
BREAKOUT_MIN_DIST = 4.5 * PIP
MAX_WAIT = 20

SESSION_START_H, SESSION_START_M = 3, 30
SESSION_END_H, SESSION_END_M = 16, 0
HISTORY_CANDLES = 600

TWELVE_DATA_URL = "https://api.twelvedata.com/time_series"
TELEGRAM_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"


def send_telegram_alert(message):
    try:
        resp = requests.post(TELEGRAM_URL, data={"chat_id": TELEGRAM_CHAT_ID, "text": message}, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        print(f"[!] Failed to send Telegram alert: {e}")


def fetch_recent_candles(count):
    params = {
        "symbol": SYMBOL, "interval": INTERVAL, "outputsize": count,
        "apikey": TWELVE_DATA_API_KEY, "format": "JSON",
    }
    resp = requests.get(TWELVE_DATA_URL, params=params, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    if "values" not in data:
        raise RuntimeError(f"Unexpected response from Twelve Data: {data}")
    values = list(reversed(data["values"]))
    df = pd.DataFrame(values)
    df["dt"] = pd.to_datetime(df["datetime"])
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    df["volume"] = df["volume"].astype(float) if "volume" in df.columns else 1.0
    return df[["dt", "open", "high", "low", "close", "volume"]].reset_index(drop=True)


def compute_latest_value_area(df):
    last_ts = df["dt"].iloc[-1]
    candidate_end = last_ts.replace(hour=SESSION_END_H, minute=SESSION_END_M, second=0, microsecond=0)
    if candidate_end > last_ts:
        candidate_end -= pd.Timedelta(days=1)
    candidate_start = candidate_end.replace(hour=SESSION_START_H, minute=SESSION_START_M) - pd.Timedelta(days=1)

    sess = df[(df["dt"] >= candidate_start) & (df["dt"] < candidate_end)]
    if len(sess) < 10:
        return None, None

    price_min, price_max = sess["low"].min(), sess["high"].max()
    n_bins = 50
    bins = np.linspace(price_min, price_max, n_bins + 1)
    hist = np.zeros(n_bins)
    for _, row in sess.iterrows():
        lo_bin = max(0, min(np.searchsorted(bins, row["low"], side="right") - 1, n_bins - 1))
        hi_bin = max(0, min(np.searchsorted(bins, row["high"], side="right") - 1, n_bins - 1))
        span = max(1, hi_bin - lo_bin + 1)
        for b in range(lo_bin, hi_bin + 1):
            hist[b] += row["volume"] / span

    poc_bin = int(np.argmax(hist))
    total_vol = hist.sum()
    target_vol = total_vol * 0.70
    lo, hi = poc_bin, poc_bin
    cur_vol = hist[poc_bin]
    while cur_vol < target_vol and (lo > 0 or hi < n_bins - 1):
        left_val = hist[lo - 1] if lo > 0 else -1
        right_val = hist[hi + 1] if hi < n_bins - 1 else -1
        if right_val >= left_val:
            hi += 1; cur_vol += hist[hi]
        else:
            lo -= 1; cur_vol += hist[lo]
    return bins[hi + 1], bins[lo]


def find_active_setup(df):
    n = len(df)
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values

    for i in range(n - 3, -1, -1):
        m_high, m_low = h[i], l[i]
        c2h, c2l = h[i+1], l[i+1]
        c3h, c3l = h[i+2], l[i+2]

        if not (c2h <= m_high and c2l >= m_low and c3h <= m_high and c3l >= m_low):
            continue

        entry_idx = None
        direction = None
        j_end = min(i + 3 + MAX_WAIT, n)
        for j in range(i + 3, j_end):
            up_dist = c[j] - m_high
            down_dist = m_low - c[j]
            if up_dist >= BREAKOUT_MIN_DIST:
                entry_idx = j; direction = "long"; break
            elif down_dist >= BREAKOUT_MIN_DIST:
                entry_idx = j; direction = "short"; break

        return {
            "mother_time": df["dt"].iloc[i], "mother_high": m_high, "mother_low": m_low,
            "entry_idx": entry_idx, "direction": direction,
            "entry_time": df["dt"].iloc[entry_idx] if entry_idx is not None else None,
            "entry_price": c[entry_idx] if entry_idx is not None else None,
            "is_latest_candle_entry": (entry_idx == n - 1) if entry_idx is not None else False,
        }

    return None


def main():
    if not TWELVE_DATA_API_KEY or not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[!] Missing one or more required environment variables / secrets. Exiting.")
        sys.exit(1)

    try:
        df = fetch_recent_candles(HISTORY_CANDLES)
        vah, val = compute_latest_value_area(df)
        setup = find_active_setup(df)

        if setup is None:
            print("No active setup found this run.")
            return

        # Pattern alert -- only fires if the 2nd contained candle (c3) is the
        # LATEST candle, so it only fires once, right when the pattern completes
        if setup["entry_idx"] is None:
            # still waiting for breakout -- check if pattern just completed THIS run
            n = len(df)
            # setup's mother is at some index i; c3 is i+2. Only alert if i+2 == n-1
            # (re-derive i from mother_time)
            mother_idx = df.index[df["dt"] == setup["mother_time"]][0]
            if mother_idx + 2 == n - 1:
                msg = (
                    f"Mother Candle Pattern Formed (XAU/USD, 5m)\n\n"
                    f"Mother candle: {setup['mother_time']}\n"
                    f"  High: {setup['mother_high']:.2f}  Low: {setup['mother_low']:.2f}\n"
                    f"Watch for a breakout close beyond {setup['mother_high']:.2f} (long) "
                    f"or below {setup['mother_low']:.2f} (short)."
                )
                send_telegram_alert(msg)
                print(f"[PATTERN ALERT] {setup['mother_time']}")
            else:
                print("Pattern still waiting for breakout, not newly formed this run.")
            return

        if not setup["is_latest_candle_entry"]:
            print("Most recent setup already resolved in a prior run -- no new alert.")
            return

        if vah is None:
            print("No FRVP profile available yet -- skipping.")
            return

        price = setup["entry_price"]
        bias = "long" if price < val else ("short" if price > vah else None)

        if bias != setup["direction"]:
            print(f"Breakout direction ({setup['direction']}) doesn't match FRVP bias ({bias}) -- no trade alert.")
            return

        entry = price
        if setup["direction"] == "long":
            sl = setup["mother_low"]
            risk = entry - sl
            target_1r = entry + risk
        else:
            sl = setup["mother_high"]
            risk = sl - entry
            target_1r = entry - risk

        msg = (
            f"TRADE ALERT -- {setup['direction'].upper()} (XAU/USD, 5m)\n\n"
            f"Entry: {entry:.2f}\n"
            f"Stop Loss: {sl:.2f}\n"
            f"Target (1R -- main target): {target_1r:.2f}\n"
            f"VAH: {vah:.2f}  VAL: {val:.2f}\n\n"
            f"1R hit = already a win, lock stop there.\n"
            f"If it keeps running: lock at 2R next, close at 3R (max)."
        )
        send_telegram_alert(msg)
        print(f"[TRADE ALERT] {setup['direction']} at {entry:.2f}")

    except Exception as e:
        print(f"[!] Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
