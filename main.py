import os
import time
import threading
import requests
import numpy as np
import pandas as pd
from collections import defaultdict, deque
from datetime import datetime, timezone
from flask import Flask, jsonify

# ============================================================
# CRYPTO OPTIONS TELEGRAM SIGNAL BOT - NO BINANCE
# ============================================================
# Providers:
#   Primary underlying market data: OKX public API
#   Primary options data: OKX public API
#   Options fallback: Deribit public API (BTC/ETH options)
#
# No API keys are required for market-data-only operation.
# No orders are placed by this program.
# ============================================================

BOT_NAME = os.getenv("BOT_NAME", "Crypto Options High Probability Bot")
TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TG_CHAT = os.getenv("TELEGRAM_CHAT_ID", "").strip()

OKX_BASE = os.getenv("OKX_BASE", "https://www.okx.com").rstrip("/")
DERIBIT_BASE = os.getenv("DERIBIT_BASE", "https://www.deribit.com/api/v2").rstrip("/")

INTERVAL = "5m"
HISTORY = int(os.getenv("HISTORY", "180"))
SCAN_SLEEP = float(os.getenv("SCAN_SLEEP", "8"))
OPTION_REFRESH_MIN = int(os.getenv("OPTION_REFRESH_MIN", "30"))

EMA_FAST = 9
EMA_SLOW = 20
RSI_LEN = 14
ADX_LEN = 14
ATR_LEN = 14
VOL_LEN = 20

MIN_ADX = 20
SIDEWAYS_ADX = 17
MIN_SCORE = 7
EARLY_RSI_LONG = 52
EARLY_RSI_SHORT = 48
CONF_RSI_LONG = 55
CONF_RSI_SHORT = 45

MIN_DELTA = 0.35
MAX_DELTA = 0.65
MAX_STRIKE_DISTANCE = 0.035
MAX_SPREAD = 0.025
MIN_OPTION_VOLUME = 1.0
MIN_OPTION_OI = 1.0
MIN_DTE_HOURS = float(os.getenv("MIN_DTE_HOURS", "12"))

PREMIUM_SL = 0.25
PREMIUM_T1 = 0.40
PREMIUM_T2 = 0.70

EARLY_COOLDOWN_MIN = 20
CONF_COOLDOWN_MIN = 10

REQUESTED = [
    ("Bitcoin", "BTC"), ("Ethereum", "ETH"), ("Solana", "SOL"),
    ("Ripple", "XRP"), ("Dogecoin", "DOGE"), ("Cardano", "ADA"),
    ("Avalanche", "AVAX"), ("Tron", "TRX"), ("Binance Coin", "BNB"),
    ("Near Protocol", "NEAR"), ("Aave", "AAVE"), ("Lighter", "LIT"),
    ("Ethena", "ENA"), ("Zcash", "ZEC"), ("Akedo", "AKE"),
    ("Esports Token", "ESPORTS"), ("Uniswap", "UNI"),
    ("Chainlink", "LINK"), ("Litecoin", "LTC"), ("Polygon", "MATIC"),
]

ALIASES = {
    "MATIC": ["MATIC", "POL"],
}

app = Flask(__name__)
lock = threading.RLock()

asset_name = {}
spot_symbols = {}          # requested asset -> OKX spot instId
bars = {}                  # OKX spot instId -> deque
last_bar = {}
seen = set()
last_alert = {}
stats = defaultdict(int)

# asset -> list of normalized option contract metadata
contracts = {}
option_provider = {}       # asset -> OKX / DERIBIT
last_option_discovery = 0
provider_status = {"okx": "unknown", "deribit": "unknown"}
last_provider_errors = {}

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 crypto-options-telegram-bot/2.0",
    "Accept": "application/json",
})


def telegram(text):
    if not TG_TOKEN or not TG_CHAT:
        print("\nTELEGRAM NOT CONFIGURED\n" + text)
        return False
    try:
        r = session.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            json={"chat_id": TG_CHAT, "text": text, "parse_mode": "HTML",
                  "disable_web_page_preview": True},
            timeout=12,
        )
        if not r.ok:
            print("Telegram error:", r.status_code, r.text[:300])
        return r.ok
    except Exception as e:
        print("Telegram exception:", e)
        return False


def get_json(url, params=None, timeout=15):
    r = session.get(url, params=params, timeout=timeout)
    if not r.ok:
        raise requests.HTTPError(f"HTTP {r.status_code}: {r.text[:250]}", response=r)
    data = r.json()
    if isinstance(data, dict) and str(data.get("code", "0")) not in ("0", "200"):
        raise RuntimeError(f"API error {data.get('code')}: {data.get('msg')}")
    return data


def okx(path, params=None, timeout=15):
    return get_json(OKX_BASE + path, params, timeout)


def deribit(method, params=None, timeout=15):
    data = get_json(f"{DERIBIT_BASE}/{method}", params, timeout)
    if "error" in data:
        raise RuntimeError(str(data["error"]))
    return data.get("result")


def safe_float(v, default=np.nan):
    try:
        x = float(v)
        return x if np.isfinite(x) else default
    except Exception:
        return default


def fmt(x):
    x = safe_float(x, np.nan)
    if not np.isfinite(x):
        return "-"
    if abs(x) >= 1000:
        return f"{x:,.2f}"
    if abs(x) >= 1:
        return f"{x:,.4f}"
    return f"{x:,.8f}"


def rsi(series, n=14):
    d = series.diff()
    gain = d.clip(lower=0)
    loss = -d.clip(upper=0)
    ag = gain.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    al = loss.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    rs = ag / al.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    out = out.where(al != 0, 100)
    out = out.where(~((ag == 0) & (al == 0)), 50)
    return out


def atr(df, n=14):
    pc = df.close.shift(1)
    tr = pd.concat([
        df.high - df.low,
        (df.high - pc).abs(),
        (df.low - pc).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False, min_periods=n).mean()


def adx(df, n=14):
    up = df.high.diff()
    dn = -df.low.diff()
    plus = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    minus = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    pc = df.close.shift(1)
    tr = pd.concat([
        df.high - df.low,
        (df.high - pc).abs(),
        (df.low - pc).abs(),
    ], axis=1).max(axis=1)
    atrx = tr.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    dip = 100 * plus.ewm(alpha=1/n, adjust=False, min_periods=n).mean() / atrx
    dim = 100 * minus.ewm(alpha=1/n, adjust=False, min_periods=n).mean() / atrx
    dx = 100 * (dip - dim).abs() / (dip + dim).replace(0, np.nan)
    ax = dx.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    return ax, dip, dim


def build_indicators(rows):
    if len(rows) < 65:
        return None
    d = pd.DataFrame(rows).copy()
    d["ema9"] = d.close.ewm(span=EMA_FAST, adjust=False).mean()
    d["ema20"] = d.close.ewm(span=EMA_SLOW, adjust=False).mean()
    d["rsi"] = rsi(d.close, RSI_LEN)
    d["atr"] = atr(d, ATR_LEN)
    d["adx"], d["di_plus"], d["di_minus"] = adx(d, ADX_LEN)
    tp = (d.high + d.low + d.close) / 3
    dt = pd.to_datetime(d.open_time, unit="ms", utc=True)
    day = dt.dt.date
    d["vwap"] = ((tp * d.volume).groupby(day).cumsum() /
                  d.volume.groupby(day).cumsum())
    d["vol_avg"] = d.volume.rolling(VOL_LEN).mean().shift(1)
    d["vol_ratio"] = d.volume / d.vol_avg.replace(0, np.nan)
    rng = (d.high - d.low).replace(0, np.nan)
    d["body"] = (d.close - d.open).abs() / rng
    d["ema_gap"] = (d.ema9 - d.ema20).abs() / d.atr.replace(0, np.nan)
    return d


def score(c, side, confirmed=True):
    s = 0
    if side == "LONG":
        s += 2 if c.ema9 > c.ema20 else 0
        s += 2 if c.close > c.vwap else 0
        s += 1 if c.rsi > (CONF_RSI_LONG if confirmed else EARLY_RSI_LONG) else 0
        s += 1 if c.adx >= MIN_ADX else 0
        s += 1 if c.di_plus > c.di_minus else 0
    else:
        s += 2 if c.ema9 < c.ema20 else 0
        s += 2 if c.close < c.vwap else 0
        s += 1 if c.rsi < (CONF_RSI_SHORT if confirmed else EARLY_RSI_SHORT) else 0
        s += 1 if c.adx >= MIN_ADX else 0
        s += 1 if c.di_minus > c.di_plus else 0
    s += 1 if c.vol_ratio >= 1.15 else 0
    s += 1 if c.body >= 0.55 else 0
    s += 1 if c.ema_gap >= 0.10 else 0
    return min(int(s), 10)


def grade(v):
    if v >= 9:
        return "🔥🔥 VERY HIGH"
    if v >= 8:
        return "🔥 HIGH"
    return "🟢 VALID"


# ============================================================
# OKX UNDERLYING DATA
# ============================================================

def discover_spot_symbols():
    global spot_symbols, asset_name
    try:
        data = okx("/api/v5/public/instruments", {"instType": "SPOT"})
        rows = data.get("data", [])
        by_base = {}
        for x in rows:
            if str(x.get("state", "")).lower() != "live":
                continue
            base = str(x.get("baseCcy", "")).upper()
            quote = str(x.get("quoteCcy", "")).upper()
            if quote != "USDT":
                continue
            by_base.setdefault(base, x.get("instId"))

        spot_symbols = {}
        asset_name = {}
        for name, requested in REQUESTED:
            found = None
            for candidate in ALIASES.get(requested, [requested]):
                if candidate.upper() in by_base:
                    found = by_base[candidate.upper()]
                    break
            if found:
                spot_symbols[requested] = found
                asset_name[requested] = name
        provider_status["okx"] = f"spot OK ({len(spot_symbols)} assets)"
        print("OKX spot symbols:", spot_symbols)
    except Exception as e:
        provider_status["okx"] = f"spot ERROR: {e}"
        last_provider_errors["okx_spot"] = str(e)
        print("OKX spot discovery error:", e)


def load_history(asset):
    symbol = spot_symbols[asset]
    data = okx("/api/v5/market/candles", {
        "instId": symbol,
        "bar": INTERVAL,
        "limit": min(HISTORY, 300),
    })
    rows = []
    now = int(time.time() * 1000)
    for k in data.get("data", []):
        if len(k) < 9:
            continue
        # OKX: ts,o,h,l,c,vol,volCcy,volCcyQuote,confirm
        if str(k[8]) != "1":
            continue
        rows.append({
            "open_time": int(k[0]), "open": safe_float(k[1]),
            "high": safe_float(k[2]), "low": safe_float(k[3]),
            "close": safe_float(k[4]), "volume": safe_float(k[5]),
            "close_time": int(k[0]) + 5 * 60 * 1000 - 1,
        })
    rows = [r for r in rows if np.isfinite(r["close"])]
    rows.sort(key=lambda x: x["open_time"])
    with lock:
        bars[symbol] = deque(rows[-HISTORY:], maxlen=HISTORY)
        if rows:
            last_bar[symbol] = rows[-1]["open_time"]
    return len(rows)


def refresh_latest_bar(asset):
    symbol = spot_symbols[asset]
    data = okx("/api/v5/market/candles", {
        "instId": symbol,
        "bar": INTERVAL,
        "limit": 3,
    })
    now_ms = int(time.time() * 1000)
    added = 0
    for k in reversed(data.get("data", [])):
        if len(k) < 9 or str(k[8]) != "1":
            continue
        b = {
            "open_time": int(k[0]), "open": safe_float(k[1]),
            "high": safe_float(k[2]), "low": safe_float(k[3]),
            "close": safe_float(k[4]), "volume": safe_float(k[5]),
            "close_time": int(k[0]) + 5 * 60 * 1000 - 1,
        }
        if b["close_time"] > now_ms or not np.isfinite(b["close"]):
            continue
        with lock:
            if last_bar.get(symbol) == b["open_time"]:
                continue
            bars.setdefault(symbol, deque(maxlen=HISTORY)).append(b)
            last_bar[symbol] = b["open_time"]
        analyze_asset(asset)
        stats["closed_bars"] += 1
        added += 1
    return added


# ============================================================
# OPTION DISCOVERY / NORMALIZATION
# ============================================================

def normalize_okx_contract(x):
    return {
        "provider": "OKX",
        "symbol": str(x.get("instId", "")).upper(),
        "asset": str(x.get("uly", "")).split("-")[0].upper(),
        "expiry_ms": int(safe_float(x.get("expTime"), 0)),
        "strike": safe_float(x.get("stk")),
        "side": "CALL" if str(x.get("optType", "")).upper() == "C" else "PUT",
        "inst_family": x.get("instFamily") or x.get("uly"),
        "state": str(x.get("state", "live")).lower(),
    }


def normalize_deribit_contract(x):
    return {
        "provider": "DERIBIT",
        "symbol": str(x.get("instrument_name", "")).upper(),
        "asset": str(x.get("base_currency", "")).upper(),
        "expiry_ms": int(safe_float(x.get("expiration_timestamp"), 0)),
        "strike": safe_float(x.get("strike")),
        "side": "CALL" if str(x.get("option_type", "")).lower() == "call" else "PUT",
        "inst_family": None,
        "state": str(x.get("state", "open")).lower(),
    }


def discover_okx_options():
    found = defaultdict(list)
    try:
        data = okx("/api/v5/public/instruments", {"instType": "OPTION"})
        for x in data.get("data", []):
            if str(x.get("state", "")).lower() != "live":
                continue
            uly = str(x.get("uly", "")).upper()
            base = uly.split("-")[0]
            requested = None
            for _, a in REQUESTED:
                if base in ALIASES.get(a, [a]):
                    requested = a
                    break
            if requested:
                c = normalize_okx_contract(x)
                if c["expiry_ms"] > int(time.time() * 1000):
                    found[requested].append(c)
        if found:
            provider_status["okx"] = f"options OK ({sum(map(len, found.values()))} contracts)"
        return found
    except Exception as e:
        provider_status["okx"] = f"options ERROR: {e}"
        last_provider_errors["okx_options"] = str(e)
        print("OKX option discovery error:", e)
        return found


def discover_deribit_options():
    found = defaultdict(list)
    for asset in ("BTC", "ETH"):
        try:
            rows = deribit("public/get_instruments", {"currency": asset, "kind": "option", "expired": False})
            for x in rows or []:
                if not x.get("is_active", False):
                    continue
                c = normalize_deribit_contract(x)
                if c["expiry_ms"] > int(time.time() * 1000):
                    found[asset].append(c)
            print("Deribit", asset, "contracts:", len(found[asset]))
        except Exception as e:
            last_provider_errors[f"deribit_{asset}"] = str(e)
            print("Deribit", asset, "discovery error:", e)
    if found:
        provider_status["deribit"] = f"options OK ({sum(map(len, found.values()))} contracts)"
    else:
        provider_status["deribit"] = "options unavailable"
    return found


def discover_option_contracts(force=False):
    global contracts, option_provider, last_option_discovery
    now = time.time()
    if not force and now - last_option_discovery < OPTION_REFRESH_MIN * 60:
        return

    okx_found = discover_okx_options()
    deribit_found = discover_deribit_options()

    new_contracts = {}
    new_provider = {}
    for _, asset in REQUESTED:
        # Prefer OKX where available; fallback to Deribit for BTC/ETH.
        if okx_found.get(asset):
            new_contracts[asset] = okx_found[asset]
            new_provider[asset] = "OKX"
        elif deribit_found.get(asset):
            new_contracts[asset] = deribit_found[asset]
            new_provider[asset] = "DERIBIT"

    with lock:
        contracts = new_contracts
        option_provider = new_provider
        last_option_discovery = now

    print("\n================ OPTION DISCOVERY ================")
    for name, asset in REQUESTED:
        if asset in contracts:
            exps = sorted({c["expiry_ms"] for c in contracts[asset]})
            print(asset, option_provider[asset], "ACTIVE CONTRACTS:", len(contracts[asset]),
                  "EXPIRIES:", len(exps))
        else:
            print(asset, "SKIPPED - no active option contract found on OKX or Deribit")
    print("==================================================\n")


# ============================================================
# OPTION MARKET DATA
# ============================================================

def option_expiries_for_asset(asset):
    now = int(time.time() * 1000)
    return sorted({c["expiry_ms"] for c in contracts.get(asset, [])
                   if c["expiry_ms"] > now + MIN_DTE_HOURS * 3600000})


def nearest_expiry(asset):
    exps = option_expiries_for_asset(asset)
    return exps[0] if exps else None


def fetch_okx_option_market(asset, expiry_ms):
    rows = [c for c in contracts.get(asset, []) if c["expiry_ms"] == expiry_ms]
    if not rows:
        return {}

    family = rows[0].get("inst_family")
    market = {}

    # 1) Tickers: bid/ask/last/volume.
    try:
        tick = okx("/api/v5/market/tickers", {"instType": "OPTION"})
        for t in tick.get("data", []):
            sym = str(t.get("instId", "")).upper()
            if sym not in {r["symbol"] for r in rows}:
                continue
            market[sym] = {
                "symbol": sym,
                "ltp": safe_float(t.get("last"), 0),
                "bid": safe_float(t.get("bidPx"), 0),
                "ask": safe_float(t.get("askPx"), 0),
                "volume": safe_float(t.get("vol24h"), 0),
                "oi": np.nan,
            }
    except Exception as e:
        last_provider_errors["okx_option_tickers"] = str(e)
        print("OKX option ticker error:", e)

    # 2) Option summary contains greeks/IV/forward fields.
    try:
        params = {"instFamily": family} if family else {"uly": f"{asset}-USD"}
        data = okx("/api/v5/public/opt-summary", params)
        for x in data.get("data", []):
            sym = str(x.get("instId", "")).upper()
            if sym not in {r["symbol"] for r in rows}:
                continue
            m = market.setdefault(sym, {"symbol": sym})
            for dst, src in [
                ("delta", "deltaBS"), ("gamma", "gammaBS"),
                ("theta", "thetaBS"), ("vega", "vegaBS"),
                ("mark_iv", "markVol"), ("bid_iv", "bidVol"),
                ("ask_iv", "askVol"), ("forward", "fwdPx"),
                ("index_price", "idxPx"),
            ]:
                if src in x:
                    m[dst] = safe_float(x.get(src))
            # Some OKX responses use delta/gamma/... without BS suffix.
            for key in ("delta", "gamma", "theta", "vega", "markVol", "bidVol", "askVol"):
                if key in x and key not in m:
                    m[key] = safe_float(x.get(key))
    except Exception as e:
        last_provider_errors["okx_opt_summary"] = str(e)
        print("OKX option summary error:", e)

    # 3) Open interest by instrument when available.
    for r in rows:
        sym = r["symbol"]
        if sym not in market:
            continue
        try:
            oi_data = okx("/api/v5/public/open-interest", {
                "instType": "OPTION", "instId": sym
            })
            vals = oi_data.get("data", [])
            if vals:
                market[sym]["oi"] = safe_float(vals[0].get("oi"), np.nan)
        except Exception:
            # OI is a liquidity bonus, not a hard dependency.
            pass

    return market


def fetch_deribit_option_market(asset, expiry_ms):
    rows = [c for c in contracts.get(asset, []) if c["expiry_ms"] == expiry_ms]
    market = {}
    for c in rows:
        try:
            x = deribit("public/ticker", {"instrument_name": c["symbol"]})
            if not x:
                continue
            g = x.get("greeks") or {}
            stats_x = x.get("stats") or {}
            market[c["symbol"]] = {
                "symbol": c["symbol"],
                "ltp": safe_float(x.get("last_price"), 0),
                "bid": safe_float(x.get("best_bid_price"), 0),
                "ask": safe_float(x.get("best_ask_price"), 0),
                "volume": safe_float(stats_x.get("volume"), 0),
                "oi": safe_float(x.get("open_interest"), np.nan),
                "mark_price": safe_float(x.get("mark_price")),
                "delta": safe_float(g.get("delta")),
                "gamma": safe_float(g.get("gamma")),
                "theta": safe_float(g.get("theta")),
                "vega": safe_float(g.get("vega")),
                "mark_iv": safe_float(x.get("mark_iv")),
                "bid_iv": safe_float(x.get("bid_iv")),
                "ask_iv": safe_float(x.get("ask_iv")),
                "index_price": safe_float(x.get("index_price")),
            }
        except Exception as e:
            last_provider_errors[f"deribit_ticker_{c['symbol']}"] = str(e)
    return market


def build_option_market(asset, expiry_ms):
    provider = option_provider.get(asset)
    if provider == "OKX":
        return fetch_okx_option_market(asset, expiry_ms)
    if provider == "DERIBIT":
        return fetch_deribit_option_market(asset, expiry_ms)
    return {}


def select_option(asset, direction, spot):
    if asset not in contracts or not spot or not np.isfinite(spot):
        return None

    expiry = nearest_expiry(asset)
    if not expiry:
        return None

    market = build_option_market(asset, expiry)
    needed = "CALL" if direction == "LONG" else "PUT"
    candidates = []

    for c in contracts[asset]:
        if c["expiry_ms"] != expiry or c["side"] != needed:
            continue
        m = market.get(c["symbol"])
        if not m:
            continue

        strike = safe_float(c.get("strike"), 0)
        ltp = safe_float(m.get("ltp"), 0)
        bid = safe_float(m.get("bid"), 0)
        ask = safe_float(m.get("ask"), 0)
        volume = safe_float(m.get("volume"), 0)
        oi = safe_float(m.get("oi"), np.nan)
        delta = safe_float(m.get("delta"), np.nan)

        if strike <= 0 or ltp <= 0:
            continue
        distance = abs(strike - spot) / spot
        if distance > MAX_STRIKE_DISTANCE:
            continue
        if volume < MIN_OPTION_VOLUME:
            continue
        if np.isfinite(oi) and oi < MIN_OPTION_OI:
            continue
        if bid <= 0 or ask <= 0 or ask < bid:
            continue

        spread = (ask - bid) / ((ask + bid) / 2)
        if spread > MAX_SPREAD:
            continue

        if not np.isfinite(delta):
            continue
        ad = abs(delta)
        if ad < MIN_DELTA or ad > MAX_DELTA:
            continue

        delta_score = 1 - abs(ad - 0.50) / 0.50
        distance_score = max(0, 1 - distance / MAX_STRIKE_DISTANCE)
        spread_score = max(0, 1 - spread / MAX_SPREAD)
        volume_score = min(1, volume / 100)
        oi_score = min(1, oi / 1000) if np.isfinite(oi) else 0.25

        selection_score = (
            delta_score * 35 + distance_score * 30 +
            spread_score * 20 + volume_score * 10 + oi_score * 5
        )

        row = dict(c)
        row.update(m)
        row["spread"] = spread
        row["distance"] = distance
        row["selection_score"] = selection_score
        candidates.append(row)

    if not candidates:
        return None
    candidates.sort(key=lambda x: x["selection_score"], reverse=True)
    return candidates[0]


# ============================================================
# ALERTS
# ============================================================

def option_plan(premium):
    return (
        premium * (1 - PREMIUM_SL),
        premium * (1 + PREMIUM_T1),
        premium * (1 + PREMIUM_T2),
    )


def send_option_confirm(asset, signal, option):
    direction = signal["side"]
    option_side = "CE" if direction == "LONG" else "PE"
    premium = option["ltp"]
    sl, t1, t2 = option_plan(premium)
    expiry_dt = datetime.fromtimestamp(option["expiry_ms"] / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    emoji = "🟢" if option_side == "CE" else "🔴"
    provider = option.get("provider", option_provider.get(asset, "-"))

    text = f"""<b>{emoji} CRYPTO OPTION BUY {option_side}</b>
<b>{asset_name.get(asset, asset)} ({asset})</b>
━━━━━━━━━━━━━━━━━━━━
<b>{grade(signal['score'])}</b> | Score <b>{signal['score']}/10</b>
Timeframe: <b>5 MIN</b>
Trigger: <b>{signal['trigger']}</b>
Provider: <b>{provider}</b>

<b>UNDERLYING</b>
Spot: <b>{fmt(signal['spot'])}</b>
EMA9: {fmt(signal['ema9'])}
EMA20: {fmt(signal['ema20'])}
VWAP: {fmt(signal['vwap'])}
RSI: {signal['rsi']:.1f}
ADX: {signal['adx']:.1f}
DI+: {signal['di_plus']:.1f} | DI-: {signal['di_minus']:.1f}
Volume: {signal['vol_ratio']:.2f}x
ATR: {fmt(signal['atr'])}

<b>OPTION</b>
Symbol: <b>{option['symbol']}</b>
Type: <b>{option_side}</b>
Expiry: <b>{expiry_dt}</b>
Strike: <b>{fmt(option['strike'])}</b>
Premium/LTP: <b>{fmt(option['ltp'])}</b>
Bid: {fmt(option.get('bid'))}
Ask: {fmt(option.get('ask'))}
Spread: {option.get('spread', 0)*100:.2f}%
Delta: {fmt(option.get('delta'))}
Gamma: {fmt(option.get('gamma'))}
Theta: {fmt(option.get('theta'))}
Vega: {fmt(option.get('vega'))}
IV: {fmt(option.get('mark_iv'))}
OI: {fmt(option.get('oi'))}
Volume: {fmt(option.get('volume'))}

<b>TRADE PLAN — BUY OPTION</b>
Entry: <b>{fmt(premium)}</b>
Entry zone: <b>{fmt(premium*0.98)} – {fmt(premium*1.02)}</b>
🛑 SL: <b>{fmt(sl)}</b> (-25%)
🎯 T1: <b>{fmt(t1)}</b> (+40%)
🎯 T2: <b>{fmt(t2)}</b> (+70%)
After T1: move SL to breakeven.
Risk: <b>0.5–1% maximum</b>.

<b>WHY</b>
✓ EMA9/EMA20 trend
✓ VWAP confirmation
✓ RSI momentum
✓ ADX trend strength
✓ DI direction
✓ Volume confirmation
✓ Near-ATM strike
✓ Delta 0.35–0.65
✓ Spread <= 2.5%
✓ Active expiry

⚠️ Signal only — no automatic order.
⚠️ Option premium can change rapidly because of IV, theta, spread and underlying movement.
⚠️ Do not chase above the entry zone."""
    telegram(text)


def send_early(asset, signal):
    option_side = "CALL" if signal["side"] == "LONG" else "PUT"
    emoji = "⚡🟢" if signal["side"] == "LONG" else "⚡🔴"
    text = f"""<b>{emoji} EARLY CRYPTO {option_side} WATCH</b>
<b>{asset_name.get(asset, asset)} ({asset})</b>
━━━━━━━━━━━━━━━━━━━━
<b>{grade(signal['score'])}</b> | Score <b>{signal['score']}/10</b>
5-minute pre-crossover momentum

Spot: <b>{fmt(signal['spot'])}</b>
EMA9: {fmt(signal['ema9'])}
EMA20: {fmt(signal['ema20'])}
VWAP: {fmt(signal['vwap'])}
RSI: {signal['rsi']:.1f}
ADX: {signal['adx']:.1f}
DI+: {signal['di_plus']:.1f} | DI-: {signal['di_minus']:.1f}
Volume: {signal['vol_ratio']:.2f}x

Expected option direction:
<b>BUY {option_side}</b>

⚠️ EARLY signal is not a confirmed option entry.
Wait for underlying confirmation and a liquid option contract."""
    telegram(text)


# ============================================================
# SIGNAL ENGINE
# ============================================================

def analyze_asset(asset):
    symbol = spot_symbols.get(asset)
    if not symbol:
        return
    with lock:
        rows = list(bars.get(symbol, []))
    d = build_indicators(rows)
    if d is None:
        return
    c = d.iloc[-1]
    p = d.iloc[-2]
    needed = [c.ema9, c.ema20, c.vwap, c.rsi, c.adx, c.di_plus,
              c.di_minus, c.atr, c.vol_ratio, c.body, c.ema_gap]
    if any(pd.isna(x) for x in needed):
        return
    if c.adx < SIDEWAYS_ADX:
        return

    cross_l = p.ema9 <= p.ema20 and c.ema9 > c.ema20
    cross_s = p.ema9 >= p.ema20 and c.ema9 < c.ema20

    bull = (c.ema9 > c.ema20 and c.close > c.vwap and c.rsi > CONF_RSI_LONG and
            c.adx >= MIN_ADX and c.di_plus > c.di_minus and c.vol_ratio >= 1.15)
    bear = (c.ema9 < c.ema20 and c.close < c.vwap and c.rsi < CONF_RSI_SHORT and
            c.adx >= MIN_ADX and c.di_minus > c.di_plus and c.vol_ratio >= 1.15)

    candidates = []
    if cross_l and bull and c.close > c.ema9:
        candidates.append(("LONG", score(c, "LONG", True), "EMA CROSSOVER"))
    if cross_s and bear and c.close < c.ema9:
        candidates.append(("SHORT", score(c, "SHORT", True), "EMA CROSSOVER"))

    rng = max(c.high - c.low, 1e-12)
    pull_l = (c.ema9 > c.ema20 and c.close > c.vwap and
              c.low <= c.ema9 + c.atr * 0.20 and c.close > c.open and
              c.close >= c.low + rng * 0.65 and c.rsi > 52 and
              c.adx >= MIN_ADX and c.di_plus > c.di_minus and c.vol_ratio >= 1.15)
    pull_s = (c.ema9 < c.ema20 and c.close < c.vwap and
              c.high >= c.ema9 - c.atr * 0.20 and c.close < c.open and
              c.close <= c.high - rng * 0.65 and c.rsi < 48 and
              c.adx >= MIN_ADX and c.di_minus > c.di_plus and c.vol_ratio >= 1.15)

    if pull_l:
        candidates.append(("LONG", score(c, "LONG", True), "EMA9 PULLBACK"))
    if pull_s:
        candidates.append(("SHORT", score(c, "SHORT", True), "EMA9 PULLBACK"))

    dist = abs(c.ema9 - c.ema20)
    prevdist = abs(p.ema9 - p.ema20)
    early_l = (c.ema9 < c.ema20 and c.ema9 > p.ema9 and dist < prevdist and
               c.close > c.vwap and c.rsi > EARLY_RSI_LONG and
               c.adx >= SIDEWAYS_ADX and c.adx > p.adx and
               c.di_plus >= c.di_minus and c.vol_ratio >= 1.15 and not cross_l)
    early_s = (c.ema9 > c.ema20 and c.ema9 < p.ema9 and dist < prevdist and
               c.close < c.vwap and c.rsi < EARLY_RSI_SHORT and
               c.adx >= SIDEWAYS_ADX and c.adx > p.adx and
               c.di_minus >= c.di_plus and c.vol_ratio >= 1.15 and not cross_s)

    if candidates:
        candidates.sort(key=lambda x: x[1], reverse=True)
        side, sc, trigger = candidates[0]
        if sc >= MIN_SCORE:
            signal = {
                "side": side, "score": sc, "trigger": trigger,
                "spot": float(c.close), "ema9": float(c.ema9),
                "ema20": float(c.ema20), "vwap": float(c.vwap),
                "rsi": float(c.rsi), "adx": float(c.adx),
                "di_plus": float(c.di_plus), "di_minus": float(c.di_minus),
                "vol_ratio": float(c.vol_ratio), "atr": float(c.atr),
                "open_time": int(c.open_time),
            }
            try:
                option = select_option(asset, side, signal["spot"])
                if option:
                    key = (asset, side, option["symbol"], signal["open_time"])
                    cooldown_key = (asset, side, "CONF")
                    now = time.time()
                    if key not in seen and now - last_alert.get(cooldown_key, 0) >= CONF_COOLDOWN_MIN * 60:
                        send_option_confirm(asset, signal, option)
                        seen.add(key)
                        last_alert[cooldown_key] = now
                        stats["confirmed"] += 1
            except Exception as e:
                stats["errors"] += 1
                print("Option selection error", asset, e)

    if early_l or early_s:
        side = "LONG" if early_l else "SHORT"
        sc = score(c, side, False)
        if sc >= MIN_SCORE:
            signal2 = {
                "side": side, "score": sc, "spot": float(c.close),
                "ema9": float(c.ema9), "ema20": float(c.ema20),
                "vwap": float(c.vwap), "rsi": float(c.rsi),
                "adx": float(c.adx), "di_plus": float(c.di_plus),
                "di_minus": float(c.di_minus), "vol_ratio": float(c.vol_ratio),
            }
            cooldown_key = (asset, side, "EARLY")
            if time.time() - last_alert.get(cooldown_key, 0) >= EARLY_COOLDOWN_MIN * 60:
                send_early(asset, signal2)
                last_alert[cooldown_key] = time.time()
                stats["early"] += 1


# ============================================================
# SCANNER / HEALTH
# ============================================================

def scan_loop():
    next_option_refresh = 0
    while True:
        try:
            if time.time() >= next_option_refresh:
                discover_option_contracts(force=True)
                next_option_refresh = time.time() + OPTION_REFRESH_MIN * 60
        except Exception as e:
            stats["errors"] += 1
            print("Option refresh error:", e)

        # Scan every requested asset with an OKX spot symbol.
        for asset in list(spot_symbols.keys()):
            try:
                refresh_latest_bar(asset)
            except Exception as e:
                stats["errors"] += 1
                last_provider_errors[f"spot_{asset}"] = str(e)
                print("Scan error", asset, e)
            time.sleep(0.15)
        time.sleep(SCAN_SLEEP)


@app.get("/")
def home():
    return jsonify({
        "bot": BOT_NAME,
        "status": "running",
        "timeframe": INTERVAL,
        "underlying_provider": "OKX public market data",
        "option_providers": sorted(set(option_provider.values())),
        "option_assets": sorted(list(contracts.keys())),
        "skipped_assets": [a for _, a in REQUESTED if a not in contracts],
        "min_score": MIN_SCORE,
        "stats": dict(stats),
    })


@app.get("/health")
def health():
    return jsonify({
        "ok": True,
        "timeframe": INTERVAL,
        "option_assets": len(contracts),
        "option_providers": option_provider,
        "bars": {a: len(bars.get(spot_symbols[a], [])) for a in spot_symbols},
        "confirmed": stats["confirmed"],
        "early": stats["early"],
        "errors": stats["errors"],
        "provider_status": provider_status,
        "last_provider_errors": dict(list(last_provider_errors.items())[-20:]),
    })


def startup_message():
    active = ", ".join(f"{a}:{option_provider[a]}" for a in sorted(contracts)) or "NONE"
    skipped = ", ".join(a for _, a in REQUESTED if a not in contracts) or "NONE"
    telegram(f"""🟢 <b>{BOT_NAME} ONLINE</b>

<b>Mode:</b> Crypto Options Signal Scanner
<b>Timeframe:</b> 5 minutes
<b>Minimum score:</b> {MIN_SCORE}/10
<b>Underlying:</b> OKX public market data
<b>Options:</b> OKX primary + Deribit fallback

<b>ACTIVE OPTION UNDERLYINGS</b>
{active}

<b>AUTOMATICALLY SKIPPED</b>
{skipped}

Signals:
⚡ EARLY CALL/PUT WATCH
🟢 CONFIRMED CALL BUY
🔴 CONFIRMED PUT BUY

Option filters:
✓ Delta 0.35–0.65
✓ Near-ATM strike
✓ Spread <= 2.5%
✓ Volume / liquidity
✓ OI when supplied by provider
✓ Minimum 12h to expiry
✓ Underlying trend confirmation

No automatic orders are placed.

⚠️ Option availability depends on the provider's currently listed contracts.""")


def main():
    print("=" * 70)
    print(BOT_NAME)
    print("NO BINANCE VERSION - OKX + DERIBIT")
    print("=" * 70)

    discover_spot_symbols()
    discover_option_contracts(force=True)

    for asset in list(spot_symbols):
        try:
            print("Warmup", asset, load_history(asset))
        except Exception as e:
            stats["errors"] += 1
            print("Warmup failed", asset, e)
        time.sleep(0.15)

    threading.Thread(
        target=lambda: app.run(
            host="0.0.0.0",
            port=int(os.getenv("PORT", "10000")),
            threaded=True,
            use_reloader=False,
        ), daemon=True,
    ).start()

    startup_message()
    threading.Thread(target=scan_loop, daemon=True).start()

    while True:
        time.sleep(60)


if __name__ == "__main__":
    main()
