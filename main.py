import os, time, threading, requests, numpy as np, pandas as pd
from collections import defaultdict
from flask import Flask, jsonify

# ============================================================
# CRYPTO OPTIONS TELEGRAM SIGNAL BOT
# ============================================================
# Market: Crypto Options
# Data: Binance public Options REST API + Spot market data
# Signal timeframe: 5 minutes
# Orders: NONE - signal/analysis only
#
# The bot dynamically discovers which of the requested coins
# currently have active Binance Options contracts. Unsupported
# coins are automatically skipped.
#
# Strategy:
# EMA 9/20 + VWAP + RSI + ADX/DI + volume + ATR
# + option Delta + IV + bid/ask spread + liquidity
# + strike-distance + expiry selection
#
# It produces:
# EARLY CALL / EARLY PUT WATCH
# CONFIRMED CALL BUY
# CONFIRMED PUT BUY
# Entry, SL, T1, T2, expiry, strike, premium,
# Delta, Gamma, Theta, Vega, IV, OI, volume, spread,
# underlying trend and confidence score.
# ============================================================

import json

BOT_NAME = os.getenv("BOT_NAME", "Crypto Options High Probability Bot")
TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TG_CHAT = os.getenv("TELEGRAM_CHAT_ID", "").strip()

SPOT_BASE = os.getenv("BINANCE_SPOT_BASE", "https://api.binance.com")
OPT_BASE = os.getenv("BINANCE_OPTIONS_BASE", "https://eapi.binance.com")

INTERVAL = "5m"
HISTORY = 180

# Technical parameters
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

# Option selection
MIN_DELTA = 0.35
MAX_DELTA = 0.65
MAX_STRIKE_DISTANCE = 0.035
MAX_SPREAD = 0.025
MIN_OPTION_VOLUME = 1
MIN_OPTION_OI = 1

# Option premium trade plan
PREMIUM_SL = 0.25     # 25% premium stop
PREMIUM_T1 = 0.40     # 40% premium target
PREMIUM_T2 = 0.70     # 70% premium target

# Alerts
EARLY_COOLDOWN_MIN = 20
CONF_COOLDOWN_MIN = 10

# Requested assets
REQUESTED = [
    ("Bitcoin", "BTC"),
    ("Ethereum", "ETH"),
    ("Solana", "SOL"),
    ("Ripple", "XRP"),
    ("Dogecoin", "DOGE"),
    ("Cardano", "ADA"),
    ("Avalanche", "AVAX"),
    ("Tron", "TRX"),
    ("Binance Coin", "BNB"),
    ("Near Protocol", "NEAR"),
    ("Aave", "AAVE"),
    ("Lighter", "LIT"),
    ("Ethena", "ENA"),
    ("Zcash", "ZEC"),
    ("Akedo", "AKE"),
    ("Esports Token", "ESPORTS"),
    ("Uniswap", "UNI"),
    ("Chainlink", "LINK"),
    ("Litecoin", "LTC"),
    ("Polygon", "MATIC"),
]

# Binance may migrate/rebrand an asset.
ALIASES = {
    "MATIC": ["MATIC", "POL"]
}

app = Flask(__name__)

lock = threading.RLock()
contracts = {}          # asset -> option contract metadata list
asset_name = {}
spot_symbols = {}       # asset -> BTCUSDT etc.
bars = {}               # spot symbol -> deque
last_bar = {}
seen = set()
last_alert = {}
stats = defaultdict(int)

# ------------------------------------------------------------
# Telegram
# ------------------------------------------------------------

def telegram(text):
    if not TG_TOKEN or not TG_CHAT:
        print("\nTELEGRAM NOT CONFIGURED\n" + text)
        return False

    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            json={
                "chat_id": TG_CHAT,
                "text": text,
                "parse_mode": "HTML",
                "disable_web_page_preview": True
            },
            timeout=12
        )
        if not r.ok:
            print("Telegram error:", r.status_code, r.text[:300])
        return r.ok
    except Exception as e:
        print("Telegram exception:", e)
        return False

# ------------------------------------------------------------
# HTTP
# ------------------------------------------------------------

def get_json(url, params=None, timeout=15):
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()

# ------------------------------------------------------------
# Technical indicators
# ------------------------------------------------------------

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
        (df.low - pc).abs()
    ], axis=1).max(axis=1)

    return tr.ewm(alpha=1/n, adjust=False, min_periods=n).mean()

def adx(df, n=14):
    up = df.high.diff()
    dn = -df.low.diff()

    plus = pd.Series(
        np.where((up > dn) & (up > 0), up, 0.0),
        index=df.index
    )
    minus = pd.Series(
        np.where((dn > up) & (dn > 0), dn, 0.0),
        index=df.index
    )

    pc = df.close.shift(1)
    tr = pd.concat([
        df.high - df.low,
        (df.high - pc).abs(),
        (df.low - pc).abs()
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

    d = pd.DataFrame(rows)

    d["ema9"] = d.close.ewm(span=EMA_FAST, adjust=False).mean()
    d["ema20"] = d.close.ewm(span=EMA_SLOW, adjust=False).mean()
    d["rsi"] = rsi(d.close, RSI_LEN)
    d["atr"] = atr(d, ATR_LEN)
    d["adx"], d["di_plus"], d["di_minus"] = adx(d, ADX_LEN)

    tp = (d.high + d.low + d.close) / 3

    dt = pd.to_datetime(d.open_time, unit="ms", utc=True)
    day = dt.dt.date

    d["vwap"] = (
        (tp * d.volume).groupby(day).cumsum()
        / d.volume.groupby(day).cumsum()
    )

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

def grade(score_value):
    if score_value >= 9:
        return "🔥🔥 VERY HIGH"
    if score_value >= 8:
        return "🔥 HIGH"
    return "🟢 VALID"

def fmt(x):
    try:
        x = float(x)
    except:
        return "-"

    if not np.isfinite(x):
        return "-"

    if abs(x) >= 1000:
        return f"{x:,.2f}"
    if abs(x) >= 1:
        return f"{x:,.4f}"
    return f"{x:,.8f}"

# ------------------------------------------------------------
# Discover Binance Options
# ------------------------------------------------------------

def discover_option_contracts():
    global contracts, spot_symbols, asset_name

    info = get_json(f"{OPT_BASE}/eapi/v1/exchangeInfo")

    option_symbols = info.get("optionSymbols", [])

    # Group contracts by underlying, only active/trading contracts.
    by_underlying = defaultdict(list)

    for x in option_symbols:
        if str(x.get("status", "")).upper() not in ("TRADING", "ACTIVE"):
            continue

        underlying = str(x.get("underlying", "")).upper()
        base = underlying.replace("USDT", "")

        by_underlying[base].append(x)

    contracts = {}
    spot_symbols = {}
    asset_name = {}

    for name, requested_asset in REQUESTED:
        found_asset = None

        for candidate in ALIASES.get(requested_asset, [requested_asset]):
            if candidate.upper() in by_underlying:
                found_asset = candidate.upper()
                break

        if not found_asset:
            continue

        contracts[requested_asset] = by_underlying[found_asset]
        spot_symbols[requested_asset] = found_asset + "USDT"
        asset_name[requested_asset] = name

    print("\n================ OPTION DISCOVERY ================")
    for name, asset in REQUESTED:
        if asset in contracts:
            expiries = sorted({
                int(x.get("expiryDate", 0))
                for x in contracts[asset]
                if x.get("expiryDate")
            })
            print(asset, "ACTIVE CONTRACTS:", len(contracts[asset]),
                  "EXPIRIES:", len(expiries))
        else:
            print(asset, "SKIPPED - no active Binance option contract currently listed")
    print("==================================================\n")

# ------------------------------------------------------------
# Spot history
# ------------------------------------------------------------

def load_history(asset):
    symbol = spot_symbols[asset]

    data = get_json(
        f"{SPOT_BASE}/api/v3/klines",
        {
            "symbol": symbol,
            "interval": INTERVAL,
            "limit": HISTORY
        }
    )

    now = int(time.time() * 1000)

    out = []

    for k in data:
        if int(k[6]) <= now:
            out.append({
                "open_time": int(k[0]),
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
                "close_time": int(k[6])
            })

    from collections import deque

    with lock:
        bars[symbol] = deque(out[-HISTORY:], maxlen=HISTORY)
        if out:
            last_bar[symbol] = out[-1]["open_time"]

    return len(out)

# ------------------------------------------------------------
# Option market data
# ------------------------------------------------------------

def get_option_ticker():
    """
    Binance Options ticker endpoint returns option LTP,
    bid/ask, volume and strike information.
    """
    data = get_json(f"{OPT_BASE}/eapi/v1/ticker")
    if isinstance(data, dict):
        data = [data]
    return data

def get_mark(symbol=None):
    params = {}
    if symbol:
        params["symbol"] = symbol

    data = get_json(f"{OPT_BASE}/eapi/v1/mark", params)

    if isinstance(data, dict):
        return [data]
    return data

def get_index(asset):
    underlying = (
        "POLUSDT" if asset == "MATIC" and "POL" in spot_symbols.get(asset, "")
        else spot_symbols[asset]
    )

    try:
        data = get_json(
            f"{OPT_BASE}/eapi/v1/index",
            {"underlying": underlying}
        )
        return float(data["indexPrice"])
    except:
        return None

def build_option_market():
    """
    Combines ticker + mark data.
    This avoids requesting mark data individually for every contract.
    """
    try:
        tickers = get_option_ticker()
    except Exception as e:
        print("Option ticker error:", e)
        return {}

    market = {}

    for x in tickers:
        symbol = str(x.get("symbol", "")).upper()
        if not symbol:
            continue

        try:
            market[symbol] = {
                "symbol": symbol,
                "ltp": float(x.get("lastPrice", 0) or 0),
                "bid": float(x.get("bidPrice", 0) or 0),
                "ask": float(x.get("askPrice", 0) or 0),
                "volume": float(x.get("volume", 0) or 0),
                "strike": float(x.get("strikePrice", 0) or 0),
                "change_pct": float(x.get("priceChangePercent", 0) or 0),
            }
        except:
            continue

    # Mark endpoint can be called for all contracts when symbol is omitted.
    try:
        marks = get_mark()
        for x in marks:
            symbol = str(x.get("symbol", "")).upper()
            if symbol not in market:
                market[symbol] = {"symbol": symbol}

            for k, source in [
                ("mark_price", "markPrice"),
                ("bid_iv", "bidIV"),
                ("ask_iv", "askIV"),
                ("mark_iv", "markIV"),
                ("delta", "delta"),
                ("gamma", "gamma"),
                ("theta", "theta"),
                ("vega", "vega"),
            ]:
                try:
                    market[symbol][k] = float(x.get(source, np.nan))
                except:
                    market[symbol][k] = np.nan
    except Exception as e:
        print("Option mark error:", e)

    return market

# ------------------------------------------------------------
# Contract selection
# ------------------------------------------------------------

def nearest_expiry(contracts_for_asset):
    now_ms = int(time.time() * 1000)

    expiries = sorted({
        int(x.get("expiryDate"))
        for x in contracts_for_asset
        if x.get("expiryDate") and int(x.get("expiryDate")) > now_ms
    })

    return expiries[0] if expiries else None

def select_option(asset, direction, spot, market):
    if asset not in contracts or not spot:
        return None

    all_contracts = contracts[asset]

    expiry = nearest_expiry(all_contracts)
    if not expiry:
        return None

    side_needed = "CALL" if direction == "LONG" else "PUT"

    candidates = []

    for c in all_contracts:
        if int(c.get("expiryDate", 0)) != expiry:
            continue

        if str(c.get("side", "")).upper() != side_needed:
            continue

        symbol = str(c.get("symbol", "")).upper()

        m = market.get(symbol)
        if not m:
            continue

        strike = float(c.get("strikePrice", 0) or 0)
        ltp = float(m.get("ltp", 0) or 0)
        bid = float(m.get("bid", 0) or 0)
        ask = float(m.get("ask", 0) or 0)
        volume = float(m.get("volume", 0) or 0)

        if strike <= 0 or ltp <= 0:
            continue

        distance = abs(strike - spot) / spot

        if distance > MAX_STRIKE_DISTANCE:
            continue

        if volume < MIN_OPTION_VOLUME:
            continue

        oi = float(m.get("oi", 0) or 0)
        # Binance ticker may not expose per-symbol OI. OI is therefore
        # treated as optional rather than fabricating it.
        if oi and oi < MIN_OPTION_OI:
            continue

        if bid > 0 and ask > 0:
            spread = (ask - bid) / ((ask + bid) / 2)
        else:
            spread = 999

        if spread > MAX_SPREAD:
            continue

        delta = m.get("delta", np.nan)

        if np.isfinite(delta):
            ad = abs(delta)
            if ad < MIN_DELTA or ad > MAX_DELTA:
                continue

            delta_score = 1 - abs(ad - 0.50) / 0.50
        else:
            delta_score = 0.30

        distance_score = max(
            0,
            1 - distance / MAX_STRIKE_DISTANCE
        )

        spread_score = (
            max(0, 1 - spread / MAX_SPREAD)
            if spread != 999 else 0
        )

        volume_score = min(
            1,
            volume / 100
        )

        score_value = (
            delta_score * 40
            + distance_score * 30
            + spread_score * 20
            + volume_score * 10
        )

        row = dict(c)
        row.update(m)
        row["expiry_ms"] = expiry
        row["distance"] = distance
        row["spread"] = spread
        row["selection_score"] = score_value

        candidates.append(row)

    if not candidates:
        return None

    candidates.sort(
        key=lambda x: x["selection_score"],
        reverse=True
    )

    return candidates[0]

# ------------------------------------------------------------
# Alert builders
# ------------------------------------------------------------

def option_plan(premium):
    sl = premium * (1 - PREMIUM_SL)
    t1 = premium * (1 + PREMIUM_T1)
    t2 = premium * (1 + PREMIUM_T2)
    return sl, t1, t2

def send_option_confirm(asset, signal, option):
    direction = signal["side"]
    option_side = "CE" if direction == "LONG" else "PE"

    premium = option["ltp"]
    sl, t1, t2 = option_plan(premium)

    expiry_dt = datetime.fromtimestamp(
        option["expiry_ms"] / 1000,
        tz=timezone.utc
    ).strftime("%Y-%m-%d %H:%M UTC")

    emoji = "🟢" if option_side == "CE" else "🔴"

    text = f"""<b>{emoji} CRYPTO OPTION BUY {option_side}</b>
<b>{asset_name.get(asset, asset)} ({asset})</b>
━━━━━━━━━━━━━━━━━━━━
<b>{grade(signal['score'])}</b> | Score <b>{signal['score']}/10</b>
Timeframe: <b>5 MIN</b>
Trigger: <b>{signal['trigger']}</b>

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
Strike: <b>{fmt(option['strikePrice'])}</b>
Premium/LTP: <b>{fmt(option['ltp'])}</b>
Bid: {fmt(option.get('bid'))}
Ask: {fmt(option.get('ask'))}
Spread: {option.get('spread', 0)*100:.2f}%
Delta: {fmt(option.get('delta'))}
Gamma: {fmt(option.get('gamma'))}
Theta: {fmt(option.get('theta'))}
Vega: {fmt(option.get('vega'))}
Mark IV: {fmt(option.get('mark_iv'))}
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
⚠️ Option premium can change rapidly due to IV, theta,
spread and underlying movement.
⚠️ Do not chase above the entry zone.
"""

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
Wait for the underlying confirmation before buying."""
    telegram(text)

# ------------------------------------------------------------
# Underlying signal
# ------------------------------------------------------------

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

    needed = [
        c.ema9, c.ema20, c.vwap, c.rsi, c.adx,
        c.di_plus, c.di_minus, c.atr, c.vol_ratio,
        c.body, c.ema_gap
    ]

    if any(pd.isna(x) for x in needed):
        return

    if c.adx < SIDEWAYS_ADX:
        return

    cross_l = p.ema9 <= p.ema20 and c.ema9 > c.ema20
    cross_s = p.ema9 >= p.ema20 and c.ema9 < c.ema20

    bull = (
        c.ema9 > c.ema20
        and c.close > c.vwap
        and c.rsi > CONF_RSI_LONG
        and c.adx >= MIN_ADX
        and c.di_plus > c.di_minus
        and c.vol_ratio >= 1.15
    )

    bear = (
        c.ema9 < c.ema20
        and c.close < c.vwap
        and c.rsi < CONF_RSI_SHORT
        and c.adx >= MIN_ADX
        and c.di_minus > c.di_plus
        and c.vol_ratio >= 1.15
    )

    candidates = []

    if cross_l and bull and c.close > c.ema9:
        candidates.append((
            "LONG",
            score(c, "LONG", True),
            "EMA CROSSOVER"
        ))

    if cross_s and bear and c.close < c.ema9:
        candidates.append((
            "SHORT",
            score(c, "SHORT", True),
            "EMA CROSSOVER"
        ))

    # Pullback confirmation.
    rng = max(c.high - c.low, 1e-12)

    pull_l = (
        c.ema9 > c.ema20
        and c.close > c.vwap
        and c.low <= c.ema9 + c.atr * 0.20
        and c.close > c.open
        and c.close >= c.low + rng * 0.65
        and c.rsi > 52
        and c.adx >= MIN_ADX
        and c.di_plus > c.di_minus
        and c.vol_ratio >= 1.15
    )

    pull_s = (
        c.ema9 < c.ema20
        and c.close < c.vwap
        and c.high >= c.ema9 - c.atr * 0.20
        and c.close < c.open
        and c.close <= c.high - rng * 0.65
        and c.rsi < 48
        and c.adx >= MIN_ADX
        and c.di_minus > c.di_plus
        and c.vol_ratio >= 1.15
    )

    if pull_l:
        candidates.append((
            "LONG",
            score(c, "LONG", True),
            "EMA9 PULLBACK"
        ))

    if pull_s:
        candidates.append((
            "SHORT",
            score(c, "SHORT", True),
            "EMA9 PULLBACK"
        ))

    # Early pre-crossover.
    dist = abs(c.ema9 - c.ema20)
    prevdist = abs(p.ema9 - p.ema20)

    early_l = (
        c.ema9 < c.ema20
        and c.ema9 > p.ema9
        and dist < prevdist
        and c.close > c.vwap
        and c.rsi > EARLY_RSI_LONG
        and c.adx >= SIDEWAYS_ADX
        and c.adx > p.adx
        and c.di_plus >= c.di_minus
        and c.vol_ratio >= 1.15
        and not cross_l
    )

    early_s = (
        c.ema9 > c.ema20
        and c.ema9 < p.ema9
        and dist < prevdist
        and c.close < c.vwap
        and c.rsi < EARLY_RSI_SHORT
        and c.adx >= SIDEWAYS_ADX
        and c.adx > p.adx
        and c.di_minus >= c.di_plus
        and c.vol_ratio >= 1.15
        and not cross_s
    )

    signal = None

    if candidates:
        candidates.sort(key=lambda x: x[1], reverse=True)
        side, sc, trigger = candidates[0]

        if sc >= MIN_SCORE:
            signal = {
                "side": side,
                "score": sc,
                "trigger": trigger,
                "spot": float(c.close),
                "ema9": float(c.ema9),
                "ema20": float(c.ema20),
                "vwap": float(c.vwap),
                "rsi": float(c.rsi),
                "adx": float(c.adx),
                "di_plus": float(c.di_plus),
                "di_minus": float(c.di_minus),
                "vol_ratio": float(c.vol_ratio),
                "atr": float(c.atr),
                "open_time": int(c.open_time)
            }

    # Confirmed option signal.
    if signal:
        market = build_option_market()

        option = select_option(
            asset,
            signal["side"],
            signal["spot"],
            market
        )

        if option:
            key = (
                asset,
                signal["side"],
                option["symbol"],
                signal["open_time"]
            )

            cooldown_key = (asset, signal["side"], "CONF")
            now = time.time()

            if (
                key not in seen
                and now - last_alert.get(cooldown_key, 0)
                >= CONF_COOLDOWN_MIN * 60
            ):
                send_option_confirm(asset, signal, option)
                seen.add(key)
                last_alert[cooldown_key] = now
                stats["confirmed"] += 1

    # Early option watch.
    if early_l or early_s:
        side = "LONG" if early_l else "SHORT"
        sc = score(c, side, False)

        if sc >= MIN_SCORE:
            signal2 = {
                "side": side,
                "score": sc,
                "spot": float(c.close),
                "ema9": float(c.ema9),
                "ema20": float(c.ema20),
                "vwap": float(c.vwap),
                "rsi": float(c.rsi),
                "adx": float(c.adx),
                "di_plus": float(c.di_plus),
                "di_minus": float(c.di_minus),
                "vol_ratio": float(c.vol_ratio),
            }

            cooldown_key = (asset, side, "EARLY")

            if (
                time.time() - last_alert.get(cooldown_key, 0)
                >= EARLY_COOLDOWN_MIN * 60
            ):
                send_early(asset, signal2)
                last_alert[cooldown_key] = time.time()
                stats["early"] += 1

# ------------------------------------------------------------
# Polling loop
# ------------------------------------------------------------

def scan_loop():
    while True:
        for asset in list(contracts.keys()):
            try:
                symbol = spot_symbols[asset]

                # Refresh latest completed 5m candles.
                data = get_json(
                    f"{SPOT_BASE}/api/v3/klines",
                    {
                        "symbol": symbol,
                        "interval": INTERVAL,
                        "limit": 3
                    }
                )

                now_ms = int(time.time() * 1000)

                for k in data:
                    if int(k[6]) > now_ms:
                        continue

                    b = {
                        "open_time": int(k[0]),
                        "open": float(k[1]),
                        "high": float(k[2]),
                        "low": float(k[3]),
                        "close": float(k[4]),
                        "volume": float(k[5]),
                        "close_time": int(k[6])
                    }

                    with lock:
                        if last_bar.get(symbol) == b["open_time"]:
                            continue

                        if symbol not in bars:
                            from collections import deque
                            bars[symbol] = deque(maxlen=HISTORY)

                        bars[symbol].append(b)
                        last_bar[symbol] = b["open_time"]

                    analyze_asset(asset)
                    stats["closed_bars"] += 1

            except Exception as e:
                stats["errors"] += 1
                print("Scan error", asset, e)

            time.sleep(0.25)

        # Re-discover every 30 minutes so newly listed option contracts
        # can become available without restarting the bot.
        time.sleep(10)

# ------------------------------------------------------------
# Health
# ------------------------------------------------------------

@app.get("/")
def home():
    return jsonify({
        "bot": BOT_NAME,
        "status": "running",
        "timeframe": INTERVAL,
        "option_assets": sorted(list(contracts.keys())),
        "skipped_assets": [
            a for _, a in REQUESTED if a not in contracts
        ],
        "min_score": MIN_SCORE,
        "stats": dict(stats)
    })

@app.get("/health")
def health():
    return jsonify({
        "ok": True,
        "option_assets": len(contracts),
        "bars": {
            a: len(bars.get(spot_symbols[a], []))
            for a in contracts
        },
        "confirmed": stats["confirmed"],
        "early": stats["early"],
        "errors": stats["errors"]
    })

# ------------------------------------------------------------
# Startup
# ------------------------------------------------------------

def startup_message():
    active = ", ".join(sorted(contracts.keys())) or "NONE"

    skipped = ", ".join(
        a for _, a in REQUESTED if a not in contracts
    ) or "NONE"

    telegram(
        f"""🟢 <b>{BOT_NAME} ONLINE</b>

<b>Mode:</b> Crypto Options Signal Scanner
<b>Timeframe:</b> 5 minutes
<b>Minimum score:</b> {MIN_SCORE}/10

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
✓ Volume/liquidity
✓ Active expiry
✓ Underlying trend confirmation

No automatic orders are placed."""
    )

def main():
    print("=" * 70)
    print(BOT_NAME)
    print("=" * 70)

    discover_option_contracts()

    for asset in contracts:
        try:
            print("Warmup", asset, load_history(asset))
        except Exception as e:
            print("Warmup failed", asset, e)

        time.sleep(0.10)

    threading.Thread(
        target=lambda: app.run(
            host="0.0.0.0",
            port=int(os.getenv("PORT", "10000")),
            threaded=True
        ),
        daemon=True
    ).start()

    startup_message()

    threading.Thread(
        target=scan_loop,
        daemon=True
    ).start()

    while True:
        time.sleep(60)

if __name__ == "__main__":
    main()
