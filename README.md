# Crypto Options Telegram Signal Bot

This version is ONLY for crypto options. It does not contain NSE/Dhan/F&O stock logic.

Requested assets:
BTC ETH SOL XRP DOGE ADA AVAX TRX BNB NEAR AAVE LIT ENA ZEC AKE ESPORTS UNI LINK LTC MATIC

The bot dynamically reads Binance Options exchange information. If Binance does not currently list active options for an asset, that asset is skipped automatically. This is important because an altcoin being tradable on Binance Spot does not mean an option market exists for it.

## Strategy

5-minute underlying:
- EMA 9
- EMA 20
- Session/day VWAP
- RSI 14
- ADX 14
- DI+ / DI-
- Volume vs 20-candle average
- ATR 14
- Candle body strength
- EMA separation

Option selection:
- Active expiry
- CE for bullish underlying / PE for bearish underlying
- Strike distance <= 3.5%
- Delta approximately 0.35 to 0.65
- Bid/ask spread <= 2.5%
- Minimum option volume
- Near-ATM preference
- Highest liquidity/selection score

Trade plan:
- Entry = current option LTP
- Entry zone = +/- 2%
- SL = -25% premium
- T1 = +40%
- T2 = +70%
- Move SL to breakeven after T1

## Telegram
Create a Telegram bot with BotFather and put:
TELEGRAM_BOT_TOKEN
TELEGRAM_CHAT_ID

## Binance
Public market-data endpoints are used for scanning. No Binance API key is required for this signal-only market-data version.

If you later want automatic order execution, API credentials and a separate signed-order engine are required. Do not put withdrawal permission on an API key.

## Render
Build:
pip install -r requirements.txt

Start:
python main.py

Health:
https://YOUR-RENDER-URL/health

## Important
This bot sends signals, not guaranteed profits. Crypto options have significant IV, theta, liquidity, spread and gap risk. Use predefined risk.
