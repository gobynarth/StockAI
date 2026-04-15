"""
Live signal generator: loads all 3 Kronos models, runs today's data,
reports consensus signals (all 3 agree = high confidence trade).
Tracks paper trades and real outcomes for ongoing validation.
Sends daily email briefing via Gmail SMTP.

Active tickers: RIVN, ENVX, TSLA, BITF, PATH, HON
Monitor tickers: none

Regime filters:
  - Skip ENVX longs when VIX < 15 (long WR drops to 27%)
  - Skip ENVX within 14 days of earnings (long WR drops to 41%)
  - Skip BITF longs when VIX < 15 (long WR drops to 22%)

Usage: python live_signal.py
Email: set env var GMAIL_APP_PASSWORD=xxxx-xxxx-xxxx-xxxx
Portfolio: set env var PORTFOLIO_SIZE=10000  (optional, for $ amounts in email)
"""
import os, sys, json, urllib.request, csv, smtplib, time, asyncio, html
asyncio.set_event_loop(asyncio.new_event_loop())
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import timedelta, datetime
from ib_insync import IB, Stock, MarketOrder, LimitOrder, StopOrder, Order
from env_paths import add_kronos_to_path, base_path

add_kronos_to_path()
from model import Kronos, KronosTokenizer, KronosPredictor

# ── Active tickers (paper trade + position sized) ──────────────────────────
ACTIVE = {
    "RIVN": {"horizon": 40, "temp": 1.0, "lookback": 200, "tier": "A", "oos_acc": 75.0,
             "alloc": 0.04, "tp": 0.15, "sl": 0.02,
             "trailing_pct": 0.03},  # TRAIL_3: Sharpe 0.47 vs 0.30 fixed
    "ENVX": {"horizon": 90, "temp": 1.0, "lookback": 200, "tier": "A", "oos_acc": 69.5,
             "alloc": 0.03, "tp": 0.15, "sl": 0.02,
             "trailing_pct": 0.03,  # TRAIL_3: Sharpe 0.47 vs 0.06 fixed
             "skip_low_vix": True, "skip_near_earnings": True},
    "TSLA": {"horizon": 90, "temp": 0.5, "lookback": 400, "tier": "B", "oos_acc": 57.2,
             "alloc": 0.028, "tp": 0.20, "sl": 0.02},
             # TSLA: FIXED wins (Sharpe 0.29 vs 0.13 trailing). Keep fixed TP/SL.
    "BITF": {"horizon": 60, "temp": 1.0, "lookback": 200, "tier": "A", "oos_acc": 83.0,
             "alloc": 0.03, "tp": 0.25, "sl": 0.02,
             "trailing_pct": 0.03,  # TRAIL_3 wins (screener survivor, Sharpe 0.73)
             "skip_low_vix": True},  # low VIX long_wr drops 21.7% vs 45.6%
    "PATH": {"horizon": 60, "temp": 1.0, "lookback": 200, "tier": "A", "oos_acc": 88.5,
             "alloc": 0.02, "tp": 0.20, "sl": 0.10},
             # Ensemble agree 88.5%. FIXED wins (Sharpe 1.25). No filters needed.
    "HON":  {"horizon": 60, "temp": 1.0, "lookback": 200, "tier": "A", "oos_acc": 74.0,
             "alloc": 0.02, "tp": 0.15, "sl": 0.10},
             # Ensemble agree 74.0%. FIXED wins (Sharpe 2.24, WR 81.1%). No filters needed.
}

# No monitors — active only
MONITOR = {}
WATCHLIST = ACTIVE
EXIT_RULES = {}

CRYPTO = {"BTC", "SOL", "TAO", "ETH", "DOGE", "XRP"}
BASE = base_path()
PAPER_TRADE_LOG = os.path.join(BASE, "paper_trades.csv")
SIGNAL_HISTORY = os.path.join(BASE, "signal_history.csv")

MODELS = [
    {"name": "mini",  "model_id": "NeoQuasar/Kronos-mini",  "tok_id": "NeoQuasar/Kronos-Tokenizer-2k",   "max_ctx": 2048},
    {"name": "small", "model_id": "NeoQuasar/Kronos-small", "tok_id": "NeoQuasar/Kronos-Tokenizer-base", "max_ctx": 512},
    {"name": "base",  "model_id": "NeoQuasar/Kronos-base",  "tok_id": "NeoQuasar/Kronos-Tokenizer-base", "max_ctx": 512},
]


def fmt_pct(value, digits=1):
    if pd.isna(value) or value == "":
        return "-"
    return f"{float(value):+.{digits}f}%"


def fmt_money(value):
    if pd.isna(value) or value == "":
        return "-"
    return f"${float(value):,.2f}"


def fmt_date(value):
    if pd.isna(value) or value == "":
        return "-"
    return str(value)


def download_daily_data(yf_ticker):
    raw = yf.download(yf_ticker, period="3y", interval="1d",
                      auto_adjust=True, progress=False, timeout=30)
    raw = raw.reset_index()
    raw.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in raw.columns]
    date_col = "date" if "date" in raw.columns else "datetime"
    raw = raw.rename(columns={date_col: "timestamps"})
    raw["amount"] = raw["close"] * raw["volume"]
    raw["timestamps"] = pd.to_datetime(raw["timestamps"]).dt.tz_localize(None)
    return raw[["timestamps", "open", "high", "low", "close", "volume", "amount"]].dropna()


def load_data(ticker, lookback):
    yf_ticker = f"{ticker}-USD" if ticker in CRYPTO else ticker
    csv_path = os.path.join(BASE, "data", f"{ticker}.csv")
    raw = None
    if os.path.exists(csv_path):
        raw = pd.read_csv(csv_path, parse_dates=["timestamps"])
        raw["timestamps"] = pd.to_datetime(raw["timestamps"]).dt.tz_localize(None)
    if raw is None or raw.empty:
        raw = download_daily_data(yf_ticker)
    else:
        latest = pd.to_datetime(raw["timestamps"]).max().normalize()
        today = pd.Timestamp.now().normalize()
        if latest < today:
            try:
                fresh = download_daily_data(yf_ticker)
                if not fresh.empty and pd.to_datetime(fresh["timestamps"]).max() >= pd.to_datetime(raw["timestamps"]).max():
                    raw = fresh
                    fresh.to_csv(csv_path, index=False)
            except Exception as e:
                print(f"  {ticker}: refresh failed ({e}), using cached data")
    raw = raw[["timestamps", "open", "high", "low", "close", "volume", "amount"]].dropna()
    return raw


def get_prediction(predictor, raw, params, model_max_ctx):
    lb = min(params["lookback"], model_max_ctx)
    horizon = params["horizon"]
    x_df = raw.iloc[-lb:][["open", "high", "low", "close", "volume", "amount"]].reset_index(drop=True)
    x_ts = raw.iloc[-lb:]["timestamps"].reset_index(drop=True)
    entry_close = raw.iloc[-1]["close"]
    entry_date = raw.iloc[-1]["timestamps"]
    y_dates = pd.bdate_range(start=entry_date + timedelta(days=1), periods=horizon)
    y_ts = pd.Series(y_dates)
    pred = predictor.predict(df=x_df, x_timestamp=x_ts, y_timestamp=y_ts,
                             pred_len=horizon, T=params["temp"], top_p=0.9,
                             sample_count=1, verbose=False)
    pred_close = pred["close"].iloc[-1]
    direction = "UP" if pred_close > entry_close else "DOWN"
    pct = (pred_close - entry_close) / entry_close * 100
    return direction, pct, entry_close, pred_close, y_dates[-1]


def get_vix():
    """Get current VIX level."""
    try:
        vix = yf.download("^VIX", period="5d", interval="1d",
                          auto_adjust=True, progress=False, timeout=15)
        vix.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in vix.columns]
        return float(vix["close"].iloc[-1])
    except Exception as e:
        print(f"  VIX fetch failed ({e}), defaulting to 20 (no filter)")
        return 20.0


def get_earnings_dates(ticker):
    """Get known earnings dates for a ticker."""
    try:
        t = yf.Ticker(ticker)
        hist = t.earnings_dates
        if hist is not None and not hist.empty:
            return [pd.to_datetime(d).tz_localize(None).normalize() for d in hist.index]
    except Exception:
        pass
    return []


def near_earnings(ticker, window_days=14):
    """True if today is within window_days of an earnings date."""
    today = pd.Timestamp.now().normalize()
    for ed in get_earnings_dates(ticker):
        if abs((today - ed).days) <= window_days:
            return True
    return False


def check_open_paper_trades(ohlc_cache):
    """Check if any open paper trades have hit TP/SL."""
    if not os.path.exists(PAPER_TRADE_LOG):
        return []
    df = pd.read_csv(PAPER_TRADE_LOG)
    closed = []
    updates = []
    for _, row in df.iterrows():
        if row["status"] != "OPEN":
            updates.append(row.to_dict())
            continue
        ticker = row["ticker"]
        if ticker not in ohlc_cache:
            try:
                ohlc_cache[ticker] = yf.download(
                    ticker, period="6mo", interval="1d",
                    auto_adjust=True, progress=False, timeout=30)
            except Exception:
                updates.append(row.to_dict())
                continue
        ohlc = ohlc_cache[ticker]
        if ohlc.empty:
            updates.append(row.to_dict())
            continue
        ohlc.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in ohlc.columns]
        entry_price = row["entry_price"]
        tp = row["tp_pct"]
        sl = row["sl_pct"]
        direction = row["direction"]
        entry_date = pd.to_datetime(row["entry_date"])
        target_date = pd.to_datetime(row["target_date"])
        try:
            trailing_pct = float(row.get("trailing_pct", 0) or 0)
        except (ValueError, TypeError):
            trailing_pct = 0.0

        ohlc.index = pd.to_datetime(ohlc.index).tz_localize(None) if ohlc.index.tz else pd.to_datetime(ohlc.index)
        future = ohlc.loc[entry_date:]
        if len(future) <= 1:
            updates.append(row.to_dict())
            continue
        future = future.iloc[1:]

        exit_reason = None
        exit_price = None
        exit_date = None
        peak = entry_price
        current_sl = entry_price * (1 - sl) if direction == "UP" else entry_price * (1 + sl)

        for date, bar in future.iterrows():
            high  = bar.get("high",  bar["close"])
            low   = bar.get("low",   bar["close"])
            close = bar["close"]

            if direction == "UP":
                if high > peak:
                    peak = high
                if trailing_pct > 0:
                    candidate = peak * (1 - trailing_pct)
                    if candidate > current_sl:
                        current_sl = candidate
                if low <= current_sl:
                    reason = "TRAIL" if trailing_pct > 0 else "SL"
                    exit_reason, exit_price, exit_date = reason, current_sl, date
                    break
                if trailing_pct == 0 and high >= entry_price * (1 + tp):
                    exit_reason, exit_price, exit_date = "TP", entry_price * (1 + tp), date
                    break
            else:  # SHORT
                if low < peak or peak == entry_price:
                    peak = low if peak == entry_price else min(peak, low)
                if trailing_pct > 0:
                    candidate = peak * (1 + trailing_pct)
                    if candidate < current_sl:
                        current_sl = candidate
                if high >= current_sl:
                    reason = "TRAIL" if trailing_pct > 0 else "SL"
                    exit_reason, exit_price, exit_date = reason, current_sl, date
                    break
                if trailing_pct == 0 and low <= entry_price * (1 - tp):
                    exit_reason, exit_price, exit_date = "TP", entry_price * (1 - tp), date
                    break

            if date >= target_date:
                exit_reason, exit_price, exit_date = "EXPIRY", close, date
                break

        if exit_reason:
            r = row.to_dict()
            r["status"] = "CLOSED"
            r["exit_reason"] = exit_reason
            r["exit_price"] = round(float(exit_price), 2)
            r["exit_date"] = str(exit_date.date()) if hasattr(exit_date, 'date') else str(exit_date)
            if direction == "UP":
                r["pnl_pct"] = round((float(exit_price) - entry_price) / entry_price * 100, 2)
            else:
                r["pnl_pct"] = round((entry_price - float(exit_price)) / entry_price * 100, 2)
            updates.append(r)
            closed.append(r)
        else:
            updates.append(row.to_dict())

    if closed:
        pd.DataFrame(updates).to_csv(PAPER_TRADE_LOG, index=False)
    return closed


def open_paper_trade(ticker, direction, entry_price, entry_date, target_date, horizon, alloc):
    """Log a new paper trade with position sizing."""
    cfg = ACTIVE.get(ticker, {})
    tp = cfg.get("tp", 0.15)
    sl = cfg.get("sl", 0.02)
    trailing_pct = cfg.get("trailing_pct", 0)
    trade = {
        "ticker": ticker,
        "direction": direction,
        "entry_price": round(float(entry_price), 2),
        "entry_date": str(entry_date.date()) if hasattr(entry_date, 'date') else str(entry_date),
        "target_date": str(target_date.date()) if hasattr(target_date, 'date') else str(target_date),
        "trailing_pct": trailing_pct,
        "horizon": horizon,
        "tp_pct": tp,
        "sl_pct": sl,
        "alloc_pct": alloc,
        "status": "OPEN",
        "exit_reason": "",
        "exit_price": "",
        "exit_date": "",
        "pnl_pct": "",
    }
    file_exists = os.path.exists(PAPER_TRADE_LOG)
    with open(PAPER_TRADE_LOG, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=trade.keys())
        if not file_exists:
            w.writeheader()
        w.writerow(trade)
    return trade


def load_open_trade_snapshots(open_trades, ohlc_cache):
    """Attach latest close and unrealized P/L for open trades."""
    snapshots = []
    for _, trade in open_trades.iterrows():
        ticker = trade["ticker"]
        if ticker not in ohlc_cache:
            yf_ticker = f"{ticker}-USD" if ticker in CRYPTO else ticker
            try:
                raw = yf.download(yf_ticker, period="3mo", interval="1d",
                                  auto_adjust=True, progress=False, timeout=30)
                if not raw.empty:
                    raw.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in raw.columns]
                ohlc_cache[ticker] = raw
            except Exception:
                ohlc_cache[ticker] = pd.DataFrame()

        latest_close = np.nan
        if ticker in ohlc_cache and not ohlc_cache[ticker].empty:
            latest_close = float(ohlc_cache[ticker]["close"].iloc[-1])

        entry_price = float(trade["entry_price"])
        direction = trade["direction"]
        tp_price = entry_price * (1 + float(trade["tp_pct"])) if direction == "UP" else entry_price * (1 - float(trade["tp_pct"]))
        sl_price = entry_price * (1 - float(trade["sl_pct"])) if direction == "UP" else entry_price * (1 + float(trade["sl_pct"]))
        trailing_pct = float(trade.get("trailing_pct", 0) or 0)

        unrealized_pct = np.nan
        if not pd.isna(latest_close):
            if direction == "UP":
                unrealized_pct = (latest_close - entry_price) / entry_price * 100
            else:
                unrealized_pct = (entry_price - latest_close) / entry_price * 100

        snapshots.append({
            "ticker": ticker,
            "direction": direction,
            "entry_price": entry_price,
            "entry_date": trade["entry_date"],
            "target_date": trade["target_date"],
            "alloc_pct": float(trade.get("alloc_pct", 0) or 0),
            "tp_price": tp_price,
            "sl_price": sl_price,
            "trailing_pct": trailing_pct,
            "latest_close": latest_close,
            "unrealized_pct": unrealized_pct,
        })
    return snapshots


def build_email_html(today, vix_level, vix_regime, action_rows, skipped_regime,
                     closed_trade_rows, open_trade_rows, scorecard, ib_status_lines):
    css = """
    body { font-family: Arial, sans-serif; color: #1f2937; margin: 0; background: #f5f7fb; }
    .wrap { max-width: 980px; margin: 0 auto; padding: 24px; }
    .hero { background: linear-gradient(135deg, #0f172a, #1d4ed8); color: white; padding: 20px 24px; border-radius: 16px; }
    .hero h1 { margin: 0 0 6px; font-size: 24px; }
    .hero p { margin: 4px 0; opacity: 0.92; }
    .section { background: white; border-radius: 16px; padding: 18px 20px; margin-top: 18px; box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08); }
    .section h2 { margin: 0 0 12px; font-size: 18px; }
    .pill { display: inline-block; padding: 4px 10px; border-radius: 999px; font-size: 12px; font-weight: 700; margin-right: 8px; }
    .buy { background: #dcfce7; color: #166534; }
    .sell { background: #fee2e2; color: #991b1b; }
    .skip { background: #e5e7eb; color: #374151; }
    .up { color: #166534; font-weight: 700; }
    .down { color: #b91c1c; font-weight: 700; }
    table { width: 100%; border-collapse: collapse; }
    th, td { text-align: left; padding: 10px 8px; border-bottom: 1px solid #e5e7eb; vertical-align: top; }
    th { font-size: 12px; text-transform: uppercase; letter-spacing: 0.04em; color: #6b7280; }
    .muted { color: #6b7280; }
    ul { margin: 0; padding-left: 18px; }
    """

    def action_badge(action):
        cls = "buy" if action == "BUY" else "sell" if action == "SELL/SHORT" else "skip"
        return f'<span class="pill {cls}">{html.escape(action)}</span>'

    active_rows_html = ""
    for row in action_rows:
        if row["latest_price"] == row["latest_price"]:
            px_class = "up" if row["price_move_pct"] >= 0 else "down"
            latest = f'<span class="{px_class}">{fmt_money(row["latest_price"])} ({fmt_pct(row["price_move_pct"])})</span>'
        else:
            latest = "-"
        active_rows_html += f"""
        <tr>
          <td><strong>{html.escape(row['ticker'])}</strong><br><span class="muted">{html.escape(row['tier'])} | {html.escape(row['consensus'])}</span></td>
          <td>{action_badge(row['action'])}<div class="muted">{html.escape(row['instruction'])}</div></td>
          <td>{fmt_money(row['entry_price'])}<br><span class="muted">{html.escape(row['origin'])}</span></td>
          <td>{latest}</td>
          <td>{html.escape(row['exit_rule'])}</td>
          <td>{fmt_date(row['target_date'])}<br><span class="muted">{row['alloc_pct']*100:.1f}% alloc</span></td>
        </tr>
        """
    if not active_rows_html:
        active_rows_html = '<tr><td colspan="6" class="muted">No active signals today.</td></tr>'

    open_rows_html = ""
    for row in open_trade_rows:
        latest = "-"
        pnl = "-"
        if row["latest_close"] == row["latest_close"]:
            pnl_class = "up" if row["unrealized_pct"] >= 0 else "down"
            latest = fmt_money(row["latest_close"])
            pnl = f'<span class="{pnl_class}">{fmt_pct(row["unrealized_pct"])}</span>'
        exit_rule = (f"{row['trailing_pct']*100:.0f}% trailing stop, initial SL {fmt_money(row['sl_price'])}"
                     if row["trailing_pct"] > 0 else
                     f"TP {fmt_money(row['tp_price'])} / SL {fmt_money(row['sl_price'])}")
        open_rows_html += f"""
        <tr>
          <td><strong>{html.escape(row['ticker'])}</strong><br><span class="muted">{html.escape(row['direction'])}</span></td>
          <td>{fmt_money(row['entry_price'])}</td>
          <td>{latest}</td>
          <td>{pnl}</td>
          <td>{html.escape(exit_rule)}</td>
          <td>{fmt_date(row['target_date'])}</td>
        </tr>
        """
    if not open_rows_html:
        open_rows_html = '<tr><td colspan="6" class="muted">No open paper trades.</td></tr>'

    closed_rows_html = ""
    for row in closed_trade_rows:
        pnl_class = "up" if float(row["pnl_pct"]) >= 0 else "down"
        closed_rows_html += f"""
        <tr>
          <td><strong>{html.escape(row['ticker'])}</strong><br><span class="muted">{html.escape(row['direction'])}</span></td>
          <td>{html.escape(str(row['exit_reason']))}</td>
          <td>{fmt_money(row['exit_price'])}</td>
          <td><span class="{pnl_class}">{fmt_pct(row['pnl_pct'])}</span></td>
          <td>{fmt_date(row['exit_date'])}</td>
        </tr>
        """
    if not closed_rows_html:
        closed_rows_html = '<tr><td colspan="5" class="muted">No trades closed today.</td></tr>'

    filters_html = "".join(f"<li>{html.escape(item)}</li>" for item in skipped_regime) if skipped_regime else "<li>No regime filters triggered today.</li>"
    ib_html = "".join(f"<li>{html.escape(item)}</li>" for item in ib_status_lines)

    scorecard_html = ""
    if scorecard:
        scorecard_html = (
            f"<p><strong>Closed trades:</strong> {scorecard['closed_count']} | "
            f"<strong>Win rate:</strong> {scorecard['win_rate']:.0f}% | "
            f"<strong>Total P&amp;L:</strong> {fmt_pct(scorecard['total_pnl'])}</p>"
        )

    return f"""<html><head><meta charset="utf-8"><style>{css}</style></head><body>
    <div class="wrap">
      <div class="hero">
        <h1>StockAI Daily Briefing</h1>
        <p>{html.escape(today)} | VIX {vix_level:.1f} ({html.escape(vix_regime)})</p>
        <p>Only trade when all 3 Kronos models agree. Follow the exit rules exactly.</p>
      </div>
      <div class="section">
        <h2>What To Do Today</h2>
        <table><thead><tr><th>Ticker</th><th>Action</th><th>Entry</th><th>Now</th><th>Exit Rule</th><th>Hold Until</th></tr></thead>
        <tbody>{active_rows_html}</tbody></table>
      </div>
      <div class="section">
        <h2>Open Paper Trades</h2>
        <table><thead><tr><th>Ticker</th><th>Entry</th><th>Latest</th><th>Unrealized</th><th>TP / SL</th><th>Target Date</th></tr></thead>
        <tbody>{open_rows_html}</tbody></table>
        {scorecard_html}
      </div>
      <div class="section">
        <h2>Closed Today</h2>
        <table><thead><tr><th>Ticker</th><th>Exit</th><th>Exit Price</th><th>P&amp;L</th><th>Date</th></tr></thead>
        <tbody>{closed_rows_html}</tbody></table>
      </div>
      <div class="section"><h2>Regime Filters</h2><ul>{filters_html}</ul></div>
      <div class="section"><h2>IBKR Paper Status</h2><ul>{ib_html}</ul></div>
    </div></body></html>"""


def send_discord(webhook_url, message):
    try:
        data = json.dumps({"content": message}).encode("utf-8")
        req = urllib.request.Request(webhook_url, data=data,
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10)
        print("Discord notification sent.")
    except Exception as e:
        print(f"Discord send failed: {e}")


def send_email(subject, body_text, body_html=""):
    """Send daily briefing email via Gmail SMTP."""
    app_pw = os.environ.get("GMAIL_APP_PASSWORD", "")
    if not app_pw:
        print("No GMAIL_APP_PASSWORD env var set -- skipping email.")
        return
    sender = "gobynarth@gmail.com"
    recipient = "gobynarth@gmail.com"
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = f"StockAI <{sender}>"
    msg["To"] = recipient
    msg.attach(MIMEText(body_text, "plain"))
    if body_html:
        msg.attach(MIMEText(body_html, "html"))
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender, app_pw)
            server.sendmail(sender, recipient, msg.as_string())
        print("Email sent.")
    except Exception as e:
        print(f"Email send failed: {e}")


# ── IBKR Integration ─────────────────────────────────────────────────────────
IB_HOST = os.environ.get("IB_HOST", "127.0.0.1")
IB_PORT = int(os.environ.get("IB_PORT", "4002"))  # IB Gateway paper default
IB_CLIENT_ID = int(os.environ.get("IB_CLIENT_ID", "1"))

def connect_ibkr():
    """Connect to IB Gateway. Returns IB instance or None if unavailable."""
    ib = IB()
    try:
        ib.connect(IB_HOST, IB_PORT, clientId=IB_CLIENT_ID, timeout=10)
        print(f"  IBKR connected on {IB_HOST}:{IB_PORT} (accounts: {ib.managedAccounts()})")
        return ib
    except Exception as e:
        print(f"  IBKR not available ({e}) -- orders will NOT be submitted")
        return None


def submit_ibkr_order(ib, ticker, direction, entry_price, shares, tp, sl, trailing_pct=0):
    """Submit a bracket order (or trailing stop order) to IBKR.
    direction: 'UP' = buy, 'DOWN' = sell short
    Returns list of placed order descriptions.
    """
    contract = Stock(ticker, "SMART", "USD")
    ib.qualifyContracts(contract)

    action = "BUY" if direction == "UP" else "SELL"
    reverse = "SELL" if direction == "UP" else "BUY"

    # Parent: market order to enter
    parent = MarketOrder(action, shares)
    parent.transmit = False  # don't send until children attached

    orders_placed = []

    # All orders GTC so they survive outside market hours
    parent.tif = "GTC"

    if trailing_pct > 0:
        # Trailing stop as the exit
        trail_pct = round(trailing_pct * 100, 1)
        trail_order = Order(
            action=reverse,
            totalQuantity=shares,
            orderType="TRAIL",
            trailingPercent=trail_pct,
        )
        trail_order.parentId = 0
        trail_order.tif = "GTC"
        trail_order.transmit = True

        parent.transmit = False
        trade_parent = ib.placeOrder(contract, parent)
        time.sleep(1)

        trail_order.parentId = trade_parent.order.orderId
        ib.placeOrder(contract, trail_order)
        orders_placed.append(f"{action} {shares} {ticker} MKT + {trail_pct}% trailing stop")

    else:
        # Bracket order: parent + TP limit + SL stop
        if direction == "UP":
            tp_price = round(entry_price * (1 + tp), 2)
            sl_price = round(entry_price * (1 - sl), 2)
        else:
            tp_price = round(entry_price * (1 - tp), 2)
            sl_price = round(entry_price * (1 + sl), 2)

        tp_order = LimitOrder(reverse, shares, tp_price)
        tp_order.parentId = 0
        tp_order.tif = "GTC"
        tp_order.transmit = False

        sl_order = StopOrder(reverse, shares, sl_price)
        sl_order.parentId = 0
        sl_order.tif = "GTC"
        sl_order.transmit = True

        parent.transmit = False
        trade_parent = ib.placeOrder(contract, parent)
        time.sleep(1)

        tp_order.parentId = trade_parent.order.orderId
        sl_order.parentId = trade_parent.order.orderId
        ib.placeOrder(contract, tp_order)
        ib.placeOrder(contract, sl_order)
        orders_placed.append(
            f"{action} {shares} {ticker} MKT | TP ${tp_price} | SL ${sl_price}")

    ib.sleep(2)  # let orders propagate
    return orders_placed


# ── Main ──────────────────────────────────────────────────────────────────────
today = datetime.now().strftime("%Y-%m-%d")
portfolio_size = float(os.environ.get("PORTFOLIO_SIZE", 0))

print(f"\nLOADING MODELS...")
predictors = {}
for m in MODELS:
    print(f"  Loading kronos-{m['name']}...")
    tok = KronosTokenizer.from_pretrained(m["tok_id"])
    mdl = Kronos.from_pretrained(m["model_id"])
    predictors[m["name"]] = KronosPredictor(mdl, tok, max_context=m["max_ctx"])

# ── Regime checks ─────────────────────────────────────────────────────────────
print(f"\nREGIME CHECKS...")
vix_level = get_vix()
print(f"  VIX: {vix_level:.1f}")
vix_regime = ("LOW" if vix_level < 15 else "MED" if vix_level < 25
              else "HIGH" if vix_level < 35 else "EXTREME")
print(f"  Regime: {vix_regime}")

envx_near_earnings = near_earnings("ENVX", window_days=14)
if envx_near_earnings:
    print(f"  ENVX: within 14 days of earnings -- FILTERING")
else:
    print(f"  ENVX: clear of earnings")

# ── Connect to IBKR ──────────────────────────────────────────────────────────
print(f"\nIBKR CONNECTION...")
ib = connect_ibkr()
ibkr_orders = []
if ib:
    ib_status_lines = [f"Connected to {IB_HOST}:{IB_PORT}",
                       f"Accounts: {', '.join(ib.managedAccounts()) or 'none reported'}"]
else:
    ib_status_lines = [f"Not connected on {IB_HOST}:{IB_PORT}",
                       "No paper orders were submitted."]

# ── Check open paper trades ───────────────────────────────────────────────────
print(f"\nCHECKING OPEN PAPER TRADES...")
ohlc_cache = {}
closed_trades = check_open_paper_trades(ohlc_cache)
if closed_trades:
    print(f"  {len(closed_trades)} trades closed:")
    for t in closed_trades:
        print(f"    {t['ticker']} {t['direction']}: {t['exit_reason']} at ${t['exit_price']} ({t['pnl_pct']:+.1f}%)")
else:
    print("  No trades closed today.")

open_tickers = set()
if os.path.exists(PAPER_TRADE_LOG):
    ptdf = pd.read_csv(PAPER_TRADE_LOG)
    open_tickers = set(ptdf[ptdf["status"] == "OPEN"]["ticker"].values)
closed_today_tickers = {
    t["ticker"] for t in closed_trades
    if str(t.get("exit_date", "")).startswith(today)
}

# ── Generate signals ──────────────────────────────────────────────────────────
print(f"\n{'='*80}")
print(f"DAILY SIGNALS -- {today}  (VIX: {vix_level:.1f} [{vix_regime}])")
print(f"{'='*80}")
print(f"{'Ticker':<7} {'Tier':>4} {'mini':>6} {'small':>6} {'base':>6} {'Consensus':>11} {'Action':>10} {'Filter':>12}")
print(f"{'-'*75}")

results = []
action_lines = []
action_rows = []
new_paper_trades = []
skipped_regime = []

# ── Load signal history for staleness tracking ───────────────────────────────
signal_history = {}
if os.path.exists(SIGNAL_HISTORY):
    hist_df = pd.read_csv(SIGNAL_HISTORY, parse_dates=["date"])
    hist_df = hist_df.sort_values("date")
    for tk in hist_df["ticker"].unique():
        signal_history[tk] = hist_df[hist_df["ticker"] == tk]


def get_signal_age(ticker, current_consensus):
    """Count consecutive days this consensus has been firing.
    Returns (days_in_a_row, first_fire_date, first_fire_price).
    Day 1 = first time today, Day 2 = also fired yesterday, etc.
    """
    if ticker not in signal_history or current_consensus == "MIXED":
        return 1, None, None
    h = signal_history[ticker].sort_values("date", ascending=False)
    days = 1
    first_date = None
    first_price = None
    for _, row in h.iterrows():
        if row["consensus"] == current_consensus:
            days += 1
            first_date = row["date"]
            first_price = row.get("entry", None)
        else:
            break
    return days, first_date, first_price

for ticker, params in WATCHLIST.items():
    raw = load_data(ticker, params["lookback"])
    dirs = {}
    pcts = {}
    entry_close = None
    target_date = None
    for m in MODELS:
        direction, pct, ec, pc, td = get_prediction(
            predictors[m["name"]], raw, params, m["max_ctx"]
        )
        dirs[m["name"]] = direction
        pcts[m["name"]] = pct
        entry_close = ec
        target_date = td

    all_up = all(d == "UP" for d in dirs.values())
    all_down = all(d == "DOWN" for d in dirs.values())
    is_inverse = params.get("inverse", False)
    if all_up:
        consensus = "AGREE UP"
        action = "SELL/SHORT" if is_inverse else "BUY"
    elif all_down:
        consensus = "AGREE DOWN"
        action = "BUY" if is_inverse else "SELL/SHORT"
    else:
        consensus, action = "MIXED", "SKIP"

    # ── Regime filtering (active tickers only) ────────────────────────────
    is_active = ticker in ACTIVE
    filter_reason = ""

    if is_active and action in ("BUY", "SELL/SHORT"):
        cfg = ACTIVE[ticker]
        if action == "SELL/SHORT" and not cfg.get("allow_short", False):
            filter_reason = "SHORT DISABLED"
            skipped_regime.append(f"{ticker}: skipped short (not validated live)")
            action = "SKIP"
        # VIX filter — applies to longs AND shorts (low VIX = no motion in either direction)
        if action != "SKIP" and cfg.get("skip_low_vix") and vix_level < 15:
            filter_reason = "VIX<15 SKIP"
            skipped_regime.append(f"{ticker}: skipped {action} (VIX {vix_level:.1f} < 15)")
            action = "SKIP"
        # Earnings filter
        elif action != "SKIP" and cfg.get("skip_near_earnings") and ticker == "ENVX" and envx_near_earnings:
            filter_reason = "EARNINGS SKIP"
            skipped_regime.append(f"{ticker}: skipped {action} (near earnings)")
            action = "SKIP"

    if not is_active and action != "SKIP":
        filter_reason = "MONITOR"

    avg_pct = np.mean(list(pcts.values()))
    tp = params.get("tp", EXIT_RULES.get(ticker, {}).get("tp", 0.15))
    sl = params.get("sl", EXIT_RULES.get(ticker, {}).get("sl", 0.02))
    tp_sl = f"TP{tp*100:.0f}/SL{sl*100:.0f}"

    print(f"{ticker:<7} {params['tier']:>4} {dirs['mini']:>6} {dirs['small']:>6} {dirs['base']:>6} "
          f"{consensus:>11} {action:>10} {filter_reason:>12}")

    # ── Paper trade (active tickers only) ─────────────────────────────────
    if is_active and action != "SKIP" and ticker not in open_tickers and ticker not in closed_today_tickers:
        cfg = ACTIVE[ticker]
        if is_inverse:
            trade_dir = "DOWN" if all_up else "UP"
        else:
            trade_dir = "UP" if all_up else "DOWN"
        alloc = cfg["alloc"]
        t = open_paper_trade(ticker, trade_dir, entry_close,
                             raw.iloc[-1]["timestamps"], target_date,
                             params["horizon"], alloc)
        new_paper_trades.append(t)
        dollar_str = ""
        if portfolio_size > 0:
            dollar_amt = portfolio_size * alloc
            dollar_str = f" (${dollar_amt:,.0f})"
        tp_price = entry_close * (1 + tp)
        sl_price = entry_close * (1 - sl)
        print(f"         >> PAPER TRADE: {trade_dir} @ ${entry_close:.2f}, "
              f"{alloc*100:.1f}% alloc{dollar_str}")
        print(f"         >> TP=${tp_price:.2f} (+{tp*100:.0f}%)  SL=${sl_price:.2f} (-{sl*100:.0f}%)")

        # ── Submit to IBKR ───────────────────────────────────────────────
        if ib and portfolio_size > 0:
            shares = int(portfolio_size * alloc / entry_close)
            if shares >= 1:
                try:
                    placed = submit_ibkr_order(
                        ib, ticker, trade_dir, entry_close, shares,
                        cfg["tp"], cfg["sl"], cfg.get("trailing_pct", 0))
                    for desc in placed:
                        print(f"         >> IBKR: {desc}")
                        ibkr_orders.append(desc)
                except Exception as e:
                    print(f"         >> IBKR ORDER FAILED: {e}")
                    ibkr_orders.append(f"FAILED {ticker}: {e}")

    elif is_active and action != "SKIP" and ticker in open_tickers:
        print(f"         >> Already in open trade, skipping")
    elif is_active and action != "SKIP" and ticker in closed_today_tickers:
        print(f"         >> Closed earlier today, skipping same-day re-entry")

    if action != "SKIP" and is_active:
        alloc = ACTIVE[ticker]["alloc"]
        trailing_pct = ACTIVE[ticker].get("trailing_pct", 0)
        is_short = (action == "SELL/SHORT")

        # Direction-aware TP/SL prices
        if is_short:
            tp_price = entry_close * (1 - tp)   # short TP = price drops
            sl_price = entry_close * (1 + sl)   # short SL = price rises
        else:
            tp_price = entry_close * (1 + tp)
            sl_price = entry_close * (1 - sl)

        # ── First fire date + price ─────────────────────────────────────
        days_old, first_date, first_price = get_signal_age(ticker, consensus)
        if days_old == 1 or first_date is None:
            origin = "first fired today"
        else:
            origin = f"first fired {first_date.strftime('%Y-%m-%d')} @ ${first_price:.2f}"

        # Build exit instruction based on trailing vs fixed
        if trailing_pct > 0:
            if is_short:
                exit_line = f"Set {trailing_pct*100:.0f}% trailing BUY-STOP (broker auto-covers as price falls)"
            else:
                exit_line = f"Set {trailing_pct*100:.0f}% trailing SELL stop (broker auto-adjusts as price rises)"
            exit_line += f" | initial SL ${sl_price:.2f}"
        else:
            verb = "Cover/buy-back" if is_short else "Sell"
            exit_line = f"{verb} if hits ${tp_price:.2f} (TP) or ${sl_price:.2f} (SL)"

        line = (f"{action} {ticker} [{params['tier']}] @ ${entry_close:.2f}  ({origin})\n"
                f"    {exit_line}\n"
                f"    Hold until {target_date.strftime('%Y-%m-%d')} ({params['horizon']}d) | {alloc*100:.1f}% alloc")
        if portfolio_size > 0:
            dollar_amt = portfolio_size * alloc
            line += f" = ${dollar_amt:,.0f}"
        action_lines.append(line)
        latest_price = float(raw.iloc[-1]["close"])
        if not is_short:
            price_move_pct = (latest_price - entry_close) / entry_close * 100
            instruction = "Go long"
        else:
            price_move_pct = (entry_close - latest_price) / entry_close * 100
            instruction = "Open short"
        action_rows.append({
            "ticker": ticker,
            "tier": params["tier"],
            "consensus": consensus,
            "action": action,
            "instruction": instruction,
            "entry_price": float(entry_close),
            "latest_price": latest_price,
            "price_move_pct": price_move_pct,
            "origin": origin,
            "exit_rule": exit_line,
            "target_date": target_date.strftime("%Y-%m-%d"),
            "alloc_pct": alloc,
        })

    results.append({
        "date": today, "ticker": ticker, "tier": params["tier"],
        "mini": dirs["mini"], "small": dirs["small"], "base": dirs["base"],
        "consensus": consensus, "action": action,
        "avg_pct": round(avg_pct, 2), "horizon": params["horizon"],
        "target_date": target_date.strftime("%Y-%m-%d"),
        "entry": round(float(entry_close), 2),
        "tp": tp, "sl": sl,
        "filter": filter_reason,
    })

# ── Regime filter summary ─────────────────────────────────────────────────────
if skipped_regime:
    print(f"\nREGIME FILTERS APPLIED:")
    for s in skipped_regime:
        print(f"  {s}")

# ── Paper trade summary ───────────────────────────────────────────────────────
if os.path.exists(PAPER_TRADE_LOG):
    ptdf = pd.read_csv(PAPER_TRADE_LOG)
    open_trades = ptdf[ptdf["status"] == "OPEN"]
    closed_trades_all = ptdf[ptdf["status"] == "CLOSED"]
    print(f"\n{'='*80}")
    print(f"PAPER PORTFOLIO")
    print(f"{'='*80}")
    print(f"  Open trades: {len(open_trades)}")
    for _, t in open_trades.iterrows():
        alloc_str = f" | {t['alloc_pct']*100:.1f}%" if "alloc_pct" in t and pd.notna(t.get("alloc_pct")) else ""
        print(f"    {t['ticker']} {t['direction']} @ ${t['entry_price']:.2f} "
              f"(entered {t['entry_date']}, target {t['target_date']}){alloc_str}")
    if len(closed_trades_all) > 0:
        wins = closed_trades_all[closed_trades_all["pnl_pct"] > 0]
        total_pnl = closed_trades_all["pnl_pct"].sum()
        win_rate = len(wins) / len(closed_trades_all) * 100
        print(f"  Closed trades: {len(closed_trades_all)} | Win rate: {win_rate:.0f}% | Total P&L: {total_pnl:+.1f}%")

print(f"\n{'='*80}")
print("Active: RIVN 4% | ENVX 3% | TSLA 2.8% | BITF 3% | PATH 2% | HON 2%")
print("Ensemble filter: only trade when all 3 models agree")
print(f"{'='*80}")

# ── Save signal history ───────────────────────────────────────────────────────
hist_cols = list(results[0].keys())
new_hist = pd.DataFrame(results)
if os.path.exists(SIGNAL_HISTORY):
    old_hist = pd.read_csv(SIGNAL_HISTORY)
    hist = pd.concat([old_hist, new_hist], ignore_index=True)
    hist = hist.drop_duplicates(subset=["date", "ticker"], keep="last")
else:
    hist = new_hist
hist = hist[hist_cols]
hist.to_csv(SIGNAL_HISTORY, index=False)

out = os.path.join(BASE, f"live_signals_{today.replace('-', '')}.csv")
pd.DataFrame(results).to_csv(out, index=False)
print(f"\nSaved: {out}")

# ── Build email ───────────────────────────────────────────────────────────────
email_lines = []
email_lines.append(f"STOCKAI DAILY BRIEFING -- {today}")
email_lines.append("=" * 50)
email_lines.append(f"\nVIX: {vix_level:.1f} ({vix_regime})")

# Active signals
email_lines.append(f"\nACTIVE SIGNALS:")
if action_lines:
    for line in action_lines:
        email_lines.append(f"  {line}")
else:
    email_lines.append("  No active signals today.")

# Regime filters
if skipped_regime:
    email_lines.append(f"\nREGIME FILTERS:")
    for s in skipped_regime:
        email_lines.append(f"  {s}")

# Monitor signals
mon_lines = [r for r in results if r["tier"] in ("MON", "INV") and r["consensus"] != "MIXED"]
if mon_lines:
    email_lines.append(f"\nMONITOR (not trading):")
    for r in mon_lines:
        email_lines.append(f"  {r['ticker']}: {r['consensus']} ({r['avg_pct']:+.1f}%)")

# Closed trades
if closed_trades:
    email_lines.append(f"\nCLOSED TRADES:")
    for t in closed_trades:
        email_lines.append(
            f"  {t['ticker']} {t['direction']} -> {t['exit_reason']} "
            f"@ ${t['exit_price']} ({t['pnl_pct']:+.1f}%)")

# Paper portfolio
if os.path.exists(PAPER_TRADE_LOG):
    ptdf = pd.read_csv(PAPER_TRADE_LOG)
    open_tr = ptdf[ptdf["status"] == "OPEN"]
    closed_tr = ptdf[ptdf["status"] == "CLOSED"]
    open_trade_rows = load_open_trade_snapshots(open_tr, ohlc_cache)
    email_lines.append(f"\nPAPER PORTFOLIO ({len(open_tr)} open):")
    for t in open_trade_rows:
        ep = t['entry_price']
        alloc_str = ""
        if t["alloc_pct"]:
            alloc_str = f" | {t['alloc_pct']*100:.1f}%"
            if portfolio_size > 0:
                alloc_str += f" = ${portfolio_size * t['alloc_pct']:,.0f}"
        email_lines.append(
            f"  {t['ticker']} {t['direction']} @ ${ep:.2f} "
            f"(entered {t['entry_date']}, target {t['target_date']}){alloc_str}")
        if t["latest_close"] == t["latest_close"]:
            email_lines.append(
                f"    -> Latest ${t['latest_close']:.2f} | Unrealized {t['unrealized_pct']:+.1f}%")
        if t["trailing_pct"] > 0:
            email_lines.append(
                f"    -> Trail {t['trailing_pct']*100:.0f}% | initial SL ${t['sl_price']:.2f}")
        else:
            email_lines.append(
                f"    -> TP ${t['tp_price']:.2f} | SL ${t['sl_price']:.2f}")
    if len(closed_tr) > 0:
        wins = closed_tr[closed_tr["pnl_pct"] > 0]
        total_pnl = closed_tr["pnl_pct"].sum()
        win_rate = len(wins) / len(closed_tr) * 100
        email_lines.append(
            f"\n  SCORECARD: {len(closed_tr)} trades | "
            f"Win rate: {win_rate:.0f}% | Total P&L: {total_pnl:+.1f}%")
        scorecard = {
            "closed_count": len(closed_tr),
            "win_rate": win_rate,
            "total_pnl": total_pnl,
        }
    else:
        scorecard = None
else:
    open_trade_rows = []
    scorecard = None

# IBKR orders submitted
if ibkr_orders:
    email_lines.append(f"\nIBKR ORDERS SUBMITTED:")
    for o in ibkr_orders:
        email_lines.append(f"  {o}")
    ib_status_lines.extend(ibkr_orders)
elif ib:
    email_lines.append(f"\nIBKR: Connected, no new orders today.")
    ib_status_lines.append("Connected, no new orders submitted today.")
else:
    email_lines.append(f"\nIBKR: Not connected (orders not submitted).")

email_lines.append(f"\n---")
email_lines.append(f"RULES:")
email_lines.append(f"  - Orders auto-submitted to IBKR paper account with bracket TP/SL.")
email_lines.append(f"  - If IBKR not connected, orders must be placed manually.")
email_lines.append(f"  - If neither TP nor SL hits by target date, close position manually.")
email_lines.append(f"Position sizing: half-Kelly / 8 concurrent")
email_lines.append(f"RIVN 4% | ENVX 3% | TSLA 2.8% | BITF 3% | PATH 2% | HON 2%")

email_body = "\n".join(email_lines)
email_html = build_email_html(
    today=today,
    vix_level=vix_level,
    vix_regime=vix_regime,
    action_rows=action_rows,
    skipped_regime=skipped_regime,
    closed_trade_rows=closed_trades,
    open_trade_rows=open_trade_rows,
    scorecard=scorecard,
    ib_status_lines=ib_status_lines,
)
n_active_signals = sum(1 for r in results if r["action"] != "SKIP" and r["ticker"] in ACTIVE)
subject = f"StockAI {today}: {n_active_signals} active signal{'s' if n_active_signals != 1 else ''} (VIX {vix_level:.0f})"
send_email(subject, email_body, email_html)

# Discord (optional)
webhook = os.environ.get("DISCORD_WEBHOOK", "")
if webhook:
    parts = [f"**StockAI -- {today}** (VIX {vix_level:.1f})"]
    if action_lines:
        parts.extend(action_lines)
    else:
        parts.append("No active signals today.")
    if skipped_regime:
        parts.append("Regime filters: " + "; ".join(skipped_regime))
    send_discord(webhook, "\n".join(parts))

# ── Disconnect IBKR ──────────────────────────────────────────────────────────
if ib:
    ib.disconnect()
    print("IBKR disconnected.")
