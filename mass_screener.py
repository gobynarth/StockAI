"""
Mass screener: run Kronos-base h=60 on top 500 US stocks.
Auto-discover new edge tickers.

Output: /workspace/screener_results.csv ranked by validation accuracy.
"""
import os, sys
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import timedelta
from env_paths import add_kronos_to_path, base_path

add_kronos_to_path()
from model import Kronos, KronosTokenizer, KronosPredictor

# Universe: weighted toward small/mid caps where edge is more likely
# Combines S&P 500 + popular small-caps + retail favorites
UNIVERSE = [
    # S&P 500 large caps (smaller subset — most won't have edge)
    "AAPL","MSFT","NVDA","GOOGL","AMZN","META","TSLA","BRK-B","UNH","XOM",
    "JPM","JNJ","V","PG","MA","HD","CVX","ABBV","MRK","KO","AVGO","PEP",
    "WMT","COST","BAC","TMO","CSCO","ACN","ABT","DHR","WFC","MCD","ADBE",
    "NEE","CRM","TXN","NFLX","DIS","NKE","BMY","CMCSA","PM","RTX","UPS",
    "COP","LIN","HON","INTU","SCHW","T","ORCL","UNP","IBM","QCOM","AMGN",
    "INTC","SBUX","BA","BLK","CAT","ELV","DE","SPGI","PLD","GE","NOW",
    "AMD","ISRG","AMT","BKNG","MDLZ","TJX","GILD","ADP","SYK","C","MO",
    "MMM","ZTS","CB","DUK","SO","TGT","MU","REGN","CCI","BSX","PYPL",
    "VRTX","FISV","SHW","FDX","EOG","ATVI","GM","CL","PNC","CME","FCX",
    "AON","NSC","ITW","HUM","MMC","BDX","USB","ICE","WM","EQIX","TFC",
    # Mid/small caps with momentum (sweet spot for Kronos)
    "RIVN","ENVX","COIN","NIO","RIOT","SMCI","CRWD","MARA","SOFI","UPST",
    "AFRM","PLTR","HOOD","LCID","IONQ","RKLB","ASTS","LUNR","IREN","CAR",
    "DJT","NBIS","RDDT","ANET","PANW","SNOW","DDOG","NET","ZS","TEAM",
    "OKTA","DOCU","TWLO","ZM","FSLY","WORK","BILL","ENPH","SEDG","FSLR",
    "PLUG","RUN","NOVA","CHPT","BLNK","BLDP","FCEL","BE","STEM","SHLS",
    "PARA","WBD","FUBO","ROKU","SPOT","SNAP","PINS","BMBL","UBER","LYFT",
    "DASH","ABNB","SHOP","CHWY","WAYF","RBLX","U","TTD","PSCS","DKNG",
    "CZR","MGM","WYNN","LVS","PENN","BYD","RCL","CCL","NCLH","AAL","UAL",
    "DAL","LUV","JBLU","SAVE","SPCE","JOBY","ACHR","BNGO","CRSP","NVAX",
    "MRNA","BNTX","PFE","GILD","REGN","VRTX","BIIB","TWST","BEAM","EDIT",
    "NTLA","SANA","RXRX","SDGR","RGEN","ARKG","ARKK","ARKQ","ARKW","ARKF",
    # High beta / volatile retail names
    "GME","AMC","BBBY","MULN","NKLA","FSR","WKHS","SOLO","RIDE","HYLN",
    "GOEV","XPEV","LI","NIO","BIDU","BABA","PDD","JD","TME","DIDI",
    "BILI","TIGR","FUTU","NKE","LULU","UAA","UA","HBI","GPS","ANF",
    "URBN","AEO","KSS","DDS","M","JWN","TGT","WMT","COST","DG","DLTR",
    # Tech mid-caps
    "OKLO","BLDR","FOUR","CRDO","CELH","DASH","ROKU","RBLX","FUBO","NBIS",
    "GTLB","S","FROG","ESTC","WIX","SQSP","HUBS","ENB","LSPD","BIGC",
    "OUST","INDI","LAZR","VLDR","MVIS","GH","INSM","BPMC","REPL","MARK",
    "MNMD","CNK","IMAX","PEAR","COMP","REAL","OPEN","OPAD","Z","ZG",
    "SFIX","REVG","HCKT","HAL","SLB","BKR","DVN","FANG","PXD","OXY",
    "MRO","APA","CTRA","HES","COG","EQT","RRC","AR","SWN","CHK",
    # Fintech / crypto adjacent
    "MSTR","BITO","BITX","CONL","CONY","MARA","RIOT","CIFR","HUT","BTBT",
    "GREE","WULF","BITF","CLSK","HIVE","VIBE","BRPHF","CAN","SOS","EBON",
    # Biotech mid-caps
    "VKTX","SAVA","HIMS","CVS","WBA","ABT","TMO","DHR","ISRG","SYK",
    "MDT","BSX","ZBH","RMD","DXCM","TDOC","ONEM","NVTA","AMWL","HIMS",
    # Energy / clean tech
    "TSLA","ENPH","SEDG","FSLR","RUN","SPWR","NOVA","ARRY","DQ","JKS",
    "CSIQ","MAXN","SHLS","NEE","DUK","SO","ED","D","PCG","EXC",
    # Defense / aerospace
    "LMT","RTX","NOC","BA","GD","HII","TXT","HEI","TDG","LHX",
    # Materials / commodities
    "FCX","NEM","GOLD","AEM","BTG","KGC","HMY","NGD","WPM","FNV",
    "X","CLF","NUE","STLD","RS","MT","TX","CMC","BBL","MOS",
    "LAC","ALB","LTHM","SQM","PLL","SGML","RIO","BHP","TECK","VALE",
    # Quantum / AI / emerging tech
    "QBTS","RGTI","QUBT","BBAI","SOUN","AI","C3AI","INOD","NICE","VRNT",
    "PATH","UIPATH","BBAI","DOMO","CLOV","PRGS","WK","NCNO","DOCN","OSCR",
]
# Dedupe
UNIVERSE = list(dict.fromkeys(UNIVERSE))
print(f"Universe size: {len(UNIVERSE)} tickers")

HORIZON     = 60
TEMPERATURE = 1.0
LOOKBACK    = 200
N_WINDOWS   = 400  # 200 selection + 200 validation

CKPT_DIR = base_path("screener_checkpoints")
DATA_DIR = base_path("data")
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


def fetch_data(ticker):
    csv_path = f"{DATA_DIR}/{ticker}.csv"
    if os.path.exists(csv_path):
        raw = pd.read_csv(csv_path, parse_dates=["timestamps"])
        raw["timestamps"] = pd.to_datetime(raw["timestamps"]).dt.tz_localize(None)
    else:
        raw = yf.download(ticker, period="5y", interval="1d",
                          auto_adjust=True, progress=False, timeout=60)
        if raw.empty:
            return None
        raw = raw.reset_index()
        raw.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in raw.columns]
        date_col = "date" if "date" in raw.columns else "datetime"
        raw = raw.rename(columns={date_col: "timestamps"})
        raw["amount"] = raw["close"] * raw["volume"]
        raw["timestamps"] = pd.to_datetime(raw["timestamps"]).dt.tz_localize(None)
        raw[["timestamps","open","high","low","close","volume","amount"]].dropna().to_csv(csv_path, index=False)
    raw = raw[["timestamps","open","high","low","close","volume","amount"]].dropna()
    return raw


print(f"Loading Kronos-base...")
tok = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
mdl = Kronos.from_pretrained("NeoQuasar/Kronos-base")
predictor = KronosPredictor(mdl, tok, max_context=512)

results = []
for idx, ticker in enumerate(UNIVERSE):
    print(f"\n[{idx+1}/{len(UNIVERSE)}] {ticker}")
    ckpt = f"{CKPT_DIR}/{ticker}.csv"

    if os.path.exists(ckpt):
        df = pd.read_csv(ckpt)
        if len(df) >= N_WINDOWS:
            print(f"  done already, val acc={df.iloc[N_WINDOWS//2:]['correct'].mean()*100:.1f}%")
            continue

    try:
        raw = fetch_data(ticker)
        if raw is None or len(raw) < LOOKBACK + HORIZON + 100:
            print(f"  not enough data, skip")
            continue
    except Exception as e:
        print(f"  fetch error: {e}, skip")
        continue

    max_n = len(raw) - LOOKBACK - HORIZON
    actual_n = min(N_WINDOWS, max_n)
    test_start = len(raw) - actual_n - HORIZON

    rows = []
    for i in range(actual_n):
        idx_w = test_start + i
        x_df = raw.iloc[idx_w - LOOKBACK: idx_w][["open","high","low","close","volume","amount"]].reset_index(drop=True)
        x_ts = raw.iloc[idx_w - LOOKBACK: idx_w]["timestamps"].reset_index(drop=True)
        entry_close = raw.iloc[idx_w - 1]["close"]
        entry_date  = raw.iloc[idx_w - 1]["timestamps"]
        actual_close = raw.iloc[idx_w + HORIZON - 1]["close"]
        y_dates = pd.bdate_range(start=entry_date + timedelta(days=1), periods=HORIZON)
        y_ts = pd.Series(y_dates)

        try:
            pred = predictor.predict(df=x_df, x_timestamp=x_ts, y_timestamp=y_ts,
                                     pred_len=HORIZON, T=TEMPERATURE, top_p=0.9,
                                     sample_count=1, verbose=False)
            pred_close = pred["close"].iloc[-1]
        except Exception as e:
            continue

        correct = (pred_close > entry_close) == (actual_close > entry_close)
        rows.append({
            "date": str(entry_date.date()),
            "entry_close": round(float(entry_close), 4),
            "pred_close":  round(float(pred_close), 4),
            "actual_close":round(float(actual_close), 4),
            "correct": int(correct),
        })

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{actual_n}]")

    pd.DataFrame(rows).to_csv(ckpt, index=False)
    if len(rows) >= 100:
        sel = rows[:len(rows)//2]
        val = rows[len(rows)//2:]
        sel_acc = np.mean([r["correct"] for r in sel]) * 100
        val_acc = np.mean([r["correct"] for r in val]) * 100
        long_signals = [r for r in val if r["pred_close"] > r["entry_close"]]
        long_wr = np.mean([r["correct"] for r in long_signals]) * 100 if long_signals else 0
        print(f"  -> sel={sel_acc:.1f}% val={val_acc:.1f}% long_wr={long_wr:.1f}% (n={len(long_signals)})")
        results.append({
            "ticker": ticker, "n": len(rows),
            "sel_acc": round(sel_acc, 1), "val_acc": round(val_acc, 1),
            "long_wr": round(long_wr, 1), "long_n": len(long_signals),
        })
        # Save running results
        pd.DataFrame(results).sort_values("val_acc", ascending=False).to_csv(
            base_path("screener_results.csv"), index=False)

# ── Final summary ──────────────────────────────────────────────────────────
print(f"\n\n{'='*70}\nSCREENER RESULTS - TOP 30 BY VALIDATION ACCURACY\n{'='*70}")
df = pd.DataFrame(results).sort_values("val_acc", ascending=False)
print(df.head(30).to_string(index=False))
df.to_csv(base_path("screener_results.csv"), index=False)
print(f"\nFull results: {base_path('screener_results.csv')}")
