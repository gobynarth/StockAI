"""
Mass screener with BATCHED inference. ~16x faster than sequential.
Runs Kronos-base h=60 on top 500 US stocks.
"""
import os, sys
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import timedelta
from env_paths import add_kronos_to_path, base_path

add_kronos_to_path()
from model import Kronos, KronosTokenizer, KronosPredictor

# Universe (deduped, ~400 unique tickers)
UNIVERSE = sorted(set([
    # S&P 500 representatives
    "AAPL","MSFT","NVDA","GOOGL","AMZN","META","TSLA","JPM","JNJ","V","PG","MA","HD","CVX","ABBV","MRK","KO","AVGO","PEP","WMT","COST","BAC","TMO","CSCO","ACN","ABT","DHR","WFC","MCD","ADBE","NEE","CRM","TXN","NFLX","DIS","NKE","BMY","CMCSA","RTX","UPS","COP","LIN","HON","INTU","SCHW","T","ORCL","UNP","IBM","QCOM","AMGN","INTC","SBUX","BA","BLK","CAT","ELV","DE","SPGI","PLD","GE","NOW","AMD","ISRG","BKNG","MDLZ","TJX","GILD","ADP","SYK","C","MO","MMM","ZTS","CB","DUK","SO","TGT","MU","REGN","CCI","BSX","PYPL","VRTX","SHW","FDX","EOG","GM","CL","PNC","CME","FCX","AON","NSC","ITW","HUM","MMC","BDX","USB","ICE","WM","EQIX","TFC","CVS","WBA","BIIB",
    # Mid/small cap momentum
    "RIVN","ENVX","COIN","NIO","RIOT","SMCI","CRWD","MARA","SOFI","UPST","AFRM","PLTR","HOOD","LCID","IONQ","RKLB","ASTS","LUNR","IREN","CAR","DJT","NBIS","RDDT","ANET","PANW","SNOW","DDOG","NET","ZS","TEAM","OKTA","DOCU","TWLO","ZM","BILL","ENPH","SEDG","FSLR","PLUG","RUN","NOVA","CHPT","BLNK","BLDP","FCEL","BE","STEM","SHLS","PARA","WBD","FUBO","ROKU","SPOT","SNAP","PINS","BMBL","UBER","LYFT","DASH","ABNB","SHOP","CHWY","RBLX","U","TTD","DKNG","CZR","MGM","WYNN","LVS","PENN","RCL","CCL","NCLH","AAL","UAL","DAL","LUV","JBLU","SAVE","SPCE","JOBY","ACHR","CRSP","NVAX","MRNA","BNTX","PFE","TWST","BEAM","EDIT","NTLA","RXRX","SDGR","ARKG","ARKK","ARKQ","ARKW","ARKF",
    # Volatile retail
    "GME","AMC","MULN","NKLA","FSR","WKHS","RIDE","HYLN","GOEV","XPEV","LI","BIDU","BABA","PDD","JD","TME","DIDI","BILI","TIGR","FUTU","LULU","UAA","HBI","GPS","ANF","URBN","AEO","KSS","DDS","M","JWN","DG","DLTR",
    # Tech mid-caps
    "OKLO","BLDR","FOUR","CRDO","CELH","GTLB","S","FROG","ESTC","WIX","HUBS","LSPD","BIGC","OUST","INDI","LAZR","VLDR","MVIS","GH","INSM","BPMC","REPL","MNMD","CNK","IMAX","REAL","OPEN","Z","SFIX","REVG","HAL","SLB","BKR","DVN","FANG","OXY","MRO","APA","CTRA","HES","EQT","RRC","AR","SWN","CHK",
    # Crypto
    "MSTR","BITO","CONL","CONY","CIFR","HUT","BTBT","GREE","WULF","BITF","CLSK","HIVE","CAN","SOS",
    # Biotech mid
    "VKTX","SAVA","HIMS","TDOC","NVTA","AMWL",
    # Clean energy
    "SPWR","ARRY","DQ","JKS","CSIQ","LAC","ALB","LTHM","SQM","PLL","SGML",
    # Materials
    "FCX","NEM","GOLD","AEM","BTG","KGC","HMY","NGD","WPM","FNV","X","CLF","NUE","STLD","RS","MT","CMC","MOS","RIO","BHP","TECK","VALE",
    # Quantum / AI
    "QBTS","RGTI","QUBT","BBAI","SOUN","AI","INOD","NICE","VRNT","PATH","DOMO","CLOV","PRGS","WK","NCNO","DOCN","OSCR",
    # Defense
    "LMT","NOC","GD","HII","TXT","HEI","TDG","LHX",
]))
print(f"Universe: {len(UNIVERSE)} unique tickers")

HORIZON     = 60
TEMPERATURE = 1.0
LOOKBACK    = 200
N_WINDOWS   = 400
BATCH_SIZE  = 16

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
        try:
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
        except Exception as e:
            print(f"  fetch error: {e}")
            return None
    raw = raw[["timestamps","open","high","low","close","volume","amount"]].dropna()
    return raw


print(f"Loading Kronos-base...")
tok = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
mdl = Kronos.from_pretrained("NeoQuasar/Kronos-base")
predictor = KronosPredictor(mdl, tok, max_context=512)
print("Model loaded.\n")

results = []
total = len(UNIVERSE)

for idx, ticker in enumerate(UNIVERSE):
    print(f"[{idx+1}/{total}] {ticker}", end=" ")
    ckpt = f"{CKPT_DIR}/{ticker}.csv"

    if os.path.exists(ckpt):
        df = pd.read_csv(ckpt)
        if len(df) >= 100:
            sel = df.iloc[:len(df)//2]
            val = df.iloc[len(df)//2:]
            sel_acc = sel["correct"].mean() * 100
            val_acc = val["correct"].mean() * 100
            longs = val[val["pred_close"] > val["entry_close"]]
            long_wr = longs["correct"].mean() * 100 if len(longs) > 0 else 0
            print(f"-> cached val={val_acc:.1f}% long_wr={long_wr:.1f}%")
            results.append({"ticker": ticker, "n": len(df),
                            "sel_acc": round(sel_acc,1), "val_acc": round(val_acc,1),
                            "long_wr": round(long_wr,1), "long_n": len(longs)})
            continue

    raw = fetch_data(ticker)
    if raw is None or len(raw) < LOOKBACK + HORIZON + 100:
        print("-> insufficient data")
        continue

    max_n = len(raw) - LOOKBACK - HORIZON
    actual_n = min(N_WINDOWS, max_n)
    test_start = len(raw) - actual_n - HORIZON

    # Build all batches upfront
    rows = []
    for batch_start in range(0, actual_n, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, actual_n)
        df_list, x_ts_list, y_ts_list, meta = [], [], [], []
        for i in range(batch_start, batch_end):
            idx_w = test_start + i
            x_df = raw.iloc[idx_w - LOOKBACK: idx_w][["open","high","low","close","volume","amount"]].reset_index(drop=True)
            x_ts = raw.iloc[idx_w - LOOKBACK: idx_w]["timestamps"].reset_index(drop=True)
            entry_close = raw.iloc[idx_w - 1]["close"]
            entry_date  = raw.iloc[idx_w - 1]["timestamps"]
            actual_close = raw.iloc[idx_w + HORIZON - 1]["close"]
            y_dates = pd.bdate_range(start=entry_date + timedelta(days=1), periods=HORIZON)
            y_ts = pd.Series(y_dates)

            df_list.append(x_df)
            x_ts_list.append(x_ts)
            y_ts_list.append(y_ts)
            meta.append((entry_date, entry_close, actual_close))

        try:
            preds = predictor.predict_batch(
                df_list=df_list, x_timestamp_list=x_ts_list, y_timestamp_list=y_ts_list,
                pred_len=HORIZON, T=TEMPERATURE, top_p=0.9,
                sample_count=1, verbose=False)
        except Exception as e:
            print(f"\n  batch error: {e}")
            continue

        for (entry_date, entry_close, actual_close), pred in zip(meta, preds):
            pred_close = pred["close"].iloc[-1]
            correct = (pred_close > entry_close) == (actual_close > entry_close)
            rows.append({
                "date": str(entry_date.date()),
                "entry_close": round(float(entry_close), 4),
                "pred_close":  round(float(pred_close), 4),
                "actual_close":round(float(actual_close), 4),
                "correct": int(correct),
            })

    if len(rows) >= 100:
        pd.DataFrame(rows).to_csv(ckpt, index=False)
        sel = rows[:len(rows)//2]
        val = rows[len(rows)//2:]
        sel_acc = np.mean([r["correct"] for r in sel]) * 100
        val_acc = np.mean([r["correct"] for r in val]) * 100
        longs = [r for r in val if r["pred_close"] > r["entry_close"]]
        long_wr = np.mean([r["correct"] for r in longs]) * 100 if longs else 0
        print(f"-> val={val_acc:.1f}% long_wr={long_wr:.1f}% (n={len(longs)})")
        results.append({"ticker": ticker, "n": len(rows),
                        "sel_acc": round(sel_acc,1), "val_acc": round(val_acc,1),
                        "long_wr": round(long_wr,1), "long_n": len(longs)})
        # Save running results
        pd.DataFrame(results).sort_values("val_acc", ascending=False).to_csv(
            base_path("screener_results.csv"), index=False)
    else:
        print(f"-> too few rows ({len(rows)})")

# Final
print(f"\n\n{'='*70}\nTOP 30 BY VALIDATION ACCURACY\n{'='*70}")
df_final = pd.DataFrame(results).sort_values("val_acc", ascending=False)
print(df_final.head(30).to_string(index=False))
df_final.to_csv(base_path("screener_results.csv"), index=False)
print(f"\nFull results: {base_path('screener_results.csv')}")
