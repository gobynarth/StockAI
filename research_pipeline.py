from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class WalkForwardWindow:
    train_start: int
    train_end: int
    validate_start: int
    validate_end: int
    test_start: int
    test_end: int


def build_walk_forward_windows(n_rows, train_size, validate_size, test_size, step_size):
    windows = []
    start = 0
    total = train_size + validate_size + test_size
    while start + total <= n_rows:
        windows.append(
            WalkForwardWindow(
                train_start=start,
                train_end=start + train_size,
                validate_start=start + train_size,
                validate_end=start + train_size + validate_size,
                test_start=start + train_size + validate_size,
                test_end=start + total,
            )
        )
        start += step_size
    return windows


def select_best_candidate_on_validation(candidates_df):
    ordered = candidates_df.sort_values(
        ["validate_score", "candidate_id"],
        ascending=[False, True],
    ).reset_index(drop=True)
    return ordered.iloc[0].to_dict()


def simulate_trade_from_bars(bars, direction, entry_price, tp_pct, sl_pct, trailing_pct=0.0):
    peak = entry_price
    trough = entry_price
    current_sl = entry_price * (1 - sl_pct) if direction == "UP" else entry_price * (1 + sl_pct)

    for _, bar in bars.iterrows():
        high = float(bar["high"])
        low = float(bar["low"])
        close = float(bar["close"])

        if direction == "UP":
            peak = max(peak, high)
            if trailing_pct > 0:
                current_sl = max(current_sl, peak * (1 - trailing_pct))
            if low <= current_sl:
                return {"exit_reason": "TRAIL" if trailing_pct > 0 else "SL", "exit_price": current_sl, "exit_close": close}
            if high >= entry_price * (1 + tp_pct):
                return {"exit_reason": "TP", "exit_price": entry_price * (1 + tp_pct), "exit_close": close}
        else:
            trough = min(trough, low)
            if trailing_pct > 0:
                current_sl = min(current_sl, trough * (1 + trailing_pct))
            if high >= current_sl:
                return {"exit_reason": "TRAIL" if trailing_pct > 0 else "SL", "exit_price": current_sl, "exit_close": close}
            if low <= entry_price * (1 - tp_pct):
                return {"exit_reason": "TP", "exit_price": entry_price * (1 - tp_pct), "exit_close": close}

    final_close = float(bars.iloc[-1]["close"])
    return {"exit_reason": "EXPIRY", "exit_price": final_close, "exit_close": final_close}


def run_walk_forward_selection(candidate_scores, validate_row_ids, test_row_ids):
    validate_df = candidate_scores[candidate_scores["row_id"].isin(validate_row_ids)]
    test_df = candidate_scores[candidate_scores["row_id"].isin(test_row_ids)]

    candidate_columns = [c for c in candidate_scores.columns if c != "row_id"]
    validate_means = {name: float(validate_df[name].mean()) for name in candidate_columns}
    winner = max(validate_means, key=validate_means.get)
    test_score = float(test_df[winner].mean())
    return {"winner": winner, "validate_score": validate_means[winner], "test_score": test_score}


def load_standard_checkpoint_candidates(ticker, checkpoint_dir, models, horizons):
    merged = None
    checkpoint_dir = Path(checkpoint_dir)

    for model in models:
        path = checkpoint_dir / f"{ticker}_{model}.csv"
        df = pd.read_csv(path, parse_dates=["date"])
        for horizon in horizons:
            sub = df[df["horizon"] == horizon].copy() if "horizon" in df.columns else df.copy()
            sub = sub[["date", "entry_close", "pred_close", "actual_close"]].copy()
            sub[f"{model}_h{horizon}"] = (sub["pred_close"] > sub["entry_close"]).astype(float)
            sub["actual_up"] = (sub["actual_close"] > sub["entry_close"]).astype(float)
            sub = sub[["date", "actual_up", f"{model}_h{horizon}"]]
            merged = sub if merged is None else merged.merge(sub, on=["date", "actual_up"], how="inner")

    return merged.sort_values("date").reset_index(drop=True)


def run_standard_honest_walk_forward(
    ticker,
    checkpoint_dir,
    models,
    horizons,
    train_size,
    validate_size,
    test_size,
    step_size=None,
):
    candidates = load_standard_checkpoint_candidates(
        ticker=ticker,
        checkpoint_dir=checkpoint_dir,
        models=models,
        horizons=horizons,
    )
    candidate_columns = [c for c in candidates.columns if c not in {"date", "actual_up"}]
    scored = pd.DataFrame({"row_id": range(len(candidates))})
    for col in candidate_columns:
        scored[col] = (candidates[col] == candidates["actual_up"]).astype(float)

    windows = build_walk_forward_windows(
        n_rows=len(scored),
        train_size=train_size,
        validate_size=validate_size,
        test_size=test_size,
        step_size=step_size or test_size,
    )

    results = []
    for window in windows:
        validate_row_ids = list(range(window.validate_start, window.validate_end))
        test_row_ids = list(range(window.test_start, window.test_end))
        result = run_walk_forward_selection(scored, validate_row_ids, test_row_ids)
        results.append(result)

    if not results:
        return {"ticker": ticker, "n_windows": 0, "mean_test_score": float("nan"), "winners": []}

    return {
        "ticker": ticker,
        "n_windows": len(results),
        "mean_test_score": float(pd.Series([r["test_score"] for r in results]).mean()),
        "winners": [r["winner"] for r in results],
        "results": results,
    }
