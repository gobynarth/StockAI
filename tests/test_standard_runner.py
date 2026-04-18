from pathlib import Path

import pandas as pd

from research_pipeline import run_standard_honest_walk_forward


def test_run_standard_honest_walk_forward_returns_windowed_test_results(tmp_path: Path):
    mini = pd.DataFrame(
        [
            {"date": "2024-01-01", "entry_close": 10.0, "pred_close": 11.0, "actual_close": 11.0, "horizon": 1},
            {"date": "2024-01-02", "entry_close": 10.0, "pred_close": 11.0, "actual_close": 11.0, "horizon": 1},
            {"date": "2024-01-03", "entry_close": 10.0, "pred_close": 11.0, "actual_close": 9.0, "horizon": 1},
            {"date": "2024-01-04", "entry_close": 10.0, "pred_close": 11.0, "actual_close": 9.0, "horizon": 1},
            {"date": "2024-01-05", "entry_close": 10.0, "pred_close": 9.0, "actual_close": 9.0, "horizon": 1},
            {"date": "2024-01-08", "entry_close": 10.0, "pred_close": 9.0, "actual_close": 9.0, "horizon": 1},
            {"date": "2024-01-09", "entry_close": 10.0, "pred_close": 9.0, "actual_close": 9.0, "horizon": 1},
            {"date": "2024-01-10", "entry_close": 10.0, "pred_close": 9.0, "actual_close": 9.0, "horizon": 1},
        ]
    )
    base = pd.DataFrame(
        [
            {"date": "2024-01-01", "entry_close": 10.0, "pred_close": 9.0, "actual_close": 11.0, "horizon": 1},
            {"date": "2024-01-02", "entry_close": 10.0, "pred_close": 9.0, "actual_close": 11.0, "horizon": 1},
            {"date": "2024-01-03", "entry_close": 10.0, "pred_close": 9.0, "actual_close": 9.0, "horizon": 1},
            {"date": "2024-01-04", "entry_close": 10.0, "pred_close": 9.0, "actual_close": 9.0, "horizon": 1},
            {"date": "2024-01-05", "entry_close": 10.0, "pred_close": 11.0, "actual_close": 9.0, "horizon": 1},
            {"date": "2024-01-08", "entry_close": 10.0, "pred_close": 11.0, "actual_close": 9.0, "horizon": 1},
            {"date": "2024-01-09", "entry_close": 10.0, "pred_close": 11.0, "actual_close": 9.0, "horizon": 1},
            {"date": "2024-01-10", "entry_close": 10.0, "pred_close": 11.0, "actual_close": 9.0, "horizon": 1},
        ]
    )

    mini.to_csv(tmp_path / "ABC_mini.csv", index=False)
    base.to_csv(tmp_path / "ABC_base.csv", index=False)

    result = run_standard_honest_walk_forward(
        ticker="ABC",
        checkpoint_dir=tmp_path,
        models=["mini", "base"],
        horizons=[1],
        train_size=2,
        validate_size=2,
        test_size=2,
        step_size=2,
    )

    assert result["n_windows"] == 2
    assert result["mean_test_score"] == 0.5
    assert result["winners"] == ["base_h1", "mini_h1"]
