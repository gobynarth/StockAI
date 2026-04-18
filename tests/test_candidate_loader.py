from pathlib import Path

import pandas as pd

from research_pipeline import load_standard_checkpoint_candidates


def test_load_standard_checkpoint_candidates_builds_candidate_matrix(tmp_path: Path):
    mini = pd.DataFrame(
        [
            {"date": "2024-01-01", "entry_close": 10.0, "pred_close": 11.0, "actual_close": 12.0, "horizon": 1},
            {"date": "2024-01-02", "entry_close": 11.0, "pred_close": 10.0, "actual_close": 10.0, "horizon": 1},
        ]
    )
    base = pd.DataFrame(
        [
            {"date": "2024-01-01", "entry_close": 10.0, "pred_close": 12.0, "actual_close": 12.0, "horizon": 1},
            {"date": "2024-01-02", "entry_close": 11.0, "pred_close": 9.0, "actual_close": 10.0, "horizon": 1},
        ]
    )

    mini.to_csv(tmp_path / "ABC_mini.csv", index=False)
    base.to_csv(tmp_path / "ABC_base.csv", index=False)

    result = load_standard_checkpoint_candidates(
        ticker="ABC",
        checkpoint_dir=tmp_path,
        models=["mini", "base"],
        horizons=[1],
    )

    assert list(result.columns) == ["date", "actual_up", "mini_h1", "base_h1"]
    assert result["mini_h1"].tolist() == [1.0, 0.0]
    assert result["base_h1"].tolist() == [1.0, 0.0]
    assert result["actual_up"].tolist() == [1.0, 0.0]
