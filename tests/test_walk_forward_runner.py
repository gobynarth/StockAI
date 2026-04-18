import pandas as pd

from research_pipeline import run_walk_forward_selection


def test_run_walk_forward_selection_reports_test_results_from_frozen_winner():
    rows = pd.DataFrame(
        [
            {"row_id": 0, "candidate_a": 0.40, "candidate_b": 0.70},
            {"row_id": 1, "candidate_a": 0.45, "candidate_b": 0.72},
            {"row_id": 2, "candidate_a": 0.60, "candidate_b": 0.20},
        ]
    )

    result = run_walk_forward_selection(
        candidate_scores=rows,
        validate_row_ids=[0, 1],
        test_row_ids=[2],
    )

    assert result["winner"] == "candidate_b"
    assert result["test_score"] == 0.20
