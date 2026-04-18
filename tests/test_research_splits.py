from research_pipeline import WalkForwardWindow, build_walk_forward_windows
import pandas as pd

from research_pipeline import select_best_candidate_on_validation


def test_build_walk_forward_windows_keeps_validate_and_test_separate():
    windows = build_walk_forward_windows(
        n_rows=900,
        train_size=300,
        validate_size=200,
        test_size=100,
        step_size=100,
    )

    assert windows == [
        WalkForwardWindow(train_start=0, train_end=300, validate_start=300, validate_end=500, test_start=500, test_end=600),
        WalkForwardWindow(train_start=100, train_end=400, validate_start=400, validate_end=600, test_start=600, test_end=700),
        WalkForwardWindow(train_start=200, train_end=500, validate_start=500, validate_end=700, test_start=700, test_end=800),
        WalkForwardWindow(train_start=300, train_end=600, validate_start=600, validate_end=800, test_start=800, test_end=900),
    ]


def test_select_best_candidate_uses_validation_slice_only():
    candidates = pd.DataFrame(
        [
            {"candidate_id": "a", "validate_score": 0.51, "test_score": 0.99},
            {"candidate_id": "b", "validate_score": 0.60, "test_score": 0.10},
        ]
    )

    winner = select_best_candidate_on_validation(candidates)

    assert winner["candidate_id"] == "b"
