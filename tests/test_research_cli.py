from pathlib import Path

import pandas as pd

from honest_walk_forward import build_parser, run_from_args


def test_build_parser_exposes_required_retest_arguments():
    parser = build_parser()
    args = parser.parse_args(["--ticker", "RIVN", "--train", "300", "--validate", "200", "--test", "100"])

    assert args.ticker == "RIVN"
    assert args.train == 300
    assert args.validate == 200
    assert args.test == 100


def test_build_parser_accepts_checkpoint_models_and_horizons():
    parser = build_parser()
    args = parser.parse_args(
        [
            "--ticker", "TSLA",
            "--train", "300",
            "--validate", "200",
            "--test", "100",
            "--checkpoint-dir", "checkpoints",
            "--models", "mini,base",
            "--horizons", "1,5,10",
        ]
    )

    assert args.checkpoint_dir == "checkpoints"
    assert args.models == "mini,base"
    assert args.horizons == "1,5,10"


def test_run_from_args_executes_standard_runner(tmp_path: Path):
    mini = pd.DataFrame(
        [
            {"date": "2024-01-01", "entry_close": 10.0, "pred_close": 11.0, "actual_close": 11.0, "horizon": 1},
            {"date": "2024-01-02", "entry_close": 10.0, "pred_close": 11.0, "actual_close": 11.0, "horizon": 1},
            {"date": "2024-01-03", "entry_close": 10.0, "pred_close": 11.0, "actual_close": 9.0, "horizon": 1},
            {"date": "2024-01-04", "entry_close": 10.0, "pred_close": 11.0, "actual_close": 9.0, "horizon": 1},
            {"date": "2024-01-05", "entry_close": 10.0, "pred_close": 9.0, "actual_close": 9.0, "horizon": 1},
            {"date": "2024-01-08", "entry_close": 10.0, "pred_close": 9.0, "actual_close": 9.0, "horizon": 1},
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
        ]
    )
    mini.to_csv(tmp_path / "ABC_mini.csv", index=False)
    base.to_csv(tmp_path / "ABC_base.csv", index=False)

    parser = build_parser()
    args = parser.parse_args(
        [
            "--ticker", "ABC",
            "--train", "2",
            "--validate", "2",
            "--test", "2",
            "--checkpoint-dir", str(tmp_path),
            "--models", "mini,base",
            "--horizons", "1",
        ]
    )

    result = run_from_args(args)

    assert result["ticker"] == "ABC"
    assert result["n_windows"] == 1
