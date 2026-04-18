import argparse
import json

from research_pipeline import run_standard_honest_walk_forward


def build_parser():
    parser = argparse.ArgumentParser(description="Run honest walk-forward validation on StockAI checkpoints.")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--train", type=int, required=True)
    parser.add_argument("--validate", type=int, required=True)
    parser.add_argument("--test", type=int, required=True)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--models", default="mini,small,base")
    parser.add_argument("--horizons", default="1,5,10")
    return parser


def run_from_args(args):
    models = [item.strip() for item in args.models.split(",") if item.strip()]
    horizons = [int(item.strip()) for item in args.horizons.split(",") if item.strip()]
    return run_standard_honest_walk_forward(
        ticker=args.ticker,
        checkpoint_dir=args.checkpoint_dir,
        models=models,
        horizons=horizons,
        train_size=args.train,
        validate_size=args.validate,
        test_size=args.test,
        step_size=args.step,
    )


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    print(json.dumps(run_from_args(parsed), default=str, indent=2))
