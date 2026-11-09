"""Command-line entry point."""
import argparse
import json
from pathlib import Path
from .agent import Agent
from .evaluation import TEST_SEEDS, VALIDATION_SEEDS, evaluate_policy
from .experiments import run_experiment
from .recording import record
from .train import DEFAULT_CONFIG, train


def main():
    parser = argparse.ArgumentParser(description="MountainCar continuous PPO")
    sub = parser.add_subparsers(dest="command", required=True)
    training = sub.add_parser("train")
    training.add_argument("--config", type=Path)
    training.add_argument("--seed", type=int, default=0)
    training.add_argument("--output-dir", type=Path, default=Path("runs/train"))
    experiment = sub.add_parser("experiment")
    experiment.add_argument("--config", type=Path)
    experiment.add_argument("--output-dir", type=Path, default=Path("results/revised"))
    experiment.add_argument("--phase", choices=["pilot", "confirm", "final", "report"], default="pilot")
    experiment.add_argument("--resume", action="store_true")
    evaluation = sub.add_parser("evaluate")
    evaluation.add_argument("--model", required=True, help="Checkpoint path, random, random_hold, or heuristic")
    evaluation.add_argument("--stochastic", action="store_true")
    evaluation.add_argument("--split", choices=["validation", "test"], default="validation")
    evaluation.add_argument("--output", type=Path)
    plotting = sub.add_parser("plot")
    plotting.add_argument("--output-dir", type=Path, default=Path("results"))
    recording = sub.add_parser("record")
    recording.add_argument("--model", required=True, help="Checkpoint path, random, random_hold, or heuristic")
    recording.add_argument("--output", type=Path, default=Path("results/episode.mp4"))
    recording.add_argument("--seed", type=int, default=10000)
    args = parser.parse_args()
    if args.command in ("train", "experiment"):
        config = {**DEFAULT_CONFIG, **(json.loads(args.config.read_text(encoding="utf-8")) if args.config else {})}
        result = train(config, args.seed, args.output_dir) if args.command == "train" else run_experiment(config, args.output_dir, args.resume, args.phase)
    elif args.command == "plot":
        from .plots import plot_results
        plot_results(args.output_dir)
        result = {"output_dir": str(args.output_dir)}
    else:
        import torch
        torch.set_num_threads(1)
        policy = args.model if args.model in ("random", "random_hold", "heuristic") else Agent.load(args.model)
        if args.command == "evaluate":
            result = evaluate_policy(policy, VALIDATION_SEEDS if args.split == "validation" else TEST_SEEDS, deterministic=not args.stochastic)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        else:
            result = {"video": str(record(policy, args.output, args.seed))}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
