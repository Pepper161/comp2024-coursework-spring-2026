"""CLI entrypoint for COMP2024 coursework experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.runner import load_experiment_config, run_coursework_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run COMP2024 UNSW-NB15 experiments.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/experiment.yaml",
        help="Path to experiment YAML (relative to project root or absolute).",
    )
    parser.add_argument(
        "--budget-override",
        type=int,
        default=None,
        help="Optional override for evaluations_B for quick tests.",
    )
    parser.add_argument(
        "--max-seeds",
        type=int,
        default=None,
        help="Optional cap on number of seeds from config.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip plot generation.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path

    config_raw_text = config_path.read_text(encoding="utf-8")
    config = load_experiment_config(config_path)
    # Pass raw config text through runner so saved run_config.yaml matches what user launched.
    all_runs, summary = run_coursework_experiment(
        project_root=project_root,
        config=config,
        budget_override=args.budget_override,
        max_seeds=args.max_seeds,
        skip_plots=args.skip_plots,
        config_raw_text=config_raw_text,
    )
    print(f"Finished. rows={len(all_runs)}")
    print("Saved:")
    print(f"- {project_root / config['output']['results_dir'] / 'raw' / 'all_runs.csv'}")
    print(f"- {project_root / config['output']['results_dir'] / 'summary.csv'}")
    print(f"Summary rows={len(summary)}")


if __name__ == "__main__":
    main()
