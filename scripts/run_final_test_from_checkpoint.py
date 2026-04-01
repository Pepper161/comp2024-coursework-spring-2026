"""Re-run only the final full-train/test evaluation from a saved checkpoint."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", required=True, type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    project_root = Path(args.project_root).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    config_path = Path(args.config).resolve()
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        solution_path = tmp_path / "solution.json"
        output_path = tmp_path / "result.json"
        solution_path.write_text(
            json.dumps(
                {
                    "mask": [bool(v) for v in checkpoint["selected_mask"].values()],
                    "params": dict(checkpoint["params"]),
                }
            ),
            encoding="utf-8",
        )

        worker_path = project_root / "scripts" / "final_test_worker.py"
        completed = subprocess.run(
            [
                sys.executable,
                str(worker_path),
                "--project-root",
                str(project_root),
                "--config",
                str(config_path),
                "--seed",
                str(args.seed),
                "--solution-json",
                str(solution_path),
                "--output-json",
                str(output_path),
            ],
            cwd=str(project_root),
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "Final test rerun failed.\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )

        result = json.loads(output_path.read_text(encoding="utf-8"))
        checkpoint["test_metrics"] = result["metrics"]
        checkpoint["stage"] = "final_test_completed"
        checkpoint_path.write_text(json.dumps(checkpoint, indent=2, sort_keys=True), encoding="utf-8")
        print("Final test rerun completed.")
        print(f"Updated checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
