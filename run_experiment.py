"""Entry point for COMP2024 coursework experiments."""

from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parent
    data_dir = project_root / "dataset"
    print(f"Project root: {project_root}")
    print(f"Dataset dir: {data_dir}")
    print("TODO: implement baseline + metaheuristic experiments")


if __name__ == "__main__":
    main()
