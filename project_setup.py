from pathlib import Path

PROJECT_NAME = ""

PROJECT_DIRS = [
    "notebooks",
    "src",
    "configs",
    "results",
    "results/figures",
    "results/metrics",
    "results/tables",
    "models",
    "data",
]

PROJECT_FILES = [
    "README.md",
    "requirements.txt",
    ".gitignore",
    "LICENSE",
    "notebooks/galaxy_morphology_experiment.ipynb",
    "src/dataset.py",
    "src/models.py",
    "src/train.py",
    "src/evaluate.py",
    "configs/config.yaml",
    "data/.gitkeep",
]


def create_project():
    root = Path.cwd() / PROJECT_NAME
    root.mkdir(parents=True, exist_ok=True)

    for directory in PROJECT_DIRS:
        path = root / directory
        path.mkdir(parents=True, exist_ok=True)

    for file in PROJECT_FILES:
        path = root / file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)

    print(f"Project created at: {root}")


if __name__ == "__main__":
    create_project()