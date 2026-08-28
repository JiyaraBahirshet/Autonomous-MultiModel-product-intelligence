from pathlib import Path
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_yaml(filename: str) -> dict:
    """Load a YAML configuration file from the project configs directory."""
    config_path = PROJECT_ROOT / "configs" / filename

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ValueError(
            f"Configuration must contain a YAML mapping: {config_path}"
        )

    return data


def load_project_config() -> dict:
    """Load project-wide configuration."""
    return load_yaml("project.yaml")


def load_dataset_config() -> dict:
    """Load dataset source configuration."""
    return load_yaml("datasets.yaml")


def get_project_root() -> Path:
    """Return the absolute project root."""
    return PROJECT_ROOT


def get_dataset_root(dataset_name: str) -> Path:
    """Return the configured root path for a dataset."""
    config = load_dataset_config()

    datasets = config.get("datasets", {})

    if dataset_name not in datasets:
        raise KeyError(
            f"Dataset '{dataset_name}' is not defined in datasets.yaml"
        )

    root = datasets[dataset_name].get("root")

    if not root:
        raise ValueError(
            f"Dataset '{dataset_name}' does not have a configured root"
        )

    return Path(root)


def validate_dataset_roots() -> dict:
    """Validate that all configured dataset roots exist."""
    results = {}

    for dataset_name in ("rakuten", "abo", "mave"):
        root = get_dataset_root(dataset_name)
        results[dataset_name] = {
            "path": str(root),
            "exists": root.exists(),
            "is_directory": root.is_dir(),
        }

    return results
