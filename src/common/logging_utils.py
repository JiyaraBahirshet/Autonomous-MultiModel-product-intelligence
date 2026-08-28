import logging
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = PROJECT_ROOT / "reports" / "runs"


def setup_logging(run_name: str = "phase_a") -> Path:
    """
    Configure console and file logging for a pipeline run.

    Returns:
        Path to the generated log file.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    log_file = LOG_DIR / f"{run_name}.log"

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers if setup_logging() is called more than once.
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    file_handler = logging.FileHandler(
        log_file,
        mode="w",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return log_file
