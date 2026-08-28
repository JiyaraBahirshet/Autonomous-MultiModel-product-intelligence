import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from common.logging_utils import setup_logging
from common.run_metadata import collect_run_metadata


def main():
    run_name = "a0_foundation_test"

    log_file = setup_logging(run_name)

    import logging
    logger = logging.getLogger("a0")

    metadata = collect_run_metadata(run_name)

    logger.info("A0 foundation test started.")
    logger.info("Project phase: %s", metadata["project_phase"])
    logger.info("Processing version: %s", metadata["processing_version"])
    logger.info("Seed: %s", metadata["seed"])
    logger.info("Python: %s", metadata["python_version"])
    logger.info("A0 foundation test completed successfully.")

    metadata_file = (
        Path(__file__).resolve().parents[1]
        / "reports"
        / "runs"
        / f"{run_name}_metadata.json"
    )

    metadata_file.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(f"LOG_FILE={log_file}")
    print(f"METADATA_FILE={metadata_file}")


if __name__ == "__main__":
    main()
