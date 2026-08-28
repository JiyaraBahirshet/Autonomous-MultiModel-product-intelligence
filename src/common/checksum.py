from hashlib import sha256
from pathlib import Path


DEFAULT_CHUNK_SIZE = 1024 * 1024  # 1 MiB


def calculate_sha256(
    file_path: str | Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> str:
    """
    Calculate the SHA-256 checksum of a file using streaming reads.

    The complete file is never loaded into memory.
    """
    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    digest = sha256()

    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)

    return digest.hexdigest()
