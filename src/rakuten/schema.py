from dataclasses import dataclass
from typing import Final


DATASET_NAME: Final[str] = "Rakuten 2018"
DATASET_ROLE: Final[str] = "hierarchical_categorization"
SCHEMA_VERSION: Final[str] = "v001"

CATEGORY_PATH_SEPARATOR: Final[str] = ">"

VALID_SPLITS: Final[frozenset[str]] = frozenset({"train", "test"})


@dataclass(frozen=True)
class RakutenRecord:
    """
    Logical representation of one processed Rakuten record.

    Raw fields:
        title
        category_id_path

    Derived fields:
        category_path
        depth
        split
    """

    title: str
    category_id_path: str
    category_path: tuple[str, ...]
    depth: int
    split: str


def parse_category_path(category_id_path: str) -> list[str]:
    """
    Parse a Rakuten hierarchical category path without changing
    the original category IDs.
    """
    if not isinstance(category_id_path, str):
        raise TypeError("category_id_path must be a string")

    if not category_id_path:
        raise ValueError("category_id_path cannot be empty")

    parts = category_id_path.split(CATEGORY_PATH_SEPARATOR)

    if any(not part for part in parts):
        raise ValueError(
            f"Invalid category path with empty category ID: "
            f"{category_id_path!r}"
        )

    if any(not part.isdigit() for part in parts):
        raise ValueError(
            f"Invalid category path containing non-numeric ID: "
            f"{category_id_path!r}"
        )

    return parts


def category_depth(category_id_path: str) -> int:
    """Return the number of nodes in a complete category path."""
    return len(parse_category_path(category_id_path))


def validate_split(split: str) -> str:
    """Validate the logical dataset partition."""
    if split not in VALID_SPLITS:
        raise ValueError(
            f"Invalid Rakuten split {split!r}. "
            f"Expected one of {sorted(VALID_SPLITS)}."
        )

    return split


def build_record(
    title: str,
    category_id_path: str,
    split: str,
) -> RakutenRecord:
    """
    Construct a validated logical Rakuten record.

    No source values are fabricated or renamed.
    """
    if not isinstance(title, str):
        raise TypeError("title must be a string")

    if not title.strip():
        raise ValueError("title cannot be empty")

    validate_split(split)

    category_path = parse_category_path(category_id_path)

    return RakutenRecord(
        title=title,
        category_id_path=category_id_path,
        category_path=tuple(category_path),
        depth=len(category_path),
        split=split,
    )
