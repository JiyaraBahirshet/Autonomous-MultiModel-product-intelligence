import hashlib
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = ROOT / "data" / "processed"
SPLITS_DIR = ROOT / "data" / "splits"
REPORT_DIR = ROOT / "reports" / "preprocessing"

SEED = "phase-a-split-v001"


# ---------------------------------------------------------------------
# Deterministic split functions
# ---------------------------------------------------------------------

def bucket(key):
    digest = hashlib.sha256(
        f"{SEED}|{key}".encode("utf-8")
    ).hexdigest()[:16]

    return int(digest, 16) % 100000


def split(key):
    b = bucket(key)

    if b < 80000:
        return "train"

    if b < 90000:
        return "validation"

    return "test"


# ---------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------

def read_jsonl(path):
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            line = line.strip()

            if not line:
                continue

            record = json.loads(line)

            if not isinstance(record, dict):
                raise ValueError(
                    f"Invalid JSON object in {path} at line {line_number}"
                )

            yield record


def write_record(file, record):
    file.write(
        json.dumps(
            record,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n"
    )


# ---------------------------------------------------------------------
# Rakuten
# ---------------------------------------------------------------------

def rakuten():
    source_dir = PROCESSED_DIR / "rakuten"
    output_dir = SPLITS_DIR / "rakuten"

    output_dir.mkdir(parents=True, exist_ok=True)

    train_count = 0
    validation_count = 0
    test_count = 0

    # The original Phase A v001 logic uses a 90/10 split
    # on the official training data.
    with (
        (output_dir / "train.jsonl").open(
            "w", encoding="utf-8", newline="\n"
        ) as train_file,
        (output_dir / "validation.jsonl").open(
            "w", encoding="utf-8", newline="\n"
        ) as validation_file,
    ):
        for record in read_jsonl(source_dir / "train.jsonl"):
            if bucket(record["record_id"]) < 90000:
                write_record(train_file, record)
                train_count += 1
            else:
                write_record(validation_file, record)
                validation_count += 1

    # Official Rakuten test remains unchanged.
    with (
        (output_dir / "test.jsonl").open(
            "w", encoding="utf-8", newline="\n"
        ) as test_file
    ):
        for record in read_jsonl(source_dir / "test.jsonl"):
            write_record(test_file, record)
            test_count += 1

    return {
        "train": train_count,
        "validation": validation_count,
        "test": test_count,
    }


# ---------------------------------------------------------------------
# ABO image connectivity
# ---------------------------------------------------------------------

def image_ids(record):
    ids = set()

    main_image = record.get("main_image")

    if isinstance(main_image, dict):
        image_id = main_image.get("image_id")

        if image_id:
            ids.add(str(image_id))

    other_images = record.get("other_images") or []

    for image in other_images:
        if isinstance(image, dict):
            image_id = image.get("image_id")

            if image_id:
                ids.add(str(image_id))

    return ids


class DisjointSet:
    """
    Union-Find / DSU structure.

    Used to construct the exact connected components required
    for the ABO leakage-safe split.
    """

    def __init__(self):
        self.parent = []
        self.size = []

    def add(self):
        index = len(self.parent)

        self.parent.append(index)
        self.size.append(1)

        return index

    def find(self, index):
        parent = self.parent

        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]

        return index

    def union(self, a, b):
        root_a = self.find(a)
        root_b = self.find(b)

        if root_a == root_b:
            return

        # Union by size.
        if self.size[root_a] < self.size[root_b]:
            root_a, root_b = root_b, root_a

        self.parent[root_b] = root_a
        self.size[root_a] += self.size[root_b]


def abo():
    source_path = PROCESSED_DIR / "abo" / "listings.jsonl"
    output_dir = SPLITS_DIR / "abo"

    output_dir.mkdir(parents=True, exist_ok=True)

    dsu = DisjointSet()

    # image_id -> first record index containing that image
    image_owner = {}

    # Only record IDs are retained, not complete JSON records.
    record_ids = []

    # -------------------------------------------------------------
    # PASS 1
    #
    # Build connected components using shared image IDs.
    # -------------------------------------------------------------

    for record in read_jsonl(source_path):
        index = dsu.add()

        record_id = str(record.get("record_id", index))
        record_ids.append(record_id)

        for image_id in image_ids(record):
            previous = image_owner.get(image_id)

            if previous is None:
                image_owner[image_id] = index
            else:
                dsu.union(index, previous)

    # -------------------------------------------------------------
    # Build component membership.
    #
    # This retains only integer record indices, not full records.
    # -------------------------------------------------------------

    components = defaultdict(list)

    for index in range(len(record_ids)):
        root = dsu.find(index)
        components[root].append(index)

    # -------------------------------------------------------------
    # Assign one split to every connected component.
    #
    # IMPORTANT:
    # This preserves the original v001 component key exactly:
    #
    # component|<sorted record IDs>
    # -------------------------------------------------------------

    component_split = {}

    for root, indices in components.items():
        component_record_ids = sorted(
            record_ids[index]
            for index in indices
        )

        component_key = (
            "component|"
            + "|".join(component_record_ids)
        )

        component_split[root] = split(component_key)

    # -------------------------------------------------------------
    # PASS 2
    #
    # Re-read source and stream records directly to output.
    # No complete dataset is held in memory.
    # -------------------------------------------------------------

    output_files = {
        "train": (output_dir / "train.jsonl").open(
            "w", encoding="utf-8", newline="\n"
        ),
        "validation": (output_dir / "validation.jsonl").open(
            "w", encoding="utf-8", newline="\n"
        ),
        "test": (output_dir / "test.jsonl").open(
            "w", encoding="utf-8", newline="\n"
        ),
    }

    counts = {
        "train": 0,
        "validation": 0,
        "test": 0,
    }

    try:
        for index, record in enumerate(
            read_jsonl(source_path)
        ):
            root = dsu.find(index)
            assigned_split = component_split[root]

            write_record(
                output_files[assigned_split],
                record,
            )

            counts[assigned_split] += 1

    finally:
        for file in output_files.values():
            file.close()

    return counts


# ---------------------------------------------------------------------
# MAVE
# ---------------------------------------------------------------------

def mave():
    source_path = PROCESSED_DIR / "mave" / "listings.jsonl"
    output_dir = SPLITS_DIR / "mave"

    output_dir.mkdir(parents=True, exist_ok=True)

    output_files = {
        "train": (output_dir / "train.jsonl").open(
            "w", encoding="utf-8", newline="\n"
        ),
        "validation": (output_dir / "validation.jsonl").open(
            "w", encoding="utf-8", newline="\n"
        ),
        "test": (output_dir / "test.jsonl").open(
            "w", encoding="utf-8", newline="\n"
        ),
    }

    counts = {
        "train": 0,
        "validation": 0,
        "test": 0,
    }

    try:
        for record in read_jsonl(source_path):
            key = "asin|" + str(record["asin"])
            assigned_split = split(key)

            write_record(
                output_files[assigned_split],
                record,
            )

            counts[assigned_split] += 1

    finally:
        for file in output_files.values():
            file.close()

    return counts


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    print("=" * 70)
    print("PHASE A SPLIT GENERATION")
    print("=" * 70)
    print(f"Seed: {SEED}")
    print()

    SPLITS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Generating Rakuten splits...")
    rakuten_counts = rakuten()
    print(json.dumps(rakuten_counts, indent=2))
    print()

    print("Generating ABO leakage-safe splits...")
    abo_counts = abo()
    print(json.dumps(abo_counts, indent=2))
    print()

    print("Generating MAVE splits...")
    mave_counts = mave()
    print(json.dumps(mave_counts, indent=2))
    print()

    counts = {
        "rakuten": rakuten_counts,
        "abo": abo_counts,
        "mave": mave_counts,
    }

    report = {
        "generator": "phase_a_split_generation",
        "version": "v002",
        "seed": SEED,
        "counts": counts,
        "source_modified": False,
        "implementation": {
            "rakuten": (
                "streaming 90/10 split of official training data; "
                "official test copied unchanged"
            ),
            "abo": (
                "image-connected components using union-find; "
                "component split key preserves v001 semantics"
            ),
            "mave": (
                "streaming deterministic ASIN-based 80/10/10 split"
            ),
            "memory_strategy": (
                "stream records; do not retain complete JSONL datasets "
                "in memory"
            ),
        },
    }

    report_path = (
        REPORT_DIR
        / "phase_a_split_generation_v002.json"
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("=" * 70)
    print("SPLIT GENERATION COMPLETE")
    print("=" * 70)
    print(
        f"Report written to: {report_path}"
    )


if __name__ == "__main__":
    main()