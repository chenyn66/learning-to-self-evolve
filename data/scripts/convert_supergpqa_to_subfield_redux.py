import argparse
import os
import re
from collections import Counter, defaultdict

import datasets
from datasets import Dataset, DatasetDict


def _slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def _extract_uuid(source: str) -> str | None:
    if not source:
        return None
    prefix = "SuperGPQA-"
    if source.startswith(prefix):
        return source[len(prefix):]
    return None


def _write_info(path: str, counts: dict[str, int]) -> None:
    total = sum(counts.values())
    lines = [f"Total size: {total}"]
    for name, size in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        lines.append(f"{name}: {size}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Group SuperGPQA-Redux by subfield.")
    parser.add_argument(
        "--input-path",
        default="data/supergpqa-redux",
        help="Path to existing SuperGPQA-Redux dataset.",
    )
    parser.add_argument(
        "--output-path",
        default="data/supergpqa-redux-subfield",
        help="Path to save the subfield-grouped dataset.",
    )
    parser.add_argument(
        "--min-subfield-size",
        type=int,
        default=100,
        help="Minimum size to keep a subfield group.",
    )
    parser.add_argument(
        "--source-dataset",
        default="m-a-p/SuperGPQA",
        help="Source dataset containing field/subfield metadata.",
    )
    parser.add_argument(
        "--source-split",
        default="train",
        help="Split to load from the source dataset.",
    )
    args = parser.parse_args()

    if os.path.exists(args.output_path):
        raise FileExistsError(
            f"Output path already exists: {args.output_path}. "
            "Remove it or choose a new path."
        )

    base_dataset = datasets.load_from_disk(args.input_path)
    source_dataset = datasets.load_dataset(args.source_dataset, split=args.source_split)

    uuid_to_meta = {}
    for item in source_dataset:
        uuid_to_meta[item["uuid"]] = (item["field"], item["subfield"])

    subfield_counts = Counter()
    missing_meta = 0
    for split in base_dataset.keys():
        for item in base_dataset[split]:
            uuid = _extract_uuid(item.get("source", ""))
            if not uuid or uuid not in uuid_to_meta:
                missing_meta += 1
                continue
            _, subfield = uuid_to_meta[uuid]
            subfield_counts[subfield] += 1

    large_subfields = {s for s, n in subfield_counts.items() if n >= args.min_subfield_size}

    grouped_items: dict[str, list[dict]] = defaultdict(list)
    missing_meta_second_pass = 0
    for split in base_dataset.keys():
        for item in base_dataset[split]:
            uuid = _extract_uuid(item.get("source", ""))
            if not uuid or uuid not in uuid_to_meta:
                missing_meta_second_pass += 1
                continue
            field, subfield = uuid_to_meta[uuid]
            if subfield in large_subfields:
                group_name = _slugify(subfield)
            else:
                group_name = _slugify(field)
            grouped_items[group_name].append(item)

    if missing_meta or missing_meta_second_pass:
        print(f"Warning: missing metadata for {missing_meta_second_pass} items.")

    features = next(iter(base_dataset.values())).features
    ds_dict = {
        group: Dataset.from_list(items, features=features)
        for group, items in grouped_items.items()
    }
    final_dataset = DatasetDict(ds_dict)

    final_dataset.save_to_disk(args.output_path)

    info_path = os.path.join(args.output_path, "info.txt")
    group_sizes = {group: len(items) for group, items in grouped_items.items()}
    _write_info(info_path, group_sizes)

    print(f"Saved {len(final_dataset)} groups to {args.output_path}")


if __name__ == "__main__":
    main()
