"""
07a_extract_single_from_pairs.py

Purpose:
- Extract Question 1 fields from paired datasets.
- Produce single-question JSONL files for mA.

Supported usage patterns:

1) Single file mode:
python scripts/07a_extract_single_from_pairs.py \
  --input_file data/processed/final/openbook_train_paired_shuffled.jsonl \
  --output_file data/processed/final/openbook_train_993.jsonl

2) Train + test mode:
python scripts/07a_extract_single_from_pairs.py \
  --input_train data/processed/final/openbook_train_paired_shuffled.jsonl \
  --output_train data/processed/final/openbook_train_993.jsonl \
  --input_test data/processed/final/openbookqa_test_paired_shuffled.jsonl \
  --output_test data/processed/final/openbook_test_493.jsonl
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def extract_single_questions(input_file: Path, output_file: Path, tag: str = "dataset") -> Dict[str, Any]:
    print(f"\n[INFO] Extracting single-question data: {tag}")
    print(f"[INFO] Paired input: {input_file}")
    print(f"[INFO] Output      : {output_file}")

    data = read_jsonl(input_file)
    single_rows = []

    for item in data:
        single_rows.append(
            {
                "id": item.get("id", ""),
                "question": item.get("question_1"),
                "choices": item.get("choices_1"),
                "answer": item.get("answer_1"),
            }
        )

    write_jsonl(single_rows, output_file)
    print(f"[DONE] {len(single_rows)} single-question rows saved: {output_file}")

    return {
        "tag": tag,
        "input_rows": len(data),
        "output_rows": len(single_rows),
        "output_file": str(output_file),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # Single-file mode.
    parser.add_argument("--input_file", type=str, default=None)
    parser.add_argument("--output_file", type=str, default=None)

    # Train + test mode.
    parser.add_argument("--input_train", type=str, default=None)
    parser.add_argument("--output_train", type=str, default=None)
    parser.add_argument("--input_test", type=str, default=None)
    parser.add_argument("--output_test", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = []

    single_mode = args.input_file and args.output_file
    train_mode = args.input_train and args.output_train
    test_mode = args.input_test and args.output_test

    if single_mode:
        summaries.append(extract_single_questions(Path(args.input_file), Path(args.output_file), tag="single"))

    if train_mode:
        summaries.append(extract_single_questions(Path(args.input_train), Path(args.output_train), tag="train"))

    if test_mode:
        summaries.append(extract_single_questions(Path(args.input_test), Path(args.output_test), tag="test"))

    if not summaries:
        raise SystemExit(
            "No valid input set was provided. Use either --input_file/--output_file "
            "or train/test arguments such as --input_train/--output_train."
        )

    print("\n[SUMMARY]")
    for item in summaries:
        print(json.dumps(item, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
