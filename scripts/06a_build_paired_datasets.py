"""
06a_build_paired_datasets.py

Purpose:
- Merge original OpenBookQA files with teacher-generated similar-question files by id.
- Filter invalid/missing similar records.
- Shuffle Question 2 choices and update answer_2 accordingly.
- Produce paired JSONL files for mB/mC/mD.

Supported usage patterns:

1) Single file mode:
python scripts/06a_build_paired_datasets.py \
  --original_file data/processed/splits/openbookqa_grpo_train_1000.jsonl \
  --similar_file data/teacher_outputs/openbookqa_grpo_train_1000_similar.jsonl \
  --output_file data/processed/final/openbook_train_paired_shuffled.jsonl \
  --seed 42

2) Train + test mode:
python scripts/06a_build_paired_datasets.py \
  --original_train data/processed/splits/openbookqa_grpo_train_1000.jsonl \
  --similar_train data/teacher_outputs/openbookqa_grpo_train_1000_similar.jsonl \
  --output_train data/processed/final/openbook_train_paired_shuffled.jsonl \
  --original_test data/processed/splits/openbookqa_test_500.jsonl \
  --similar_test data/teacher_outputs/openbookqa_test_500_similar.jsonl \
  --output_test data/processed/final/openbookqa_test_paired_shuffled.jsonl \
  --seed 42
"""

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

VALID_ANSWERS = {"A", "B", "C", "D"}
LETTERS = ["A", "B", "C", "D"]


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


def normalize_choices(choices: Any) -> Optional[Dict[str, str]]:
    if not isinstance(choices, dict):
        return None
    out: Dict[str, str] = {}
    for label in LETTERS:
        if label not in choices:
            return None
        text = str(choices[label]).strip()
        if not text:
            return None
        out[label] = text
    if len(set(out.values())) != 4:
        return None
    return out


def build_paired_record(original: Dict[str, Any], similar: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    q1 = original.get("question") or original.get("question_1") or original.get("original_question")
    c1 = normalize_choices(original.get("choices") or original.get("choices_1") or original.get("original_choices"))
    a1 = str(original.get("answer") or original.get("answer_1") or original.get("original_answer") or "").strip().upper()

    q2 = similar.get("similar_question") or similar.get("question_2")
    c2 = normalize_choices(similar.get("similar_choices") or similar.get("choices_2"))
    a2 = str(similar.get("similar_answer") or similar.get("answer_2") or "").strip().upper()

    if not q1 or c1 is None or a1 not in VALID_ANSWERS:
        return None, "invalid_original"
    if not q2 or c2 is None or a2 not in VALID_ANSWERS:
        return None, "invalid_similar"

    return {
        "id": original.get("id", similar.get("id", "")),
        "question_1": str(q1).strip(),
        "choices_1": c1,
        "answer_1": a1,
        "question_2": str(q2).strip(),
        "choices_2": c2,
        "answer_2": a2,
    }, None


def shuffle_question2_choices(record: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    record = dict(record)
    choices_2 = dict(record["choices_2"])
    answer_2 = record["answer_2"]
    correct_text = choices_2[answer_2]

    choice_texts = list(choices_2.values())
    rng.shuffle(choice_texts)

    new_choices: Dict[str, str] = {}
    new_answer = None
    for label, text in zip(LETTERS, choice_texts):
        new_choices[label] = text
        if text == correct_text:
            new_answer = label

    if new_answer is None:
        raise ValueError("Correct answer text could not be found after shuffling.")

    record["choices_2"] = new_choices
    record["answer_2"] = new_answer
    return record


def build_paired_dataset(
    original_file: Path,
    similar_file: Path,
    output_file: Path,
    seed: int = 42,
    shuffle: bool = True,
    tag: str = "dataset",
) -> Dict[str, Any]:
    print(f"\n[INFO] Building paired dataset: {tag}")
    print(f"[INFO] Original: {original_file}")
    print(f"[INFO] Similar : {similar_file}")
    print(f"[INFO] Output  : {output_file}")

    originals = {row["id"]: row for row in read_jsonl(original_file)}
    similar_rows = read_jsonl(similar_file)

    rng = random.Random(seed)
    paired_rows: List[Dict[str, Any]] = []
    invalid_counts: Dict[str, int] = {}
    missing_original = 0
    same_answer_before = 0
    same_answer_after = 0

    for sim in similar_rows:
        sid = sim.get("id")
        if sid not in originals:
            missing_original += 1
            continue

        rec, reason = build_paired_record(originals[sid], sim)
        if rec is None:
            invalid_counts[reason or "invalid"] = invalid_counts.get(reason or "invalid", 0) + 1
            continue

        if rec["answer_1"] == rec["answer_2"]:
            same_answer_before += 1

        if shuffle:
            rec = shuffle_question2_choices(rec, rng)

        if rec["answer_1"] == rec["answer_2"]:
            same_answer_after += 1

        paired_rows.append(rec)

    write_jsonl(paired_rows, output_file)

    total_similar = len(similar_rows)
    deleted = total_similar - len(paired_rows)
    summary = {
        "tag": tag,
        "similar_input_rows": total_similar,
        "final_paired_rows": len(paired_rows),
        "removed_or_invalid_rows": deleted,
        "missing_original_rows": missing_original,
        "invalid_counts": invalid_counts,
        "same_answer_before_shuffle": same_answer_before,
        "same_answer_after_shuffle": same_answer_after,
        "output_file": str(output_file),
    }

    print("[DONE] Paired dataset saved.")
    print(f"[INFO] Similar input rows       : {total_similar}")
    print(f"[INFO] Final paired rows        : {len(paired_rows)}")
    print(f"[INFO] Removed/invalid rows     : {deleted}")
    print(f"[INFO] Missing original rows    : {missing_original}")
    print(f"[INFO] Invalid reasons          : {invalid_counts}")
    if paired_rows:
        print(f"[INFO] Same answer letter before: {same_answer_before} (%{same_answer_before / len(paired_rows) * 100:.1f})")
        print(f"[INFO] Same answer letter after : {same_answer_after} (%{same_answer_after / len(paired_rows) * 100:.1f})")

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # Single-file mode.
    parser.add_argument("--original_file", type=str, default=None)
    parser.add_argument("--similar_file", type=str, default=None)
    parser.add_argument("--output_file", type=str, default=None)

    # Train + test mode.
    parser.add_argument("--original_train", type=str, default=None)
    parser.add_argument("--similar_train", type=str, default=None)
    parser.add_argument("--output_train", type=str, default=None)
    parser.add_argument("--original_test", type=str, default=None)
    parser.add_argument("--similar_test", type=str, default=None)
    parser.add_argument("--output_test", type=str, default=None)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_shuffle", action="store_true", help="Do not shuffle Question 2 choices.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shuffle = not args.no_shuffle

    summaries = []

    single_mode = args.original_file and args.similar_file and args.output_file
    train_mode = args.original_train and args.similar_train and args.output_train
    test_mode = args.original_test and args.similar_test and args.output_test

    if single_mode:
        summaries.append(
            build_paired_dataset(
                Path(args.original_file),
                Path(args.similar_file),
                Path(args.output_file),
                seed=args.seed,
                shuffle=shuffle,
                tag="single",
            )
        )

    if train_mode:
        summaries.append(
            build_paired_dataset(
                Path(args.original_train),
                Path(args.similar_train),
                Path(args.output_train),
                seed=args.seed,
                shuffle=shuffle,
                tag="train",
            )
        )

    if test_mode:
        summaries.append(
            build_paired_dataset(
                Path(args.original_test),
                Path(args.similar_test),
                Path(args.output_test),
                seed=args.seed,
                shuffle=shuffle,
                tag="test",
            )
        )

    if not summaries:
        raise SystemExit(
            "No valid input set was provided. Use either --original_file/--similar_file/--output_file "
            "or train/test arguments such as --original_train/--similar_train/--output_train."
        )

    print("\n[SUMMARY]")
    for item in summaries:
        print(json.dumps(item, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
