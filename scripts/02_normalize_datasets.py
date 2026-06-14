"""
02_normalize_datasets.py

Bu dosyanın amacı:
- 01_download_datasets.py ile indirilen ham OpenBookQA ve ARC dosyalarını okumak
- OpenBookQA ve ARC veri formatlarını ortak bir yapıya dönüştürmek
- Sadece 4 seçenekli soruları almak
- Seçenekleri A/B/C/D formatına çevirmek
- Normalize edilmiş dosyaları data/processed/normalized/ altına kaydetmek

Bu aşamada:
- SFT prompt'u üretmiyoruz
- Teacher model çalıştırmıyoruz
- GRPO train/test split ayırmıyoruz
- Benzer soru üretmiyoruz
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


LETTERS = ["A", "B", "C", "D"]


def read_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """
    JSONL dosyasını okur.

    JSONL formatında her satır ayrı bir JSON nesnesidir.
    Bu fonksiyon dosyadaki tüm satırları okuyup Python listesi olarak döndürür.
    """

    examples = []

    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            examples.append(json.loads(line))

    return examples


def write_jsonl(examples: List[Dict[str, Any]], file_path: Path) -> None:
    """
    Python dictionary listesini JSONL dosyasına kaydeder.
    """

    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open("w", encoding="utf-8") as f:
        for example in examples:
            line = json.dumps(example, ensure_ascii=False)
            f.write(line + "\n")


def normalize_choices(
    choices: Dict[str, Any],
    answer_key: str
) -> Optional[Tuple[Dict[str, str], str]]:
    """
    Veri setinden gelen choices alanını standart A/B/C/D formatına çevirir.

    OpenBookQA ve ARC genellikle choices alanını şöyle verir:

    {
        "text": ["choice 1", "choice 2", "choice 3", "choice 4"],
        "label": ["A", "B", "C", "D"]
    }

    Bazı ARC örneklerinde label şu şekilde olabilir:

    {
        "text": ["choice 1", "choice 2", "choice 3", "choice 4"],
        "label": ["1", "2", "3", "4"]
    }

    Biz bunları her zaman şuna çeviriyoruz:

    {
        "A": "choice 1",
        "B": "choice 2",
        "C": "choice 3",
        "D": "choice 4"
    }

    answer_key de buna göre dönüştürülür.
    Örneğin:
    label = ["1", "2", "3", "4"]
    answer_key = "3"

    Bu durumda normalize edilmiş cevap "C" olur.

    Eğer örnek 4 seçenekli değilse veya cevap bulunamazsa None döndürür.
    """

    if not isinstance(choices, dict):
        return None

    labels = choices.get("label")
    texts = choices.get("text")

    if labels is None or texts is None:
        return None

    if len(labels) != len(texts):
        return None

    # Bu projede parser ve reward fonksiyonları sade olsun diye
    # yalnızca 4 seçenekli soruları kullanıyoruz.
    if len(labels) != 4:
        return None

    clean_labels = [str(label).strip() for label in labels]
    clean_texts = [str(text).strip() for text in texts]

    # Boş seçenek varsa örneği atıyoruz.
    if any(text == "" for text in clean_texts):
        return None

    # Aynı seçenek metni tekrar ediyorsa örneği atıyoruz.
    # Bu şart zorunlu değil ama veri kalitesini artırır.
    if len(set(clean_texts)) != len(clean_texts):
        return None

    normalized_choices = {}

    for new_label, choice_text in zip(LETTERS, clean_texts):
        normalized_choices[new_label] = choice_text

    # Eski label'dan yeni A/B/C/D label'ına dönüşüm tablosu.
    # Örneğin:
    # "1" -> "A"
    # "2" -> "B"
    # "3" -> "C"
    # "4" -> "D"
    old_to_new_label = {}

    for old_label, new_label in zip(clean_labels, LETTERS):
        old_to_new_label[old_label] = new_label

    answer_key = str(answer_key).strip()

    if answer_key not in old_to_new_label:
        return None

    normalized_answer = old_to_new_label[answer_key]

    return normalized_choices, normalized_answer


def normalize_openbookqa_example(
    example: Dict[str, Any],
    split: str,
    index: int
) -> Optional[Dict[str, Any]]:
    """
    Bir OpenBookQA örneğini ortak formata çevirir.

    OpenBookQA alanları genellikle:
    - id
    - question_stem
    - choices
    - answerKey
    """

    question = example.get("question_stem")
    choices = example.get("choices")
    answer_key = example.get("answerKey")

    if question is None or choices is None or answer_key is None:
        return None

    question = str(question).strip()

    if question == "":
        return None

    normalized = normalize_choices(choices, answer_key)

    if normalized is None:
        return None

    normalized_choices, normalized_answer = normalized

    normalized_example = {
        "id": f"openbookqa_{split}_{index:06d}",
        "source": "openbookqa",
        "subset": "main",
        "split": split,
        "question": question,
        "choices": normalized_choices,
        "answer": normalized_answer,
        "original_id": example.get("id"),
        "original_answer_key": str(answer_key).strip()
    }

    return normalized_example


def normalize_arc_example(
    example: Dict[str, Any],
    subset: str,
    split: str,
    index: int
) -> Optional[Dict[str, Any]]:
    """
    Bir ARC örneğini ortak formata çevirir.

    ARC alanları genellikle:
    - id
    - question
    - choices
    - answerKey

    subset:
    - ARC-Easy
    - ARC-Challenge
    """

    question = example.get("question")
    choices = example.get("choices")
    answer_key = example.get("answerKey")

    if question is None or choices is None or answer_key is None:
        return None

    question = str(question).strip()

    if question == "":
        return None

    normalized = normalize_choices(choices, answer_key)

    if normalized is None:
        return None

    normalized_choices, normalized_answer = normalized

    safe_subset_name = subset.lower().replace("-", "_")

    normalized_example = {
        "id": f"arc_{safe_subset_name}_{split}_{index:06d}",
        "source": "arc",
        "subset": subset,
        "split": split,
        "question": question,
        "choices": normalized_choices,
        "answer": normalized_answer,
        "original_id": example.get("id"),
        "original_answer_key": str(answer_key).strip()
    }

    return normalized_example


def process_openbookqa_split(
    raw_dir: Path,
    output_dir: Path,
    split: str
) -> Dict[str, Any]:
    """
    OpenBookQA'nın belirli bir split dosyasını normalize eder.

    Örneğin:
    data/raw/openbookqa/main/train.jsonl
    dosyasını okuyup
    data/processed/normalized/openbookqa_train.jsonl
    dosyasına yazar.
    """

    input_path = raw_dir / "openbookqa" / "main" / f"{split}.jsonl"
    output_path = output_dir / f"openbookqa_{split}.jsonl"

    print(f"\n[INFO] OpenBookQA {split} okunuyor: {input_path}")

    raw_examples = read_jsonl(input_path)
    normalized_examples = []

    skipped = 0

    for i, example in enumerate(raw_examples):
        normalized = normalize_openbookqa_example(
            example=example,
            split=split,
            index=i
        )

        if normalized is None:
            skipped += 1
            continue

        normalized_examples.append(normalized)

    write_jsonl(normalized_examples, output_path)

    print(f"[OK] Kaydedildi: {output_path}")
    print(f"[INFO] Toplam ham örnek: {len(raw_examples)}")
    print(f"[INFO] Normalize edilen: {len(normalized_examples)}")
    print(f"[INFO] Atlanan: {skipped}")

    return {
        "dataset": "openbookqa",
        "subset": "main",
        "split": split,
        "input_file": str(input_path),
        "output_file": str(output_path),
        "raw_count": len(raw_examples),
        "normalized_count": len(normalized_examples),
        "skipped_count": skipped
    }


def process_arc_split(
    raw_dir: Path,
    output_dir: Path,
    subset: str,
    split: str
) -> Dict[str, Any]:
    """
    ARC'nin belirli bir subset/split dosyasını normalize eder.

    Örneğin:
    data/raw/arc/ARC-Easy/train.jsonl
    dosyasını okuyup
    data/processed/normalized/arc_ARC-Easy_train.jsonl
    dosyasına yazar.
    """

    input_path = raw_dir / "arc" / subset / f"{split}.jsonl"

    safe_subset_name = subset.lower().replace("-", "_")
    output_path = output_dir / f"arc_{safe_subset_name}_{split}.jsonl"

    print(f"\n[INFO] ARC {subset} {split} okunuyor: {input_path}")

    raw_examples = read_jsonl(input_path)
    normalized_examples = []

    skipped = 0

    for i, example in enumerate(raw_examples):
        normalized = normalize_arc_example(
            example=example,
            subset=subset,
            split=split,
            index=i
        )

        if normalized is None:
            skipped += 1
            continue

        normalized_examples.append(normalized)

    write_jsonl(normalized_examples, output_path)

    print(f"[OK] Kaydedildi: {output_path}")
    print(f"[INFO] Toplam ham örnek: {len(raw_examples)}")
    print(f"[INFO] Normalize edilen: {len(normalized_examples)}")
    print(f"[INFO] Atlanan: {skipped}")

    return {
        "dataset": "arc",
        "subset": subset,
        "split": split,
        "input_file": str(input_path),
        "output_file": str(output_path),
        "raw_count": len(raw_examples),
        "normalized_count": len(normalized_examples),
        "skipped_count": skipped
    }


def save_manifest(records: List[Dict[str, Any]], output_dir: Path) -> None:
    """
    Normalize edilen dosyalar için özet manifest kaydeder.
    """

    manifest = {
        "description": "Normalized multiple-choice datasets for GRPO MCQ project.",
        "format": {
            "id": "unique normalized id",
            "source": "openbookqa or arc",
            "subset": "main, ARC-Easy, ARC-Challenge",
            "split": "train, validation, test",
            "question": "question text",
            "choices": {
                "A": "choice text",
                "B": "choice text",
                "C": "choice text",
                "D": "choice text"
            },
            "answer": "A/B/C/D",
            "original_id": "id from original dataset",
            "original_answer_key": "answer key from original dataset"
        },
        "files": records
    }

    manifest_path = output_dir / "normalization_manifest.json"

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] Normalization manifest kaydedildi: {manifest_path}")


def print_example(output_dir: Path) -> None:
    """
    Kontrol için normalize edilmiş OpenBookQA train dosyasından bir örnek yazdırır.
    """

    example_path = output_dir / "openbookqa_train.jsonl"

    if not example_path.exists():
        print("[WARNING] Örnek dosya bulunamadı.")
        return

    with example_path.open("r", encoding="utf-8") as f:
        first_line = f.readline().strip()

    if not first_line:
        print("[WARNING] Örnek dosya boş.")
        return

    example = json.loads(first_line)

    print("\n[INFO] Normalize edilmiş örnek:")
    print(json.dumps(example, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    """
    Komut satırı argümanlarını okur.

    Varsayılan kullanım:
    python scripts/02_normalize_datasets.py

    Farklı klasörlerle kullanım:
    python scripts/02_normalize_datasets.py --raw_dir data/raw --output_dir data/processed/normalized
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--raw_dir",
        type=str,
        default="data/raw",
        help="Ham JSONL dosyalarının bulunduğu klasör."
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed/normalized",
        help="Normalize edilmiş JSONL dosyalarının kaydedileceği klasör."
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Normalize işlemi başlıyor...")
    print(f"[INFO] Ham veri klasörü: {raw_dir}")
    print(f"[INFO] Çıktı klasörü: {output_dir}")

    records = []

    # OpenBookQA splitleri
    for split in ["train", "validation", "test"]:
        record = process_openbookqa_split(
            raw_dir=raw_dir,
            output_dir=output_dir,
            split=split
        )
        records.append(record)

    # ARC subset ve splitleri
    for subset in ["ARC-Easy", "ARC-Challenge"]:
        for split in ["train", "validation", "test"]:
            record = process_arc_split(
                raw_dir=raw_dir,
                output_dir=output_dir,
                subset=subset,
                split=split
            )
            records.append(record)

    save_manifest(records, output_dir)
    print_example(output_dir)

    print("\n[DONE] Tüm veri setleri ortak formata normalize edildi.")


if __name__ == "__main__":
    main()