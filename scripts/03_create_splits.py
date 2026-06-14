"""
03_create_splits.py

Bu dosyanın amacı:
- 02_normalize_datasets.py ile oluşturulan normalize veri dosyalarını okumak
- OpenBookQA train içinden SFT ve GRPO için ayrık kümeler oluşturmak
- ARC train verilerini SFT havuzuna eklemek
- OpenBookQA test split'ini yalnızca final test için ayırmak
- Split dosyalarını data/processed/splits/ altına kaydetmek

Önemli:
OpenBookQA test verisi eğitimde kullanılmaz.
Yalnızca final değerlendirme için saklanır.
"""

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List


def read_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """
    JSONL dosyasını okuyup Python listesi olarak döndürür.
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
    Python dictionary listesini JSONL formatında kaydeder.
    """

    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open("w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")


def add_split_role(
    examples: List[Dict[str, Any]],
    split_role: str
) -> List[Dict[str, Any]]:
    """
    Her örneğe deney içindeki rolünü ekler.

    Örneğin:
    - sft_train
    - grpo_train
    - final_test
    - dev
    - unused

    Böylece daha sonra dosyaya bakınca bu örneğin nerede kullanıldığını anlayabiliriz.
    """

    updated_examples = []

    for example in examples:
        new_example = dict(example)
        new_example["split_role"] = split_role
        updated_examples.append(new_example)

    return updated_examples


def check_no_overlap(
    first_examples: List[Dict[str, Any]],
    second_examples: List[Dict[str, Any]],
    first_name: str,
    second_name: str
) -> None:
    """
    İki veri kümesinde aynı id var mı diye kontrol eder.

    Bu özellikle SFT ve GRPO splitlerinin ayrık olduğundan emin olmak için önemli.
    """

    first_ids = set(example["id"] for example in first_examples)
    second_ids = set(example["id"] for example in second_examples)

    overlap = first_ids.intersection(second_ids)

    if overlap:
        raise ValueError(
            f"{first_name} ve {second_name} arasında çakışma var! "
            f"Çakışan örnek sayısı: {len(overlap)}"
        )

    print(f"[OK] {first_name} ve {second_name} arasında çakışma yok.")


def create_openbookqa_splits(
    normalized_dir: Path,
    output_dir: Path,
    obqa_sft_size: int,
    obqa_grpo_size: int,
    seed: int
) -> Dict[str, Any]:
    """
    OpenBookQA train verisini SFT ve GRPO için ayırır.

    OpenBookQA train yaklaşık 4957 örnektir.

    Varsayılan:
    - 3000 örnek SFT için
    - 1000 örnek GRPO için
    - kalan örnekler unused/dev olarak saklanır

    OpenBookQA test ise final test için ayrı tutulur.
    """

    train_path = normalized_dir / "openbookqa_train.jsonl"
    validation_path = normalized_dir / "openbookqa_validation.jsonl"
    test_path = normalized_dir / "openbookqa_test.jsonl"

    print("\n[INFO] OpenBookQA splitleri okunuyor...")

    obqa_train = read_jsonl(train_path)
    obqa_validation = read_jsonl(validation_path)
    obqa_test = read_jsonl(test_path)

    print(f"[INFO] OpenBookQA train: {len(obqa_train)}")
    print(f"[INFO] OpenBookQA validation: {len(obqa_validation)}")
    print(f"[INFO] OpenBookQA test: {len(obqa_test)}")

    required_train_size = obqa_sft_size + obqa_grpo_size

    if len(obqa_train) < required_train_size:
        raise ValueError(
            f"OpenBookQA train yeterli değil. "
            f"Gerekli: {required_train_size}, mevcut: {len(obqa_train)}"
        )

    rng = random.Random(seed)

    shuffled_train = list(obqa_train)
    rng.shuffle(shuffled_train)

    obqa_sft = shuffled_train[:obqa_sft_size]
    obqa_grpo = shuffled_train[obqa_sft_size:obqa_sft_size + obqa_grpo_size]
    obqa_unused = shuffled_train[obqa_sft_size + obqa_grpo_size:]

    obqa_sft = add_split_role(obqa_sft, "sft_train")
    obqa_grpo = add_split_role(obqa_grpo, "grpo_train")
    obqa_unused = add_split_role(obqa_unused, "unused_train")
    obqa_validation = add_split_role(obqa_validation, "dev")
    obqa_test = add_split_role(obqa_test, "final_test")

    check_no_overlap(obqa_sft, obqa_grpo, "OpenBookQA SFT", "OpenBookQA GRPO")
    check_no_overlap(obqa_sft, obqa_test, "OpenBookQA SFT", "OpenBookQA TEST")
    check_no_overlap(obqa_grpo, obqa_test, "OpenBookQA GRPO", "OpenBookQA TEST")

    output_paths = {
        "openbookqa_sft_train": output_dir / "openbookqa_sft_train.jsonl",
        "openbookqa_grpo_train_1000": output_dir / "openbookqa_grpo_train_1000.jsonl",
        "openbookqa_unused_train": output_dir / "openbookqa_unused_train.jsonl",
        "openbookqa_dev": output_dir / "openbookqa_dev.jsonl",
        "openbookqa_test_500": output_dir / "openbookqa_test_500.jsonl",
    }

    write_jsonl(obqa_sft, output_paths["openbookqa_sft_train"])
    write_jsonl(obqa_grpo, output_paths["openbookqa_grpo_train_1000"])
    write_jsonl(obqa_unused, output_paths["openbookqa_unused_train"])
    write_jsonl(obqa_validation, output_paths["openbookqa_dev"])
    write_jsonl(obqa_test, output_paths["openbookqa_test_500"])

    print("\n[OK] OpenBookQA splitleri kaydedildi:")
    for name, path in output_paths.items():
        print(f"  - {name}: {path}")

    return {
        "openbookqa_train_total": len(obqa_train),
        "openbookqa_sft_train": len(obqa_sft),
        "openbookqa_grpo_train": len(obqa_grpo),
        "openbookqa_unused_train": len(obqa_unused),
        "openbookqa_dev": len(obqa_validation),
        "openbookqa_test": len(obqa_test),
        "files": {name: str(path) for name, path in output_paths.items()}
    }


def create_arc_sft_split(
    normalized_dir: Path,
    output_dir: Path
) -> Dict[str, Any]:
    """
    ARC train splitlerini SFT için birleştirir.

    Kullanılan dosyalar:
    - arc_arc_easy_train.jsonl
    - arc_arc_challenge_train.jsonl

    ARC test verisini SFT'ye koymuyoruz.
    ARC validation verisini de varsayılan olarak SFT'ye koymuyoruz.
    Böylece daha temiz bir ayrım kalıyor.
    """

    arc_easy_train_path = normalized_dir / "arc_arc_easy_train.jsonl"
    arc_challenge_train_path = normalized_dir / "arc_arc_challenge_train.jsonl"

    arc_easy_validation_path = normalized_dir / "arc_arc_easy_validation.jsonl"
    arc_challenge_validation_path = normalized_dir / "arc_arc_challenge_validation.jsonl"

    print("\n[INFO] ARC train splitleri okunuyor...")

    arc_easy_train = read_jsonl(arc_easy_train_path)
    arc_challenge_train = read_jsonl(arc_challenge_train_path)

    arc_easy_validation = read_jsonl(arc_easy_validation_path)
    arc_challenge_validation = read_jsonl(arc_challenge_validation_path)

    arc_sft_train = arc_easy_train + arc_challenge_train
    arc_dev = arc_easy_validation + arc_challenge_validation

    arc_sft_train = add_split_role(arc_sft_train, "sft_train")
    arc_dev = add_split_role(arc_dev, "dev")

    output_paths = {
        "arc_sft_train": output_dir / "arc_sft_train.jsonl",
        "arc_dev": output_dir / "arc_dev.jsonl",
    }

    write_jsonl(arc_sft_train, output_paths["arc_sft_train"])
    write_jsonl(arc_dev, output_paths["arc_dev"])

    print(f"[OK] ARC SFT train kaydedildi: {output_paths['arc_sft_train']}")
    print(f"[OK] ARC dev kaydedildi: {output_paths['arc_dev']}")
    print(f"[INFO] ARC SFT train örnek sayısı: {len(arc_sft_train)}")
    print(f"[INFO] ARC dev örnek sayısı: {len(arc_dev)}")

    return {
        "arc_sft_train": len(arc_sft_train),
        "arc_dev": len(arc_dev),
        "files": {name: str(path) for name, path in output_paths.items()}
    }


def create_final_sft_pool(
    output_dir: Path
) -> Dict[str, Any]:
    """
    SFT için final eğitim havuzunu oluşturur.

    Final SFT train:
    - OpenBookQA SFT train
    - ARC SFT train

    Final SFT dev:
    - OpenBookQA dev
    - ARC dev

    Bu dosyalar teacher rationale üretiminden önceki soru havuzlarıdır.
    Teacher modeli daha sonra bu örnekler için açıklamalı cevap üretecek.
    """

    obqa_sft_path = output_dir / "openbookqa_sft_train.jsonl"
    arc_sft_path = output_dir / "arc_sft_train.jsonl"

    obqa_dev_path = output_dir / "openbookqa_dev.jsonl"
    arc_dev_path = output_dir / "arc_dev.jsonl"

    obqa_sft = read_jsonl(obqa_sft_path)
    arc_sft = read_jsonl(arc_sft_path)

    obqa_dev = read_jsonl(obqa_dev_path)
    arc_dev = read_jsonl(arc_dev_path)

    sft_train = obqa_sft + arc_sft
    sft_dev = obqa_dev + arc_dev

    sft_train_path = output_dir / "sft_train_pool.jsonl"
    sft_dev_path = output_dir / "sft_dev_pool.jsonl"

    write_jsonl(sft_train, sft_train_path)
    write_jsonl(sft_dev, sft_dev_path)

    print("\n[OK] Final SFT havuzları oluşturuldu:")
    print(f"  - Train: {sft_train_path} ({len(sft_train)} örnek)")
    print(f"  - Dev: {sft_dev_path} ({len(sft_dev)} örnek)")

    return {
        "sft_train_pool": len(sft_train),
        "sft_dev_pool": len(sft_dev),
        "files": {
            "sft_train_pool": str(sft_train_path),
            "sft_dev_pool": str(sft_dev_path)
        }
    }


def save_manifest(
    output_dir: Path,
    seed: int,
    obqa_sft_size: int,
    obqa_grpo_size: int,
    openbookqa_info: Dict[str, Any],
    arc_info: Dict[str, Any],
    sft_pool_info: Dict[str, Any]
) -> None:
    """
    Split işleminin özetini JSON olarak kaydeder.
    """

    manifest = {
        "description": "Dataset splits for GRPO multiple-choice QA project.",
        "seed": seed,
        "rules": {
            "openbookqa_test_usage": "final evaluation only; never used for training",
            "openbookqa_train_split": {
                "sft_train_size": obqa_sft_size,
                "grpo_train_size": obqa_grpo_size,
                "remaining": "unused_train"
            },
            "arc_usage": {
                "train": "used for SFT train pool",
                "validation": "used for SFT dev pool",
                "test": "not used in this stage"
            }
        },
        "openbookqa": openbookqa_info,
        "arc": arc_info,
        "sft_pool": sft_pool_info
    }

    manifest_path = output_dir / "split_manifest.json"

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] Split manifest kaydedildi: {manifest_path}")


def print_sample(file_path: Path, title: str) -> None:
    """
    Kontrol için bir dosyadan ilk örneği yazdırır.
    """

    if not file_path.exists():
        print(f"[WARNING] Dosya bulunamadı: {file_path}")
        return

    with file_path.open("r", encoding="utf-8") as f:
        line = f.readline().strip()

    if not line:
        print(f"[WARNING] Dosya boş: {file_path}")
        return

    example = json.loads(line)

    print(f"\n[INFO] Örnek - {title}:")
    print(json.dumps(example, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    """
    Komut satırı argümanlarını okur.

    Varsayılan kullanım:
    python scripts/03_create_splits.py
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--normalized_dir",
        type=str,
        default="data/processed/normalized",
        help="Normalize edilmiş veri dosyalarının bulunduğu klasör."
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed/splits",
        help="Split dosyalarının kaydedileceği klasör."
    )

    parser.add_argument(
        "--obqa_sft_size",
        type=int,
        default=3000,
        help="OpenBookQA train içinden SFT için ayrılacak örnek sayısı."
    )

    parser.add_argument(
        "--obqa_grpo_size",
        type=int,
        default=1000,
        help="OpenBookQA train içinden GRPO için ayrılacak örnek sayısı."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Rastgele karıştırma için seed değeri."
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    normalized_dir = Path(args.normalized_dir)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Split oluşturma işlemi başlıyor...")
    print(f"[INFO] Normalize veri klasörü: {normalized_dir}")
    print(f"[INFO] Çıktı klasörü: {output_dir}")
    print(f"[INFO] Seed: {args.seed}")

    openbookqa_info = create_openbookqa_splits(
        normalized_dir=normalized_dir,
        output_dir=output_dir,
        obqa_sft_size=args.obqa_sft_size,
        obqa_grpo_size=args.obqa_grpo_size,
        seed=args.seed
    )

    arc_info = create_arc_sft_split(
        normalized_dir=normalized_dir,
        output_dir=output_dir
    )

    sft_pool_info = create_final_sft_pool(
        output_dir=output_dir
    )

    save_manifest(
        output_dir=output_dir,
        seed=args.seed,
        obqa_sft_size=args.obqa_sft_size,
        obqa_grpo_size=args.obqa_grpo_size,
        openbookqa_info=openbookqa_info,
        arc_info=arc_info,
        sft_pool_info=sft_pool_info
    )

    print_sample(
        file_path=output_dir / "openbookqa_grpo_train_1000.jsonl",
        title="OpenBookQA GRPO train"
    )

    print_sample(
        file_path=output_dir / "sft_train_pool.jsonl",
        title="SFT train pool"
    )

    print("\n[DONE] Split işlemi başarıyla tamamlandı.")


if __name__ == "__main__":
    main()