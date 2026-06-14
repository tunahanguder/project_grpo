"""
01_download_datasets.py

Bu dosyanın amacı:
- OpenBookQA veri setini indirmek
- ARC-Easy veri setini indirmek
- ARC-Challenge veri setini indirmek
- Her split'i ayrı ayrı JSONL dosyası olarak data/raw/ altına kaydetmek

Bu aşamada veri temizleme, prompt oluşturma veya SFT formatına çevirme yapmıyoruz.
Sadece ham veriyi indirip düzenli şekilde saklıyoruz.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any

from datasets import load_dataset, DatasetDict


def save_jsonl(dataset_split, output_path: Path) -> None:
    """
    Hugging Face Dataset split'ini JSONL formatında kaydeder.

    JSONL ne demek?
    - Her satırda bir JSON nesnesi vardır.
    - Büyük veri dosyaları için kullanışlıdır.
    - Sonraki scriptlerde satır satır okuyabiliriz.

    Örnek satır:
    {"id": "...", "question": "...", "choices": {...}, "answerKey": "A"}
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for example in dataset_split:
            json_line = json.dumps(example, ensure_ascii=False)
            f.write(json_line + "\n")


def save_dataset_dict(
    dataset: DatasetDict,
    output_dir: Path,
    dataset_name: str,
    subset_name: str
) -> Dict[str, Any]:
    """
    Bir DatasetDict içindeki train/validation/test splitlerini kaydeder.

    Örneğin:
    data/raw/openbookqa/main/train.jsonl
    data/raw/openbookqa/main/validation.jsonl
    data/raw/openbookqa/main/test.jsonl
    """

    manifest_rows = []

    for split_name, split_data in dataset.items():
        output_path = output_dir / dataset_name / subset_name / f"{split_name}.jsonl"

        print(f"[INFO] Kaydediliyor: {output_path}")
        save_jsonl(split_data, output_path)

        manifest_rows.append({
            "dataset": dataset_name,
            "subset": subset_name,
            "split": split_name,
            "num_rows": len(split_data),
            "file": str(output_path)
        })

        print(f"[OK] {dataset_name}/{subset_name}/{split_name}: {len(split_data)} örnek")

    return {
        "dataset": dataset_name,
        "subset": subset_name,
        "splits": manifest_rows
    }


def download_openbookqa(output_dir: Path, cache_dir: str | None = None) -> Dict[str, Any]:
    """
    OpenBookQA veri setini indirir.

    Hugging Face adı:
    allenai/openbookqa

    Kullanacağımız yapılandırma:
    main
    """

    print("\n[INFO] OpenBookQA indiriliyor...")

    dataset = load_dataset(
        "allenai/openbookqa",
        "main",
        cache_dir=cache_dir
    )

    return save_dataset_dict(
        dataset=dataset,
        output_dir=output_dir,
        dataset_name="openbookqa",
        subset_name="main"
    )


def download_arc_easy(output_dir: Path, cache_dir: str | None = None) -> Dict[str, Any]:
    """
    ARC-Easy veri setini indirir.

    Hugging Face adı:
    allenai/ai2_arc

    Kullanacağımız yapılandırma:
    ARC-Easy
    """

    print("\n[INFO] ARC-Easy indiriliyor...")

    dataset = load_dataset(
        "allenai/ai2_arc",
        "ARC-Easy",
        cache_dir=cache_dir
    )

    return save_dataset_dict(
        dataset=dataset,
        output_dir=output_dir,
        dataset_name="arc",
        subset_name="ARC-Easy"
    )


def download_arc_challenge(output_dir: Path, cache_dir: str | None = None) -> Dict[str, Any]:
    """
    ARC-Challenge veri setini indirir.

    Hugging Face adı:
    allenai/ai2_arc

    Kullanacağımız yapılandırma:
    ARC-Challenge
    """

    print("\n[INFO] ARC-Challenge indiriliyor...")

    dataset = load_dataset(
        "allenai/ai2_arc",
        "ARC-Challenge",
        cache_dir=cache_dir
    )

    return save_dataset_dict(
        dataset=dataset,
        output_dir=output_dir,
        dataset_name="arc",
        subset_name="ARC-Challenge"
    )


def save_manifest(manifest: Dict[str, Any], output_dir: Path) -> None:
    """
    İndirilen dosyalarla ilgili kısa bir özet dosyası oluşturur.

    Bu dosya daha sonra şunu kontrol etmek için faydalı olur:
    - Hangi veri setleri indirildi?
    - Hangi split kaç örnek içeriyor?
    - Dosyalar nereye kaydedildi?
    """

    manifest_path = output_dir / "dataset_manifest.json"

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] Manifest kaydedildi: {manifest_path}")


def print_sample(output_dir: Path) -> None:
    """
    İndirme bittikten sonra OpenBookQA train dosyasından bir örnek yazdırır.

    Bu kontrol amaçlıdır.
    Verinin gerçekten beklediğimiz gibi gelip gelmediğini görürüz.
    """

    sample_file = output_dir / "openbookqa" / "main" / "train.jsonl"

    if not sample_file.exists():
        print("[WARNING] Örnek dosya bulunamadı.")
        return

    print("\n[INFO] OpenBookQA train içinden örnek:")

    with sample_file.open("r", encoding="utf-8") as f:
        first_line = f.readline()
        example = json.loads(first_line)

    print(json.dumps(example, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    """
    Komut satırı argümanlarını okur.

    Örnek kullanım:
    python scripts/01_download_datasets.py

    Farklı klasöre kaydetmek istersek:
    python scripts/01_download_datasets.py --output_dir data/raw
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/raw",
        help="Ham veri setlerinin kaydedileceği klasör."
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Hugging Face cache klasörü. Boş bırakılırsa varsayılan cache kullanılır."
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Veri setleri indirilmeye başlanıyor...")
    print(f"[INFO] Çıktı klasörü: {output_dir}")

    manifest = {
        "description": "Raw dataset files for GRPO multiple-choice QA project.",
        "datasets": []
    }

    openbookqa_info = download_openbookqa(
        output_dir=output_dir,
        cache_dir=args.cache_dir
    )
    manifest["datasets"].append(openbookqa_info)

    arc_easy_info = download_arc_easy(
        output_dir=output_dir,
        cache_dir=args.cache_dir
    )
    manifest["datasets"].append(arc_easy_info)

    arc_challenge_info = download_arc_challenge(
        output_dir=output_dir,
        cache_dir=args.cache_dir
    )
    manifest["datasets"].append(arc_challenge_info)

    save_manifest(manifest, output_dir)
    print_sample(output_dir)

    print("\n[DONE] Tüm veri setleri başarıyla indirildi ve kaydedildi.")


if __name__ == "__main__":
    main()