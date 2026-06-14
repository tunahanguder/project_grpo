"""
05_build_sft_dataset.py

Bu dosyanın amacı:
- 04_generate_sft_rationales.py çıktısı olan teacher rationale dosyasını okumak
- is_valid=True olan örnekleri seçmek
- Öğrenci model için temiz bir SFT prompt'u oluşturmak
- Teacher cevabını assistant response olarak kullanmak
- Final SFT eğitim dosyasını JSONL olarak kaydetmek

Önemli:
Teacher prompt'u kullanılmaz.
Çünkü teacher prompt'unda doğru cevap açıkça verilir.
Öğrenci modelin prompt'unda doğru cevap olmamalıdır.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


VALID_ANSWERS = {"A", "B", "C", "D"}


def read_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """
    JSONL dosyasını okur.

    Her satır bir JSON nesnesidir.
    Dosyadaki tüm örnekleri liste olarak döndürür.
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


def format_choices(choices: Dict[str, str]) -> str:
    """
    Seçenekleri prompt içinde kullanılacak metne çevirir.

    Örnek:
    A) seçenek metni
    B) seçenek metni
    C) seçenek metni
    D) seçenek metni
    """

    lines = []

    for label in ["A", "B", "C", "D"]:
        lines.append(f"{label}) {choices[label]}")

    return "\n".join(lines)


def build_student_prompt(example: Dict[str, Any]) -> str:
    """
    Öğrenci modele verilecek temiz prompt'u oluşturur.

    Burada doğru cevap verilmez.
    Modelden soruyu çözmesi istenir.
    """

    question = example["question"]
    choices_text = format_choices(example["choices"])

    prompt = f"""
Aşağıdaki çoktan seçmeli soruyu çöz.

Soru:
{question}

Seçenekler:
{choices_text}

Cevabını kısa bir açıklama ile ver.
Son satır kesinlikle şu formatta olmalı:
Nihai cevap: <A/B/C/D>
""".strip()

    return prompt


def extract_final_answer(text: str) -> Optional[str]:
    """
    Cevap metninden 'Nihai cevap: X' biçimindeki cevabı çıkarır.

    Kabul edilen örnekler:
    Nihai cevap: A
    Nihai Cevap: B
    nihai cevap: C
    """

    pattern = r"nihai\s+cevap\s*:\s*([ABCD])"
    match = re.search(pattern, text, flags=re.IGNORECASE)

    if match is None:
        return None

    return match.group(1).upper()


def clean_teacher_response(response: str) -> str:
    """
    Teacher cevabını SFT response olarak kullanmadan önce temizler.

    Burada çok agresif temizlik yapmıyoruz.
    Sadece boşlukları ve fazla satır sonlarını düzenliyoruz.
    """

    response = response.strip()

    # Windows/Linux satır sonlarını tek biçime getir.
    response = response.replace("\r\n", "\n").replace("\r", "\n")

    # Üçten fazla boş satırı azalt.
    response = re.sub(r"\n{3,}", "\n\n", response)

    return response.strip()


def validate_final_sft_example(
    prompt: str,
    response: str,
    gold_answer: str
) -> Tuple[bool, Optional[str]]:
    """
    Final SFT örneğini kontrol eder.

    Kontroller:
    - Prompt boş mu?
    - Response boş mu?
    - Response içinde Nihai cevap var mı?
    - Nihai cevap doğru cevapla aynı mı?
    """

    if prompt.strip() == "":
        return False, "empty_prompt"

    if response.strip() == "":
        return False, "empty_response"

    final_answer = extract_final_answer(response)

    if final_answer is None:
        return False, "missing_final_answer"

    if final_answer not in VALID_ANSWERS:
        return False, "invalid_final_answer_label"

    if final_answer != gold_answer:
        return False, "final_answer_mismatch"

    return True, None


def build_chat_messages(prompt: str, response: str) -> List[Dict[str, str]]:
    """
    Chat tabanlı SFT formatı oluşturur.

    TRL SFTTrainer gibi araçlarda messages alanı doğrudan kullanılabilir.
    """

    messages = [
        {
            "role": "system",
            "content": (
                "Sen çoktan seçmeli soruları kısa ve doğru açıklamalarla çözen "
                "yardımcı bir yapay zekâ modelisin."
            )
        },
        {
            "role": "user",
            "content": prompt
        },
        {
            "role": "assistant",
            "content": response
        }
    ]

    return messages


def build_plain_text(messages: List[Dict[str, str]]) -> str:
    """
    Basit düz metin alanı oluşturur.

    Bu alan zorunlu değil ama debug ve bazı eğitim formatları için kullanışlıdır.
    """

    system_text = messages[0]["content"]
    user_text = messages[1]["content"]
    assistant_text = messages[2]["content"]

    text = (
        f"Sistem: {system_text}\n\n"
        f"Kullanıcı:\n{user_text}\n\n"
        f"Asistan:\n{assistant_text}"
    )

    return text


def build_sft_record(
    example: Dict[str, Any]
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Teacher çıktısından final SFT kaydı oluşturur.

    Başarılıysa:
        record, None

    Başarısızsa:
        None, invalid_reason
    """

    if not example.get("is_valid", False):
        return None, "teacher_output_invalid"

    teacher_response = example.get("teacher_response", "")
    gold_answer = example.get("answer")

    if gold_answer not in VALID_ANSWERS:
        return None, "invalid_gold_answer"

    prompt = build_student_prompt(example)
    response = clean_teacher_response(teacher_response)

    is_valid, invalid_reason = validate_final_sft_example(
        prompt=prompt,
        response=response,
        gold_answer=gold_answer
    )

    if not is_valid:
        return None, invalid_reason

    messages = build_chat_messages(
        prompt=prompt,
        response=response
    )

    text = build_plain_text(messages)

    record = {
        "id": example["id"],
        "source": example.get("source"),
        "subset": example.get("subset"),
        "split": example.get("split"),
        "split_role": example.get("split_role"),

        "question": example["question"],
        "choices": example["choices"],
        "answer": gold_answer,

        # SFT eğitiminde kullanacağımız ana alanlar
        "prompt": prompt,
        "response": response,
        "messages": messages,

        # Alternatif düz metin formatı
        "text": text,

        # İzlenebilirlik alanları
        "teacher_is_valid": example.get("is_valid"),
        "teacher_invalid_reason": example.get("invalid_reason"),
        "original_id": example.get("original_id")
    }

    return record, None


def save_stats(stats: Dict[str, Any], output_path: Path) -> None:
    """
    SFT dataset oluşturma istatistiklerini JSON olarak kaydeder.
    """

    stats_path = output_path.with_suffix(".stats.json")

    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"[OK] İstatistik dosyası kaydedildi: {stats_path}")


def print_sample(records: List[Dict[str, Any]]) -> None:
    """
    Oluşturulan final SFT kayıtlarından bir örnek yazdırır.
    """

    if not records:
        print("[WARNING] Yazdırılacak örnek yok.")
        return

    sample = records[0]

    print("\n[INFO] Final SFT örneği:")
    print(json.dumps(sample, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    """
    Komut satırı argümanlarını okur.

    Yerel kullanım:
    python scripts/05_build_sft_dataset.py

    Colab/Drive kullanımında dosya yollarını açık vermek daha güvenlidir.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_file",
        type=str,
        default="data/teacher_outputs/sft_rationales_train.jsonl",
        help="Teacher rationale çıktısı."
    )

    parser.add_argument(
        "--output_file",
        type=str,
        default="data/processed/final/sft_train.jsonl",
        help="Final SFT eğitim dosyası."
    )

    parser.add_argument(
        "--invalid_file",
        type=str,
        default="data/processed/final/sft_invalid.jsonl",
        help="Final SFT'ye alınmayan örneklerin kaydedileceği dosya."
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input_file)
    output_path = Path(args.output_file)
    invalid_path = Path(args.invalid_file)

    print("[INFO] Final SFT dataset oluşturma başlıyor...")
    print(f"[INFO] Input: {input_path}")
    print(f"[INFO] Output: {output_path}")
    print(f"[INFO] Invalid output: {invalid_path}")

    examples = read_jsonl(input_path)

    final_records = []
    invalid_records = []
    invalid_reason_counts = {}

    for example in examples:
        record, invalid_reason = build_sft_record(example)

        if record is None:
            invalid_example = dict(example)
            invalid_example["final_invalid_reason"] = invalid_reason
            invalid_records.append(invalid_example)

            invalid_reason_counts[invalid_reason] = (
                invalid_reason_counts.get(invalid_reason, 0) + 1
            )
            continue

        final_records.append(record)

    write_jsonl(final_records, output_path)
    write_jsonl(invalid_records, invalid_path)

    stats = {
        "input_count": len(examples),
        "final_sft_count": len(final_records),
        "invalid_count": len(invalid_records),
        "invalid_reason_counts": invalid_reason_counts,
        "output_file": str(output_path),
        "invalid_file": str(invalid_path)
    }

    save_stats(stats, output_path)

    print("\n[DONE] Final SFT dataset oluşturuldu.")
    print(f"[INFO] Input örnek sayısı: {len(examples)}")
    print(f"[INFO] Final SFT örnek sayısı: {len(final_records)}")
    print(f"[INFO] Elenen örnek sayısı: {len(invalid_records)}")

    if invalid_reason_counts:
        print("\n[INFO] Elenme sebepleri:")
        for reason, count in invalid_reason_counts.items():
            print(f"  - {reason}: {count}")

    print_sample(final_records)


if __name__ == "__main__":
    main()