"""
04_generate_sft_rationales.py

Bu dosyanın amacı:
- sft_train_pool.jsonl dosyasındaki çoktan seçmeli soruları okumak
- Qwen2.5-72B-Instruct-AWQ teacher modeli ile kısa açıklamalı cevap üretmek
- Üretilen cevabın doğru formatta olup olmadığını kontrol etmek
- Sonuçları JSONL olarak kaydetmek

Bu script SFT eğitimini yapmaz.
Sadece SFT için hedef cevapları, yani teacher rationales üretir.

Önemli:
- Teacher modele doğru cevap verilir.
- Teacher'dan cevabı bulmasını değil, doğru cevabı açıklamasını isteriz.
- Böylece yanlış cevap üretme riski azaltılır.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


VALID_ANSWERS = {"A", "B", "C", "D"}


def read_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """
    JSONL dosyasını okur.
    Her satır bir JSON nesnesidir.
    """

    examples = []

    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            examples.append(json.loads(line))

    return examples


def append_jsonl(example: Dict[str, Any], file_path: Path) -> None:
    """
    Tek bir JSON nesnesini JSONL dosyasının sonuna ekler.

    Neden append kullanıyoruz?
    Çünkü teacher üretimi uzun sürebilir.
    İşlem yarıda kalırsa baştan başlamamak için her örneği üretildiği anda kaydediyoruz.
    """

    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(example, ensure_ascii=False) + "\n")


def load_existing_ids(output_path: Path) -> set:
    """
    Daha önce üretilmiş örneklerin id'lerini okur.

    Böylece script yarıda kalırsa tekrar çalıştırıldığında
    aynı örnekleri yeniden üretmez.
    """

    existing_ids = set()

    if not output_path.exists():
        return existing_ids

    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            try:
                example = json.loads(line)
                existing_ids.add(example["id"])
            except Exception:
                continue

    return existing_ids


def format_choices(choices: Dict[str, str]) -> str:
    """
    choices sözlüğünü prompt içinde kullanılacak metne çevirir.

    Örnek:
    A) ...
    B) ...
    C) ...
    D) ...
    """

    lines = []

    for label in ["A", "B", "C", "D"]:
        choice_text = choices[label]
        lines.append(f"{label}) {choice_text}")

    return "\n".join(lines)


def build_teacher_prompt(example: Dict[str, Any]) -> str:
    """
    Teacher modele verilecek prompt'u oluşturur.

    Burada kritik nokta:
    Teacher modele doğru cevabı açıkça veriyoruz.
    Böylece modelden 'cevabı tahmin etmesini' değil,
    'verilen doğru cevabı kısa şekilde açıklamasını' istiyoruz.
    """

    question = example["question"]
    choices_text = format_choices(example["choices"])
    answer = example["answer"]

    prompt = f"""
Aşağıdaki çoktan seçmeli soru için kısa ve doğru bir açıklama üret.

Soru:
{question}

Seçenekler:
{choices_text}

Doğru cevap: {answer}

Kurallar:
- Doğru cevabı değiştirme.
- 2-4 cümlelik kısa bir açıklama yaz.
- Açıklama, doğru seçeneğin neden doğru olduğunu anlaşılır biçimde belirtmeli.
- Çok uzun yazma.
- Son satır kesinlikle şu formatta olmalı:
Nihai cevap: {answer}
""".strip()

    return prompt


def extract_final_answer(text: str) -> Optional[str]:
    """
    Üretilen metinden 'Nihai cevap: X' biçimindeki cevabı çıkarır.

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


def validate_teacher_response(text: str, gold_answer: str) -> tuple[bool, Optional[str]]:
    """
    Teacher cevabının kullanılabilir olup olmadığını kontrol eder.

    Kontroller:
    1. Boş mu?
    2. Nihai cevap formatı var mı?
    3. Nihai cevap gold answer ile aynı mı?
    4. Cevap aşırı kısa mı?
    """

    if text is None or text.strip() == "":
        return False, "empty_response"

    text = text.strip()

    final_answer = extract_final_answer(text)

    if final_answer is None:
        return False, "missing_final_answer"

    if final_answer not in VALID_ANSWERS:
        return False, "invalid_final_answer_label"

    if final_answer != gold_answer:
        return False, "final_answer_mismatch"

    word_count = len(text.split())

    if word_count < 8:
        return False, "too_short"

    return True, None


def load_teacher_model(model_name: str):
    """
    Teacher tokenizer ve modeli yükler.

    device_map='auto':
    Modeli mevcut GPU/CPU kaynaklarına otomatik yerleştirmeye çalışır.

    torch_dtype='auto':
    Model kartındaki uygun dtype ayarını kullanmaya çalışır.

    Not:
    72B AWQ model yerel bilgisayarda çok büyük olabilir.
    Bu fonksiyon esas olarak Colab A100 / güçlü GPU ortamı için düşünülmüştür.
    """

    print(f"[INFO] Teacher tokenizer yükleniyor: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    print(f"[INFO] Teacher model yükleniyor: {model_name}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True
    )

    model.eval()

    return tokenizer, model


@torch.inference_mode()
def generate_response(
    tokenizer,
    model,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float
) -> str:
    """
    Teacher modelden cevap üretir.

    Burada chat template kullanıyoruz.
    Instruct modeller için bu daha doğru bir formattır.
    """

    messages = [
        {
            "role": "system",
            "content": (
                "Sen çoktan seçmeli sorular için kısa, doğru ve öğretici "
                "açıklamalar yazan bir eğitim verisi üreticisisin."
            )
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        [text],
        return_tensors="pt"
    )

    # input tensorlarını modelin bulunduğu cihaza alıyoruz.
    # device_map='auto' kullanıldığında model parçalı yerleşebilir;
    # yine de input_ids için genellikle model.device yeterlidir.
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id
    )

    # Sadece yeni üretilen tokenları almak için prompt kısmını kesiyoruz.
    input_length = inputs["input_ids"].shape[1]
    new_tokens = generated_ids[0][input_length:]

    response = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True
    )

    return response.strip()


def build_output_record(
    example: Dict[str, Any],
    teacher_prompt: str,
    teacher_response: str,
    is_valid: bool,
    invalid_reason: Optional[str]
) -> Dict[str, Any]:
    """
    Kaydedilecek çıktı kaydını oluşturur.
    """

    output_record = dict(example)

    output_record["teacher_prompt"] = teacher_prompt
    output_record["teacher_response"] = teacher_response
    output_record["is_valid"] = is_valid
    output_record["invalid_reason"] = invalid_reason

    return output_record


def parse_args() -> argparse.Namespace:
    """
    Komut satırı argümanları.

    İlk deneme için limit kullanmak iyi olur:
    python scripts/04_generate_sft_rationales.py --limit 5

    Tam üretim:
    python scripts/04_generate_sft_rationales.py
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_file",
        type=str,
        default="data/processed/splits/sft_train_pool.jsonl",
        help="Teacher rationale üretilecek SFT soru havuzu."
    )

    parser.add_argument(
        "--output_file",
        type=str,
        default="data/teacher_outputs/sft_rationales_train.jsonl",
        help="Teacher çıktılarının kaydedileceği JSONL dosyası."
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-72B-Instruct-AWQ",
        help="Teacher model adı."
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=160,
        help="Her soru için üretilecek maksimum yeni token sayısı."
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Üretim sıcaklığı. Düşük değer daha tutarlı cevap verir."
    )

    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Nucleus sampling parametresi."
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Sadece ilk N örneği üretmek için kullanılır. Test amaçlı faydalıdır."
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Açılırsa output dosyasında olan id'ler tekrar üretilmez."
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input_file)
    output_path = Path(args.output_file)

    print("[INFO] SFT teacher rationale üretimi başlıyor...")
    print(f"[INFO] Input: {input_path}")
    print(f"[INFO] Output: {output_path}")
    print(f"[INFO] Teacher model: {args.model_name}")

    examples = read_jsonl(input_path)

    if args.limit is not None:
        examples = examples[:args.limit]
        print(f"[INFO] Limit aktif. İşlenecek örnek sayısı: {len(examples)}")
    else:
        print(f"[INFO] İşlenecek toplam örnek sayısı: {len(examples)}")

    existing_ids = set()

    if args.resume:
        existing_ids = load_existing_ids(output_path)
        print(f"[INFO] Resume aktif. Daha önce üretilmiş örnek sayısı: {len(existing_ids)}")

    tokenizer, model = load_teacher_model(args.model_name)

    total = 0
    valid_count = 0
    invalid_count = 0
    skipped_count = 0

    for example in tqdm(examples, desc="Generating rationales"):
        example_id = example["id"]

        if args.resume and example_id in existing_ids:
            skipped_count += 1
            continue

        teacher_prompt = build_teacher_prompt(example)

        try:
            teacher_response = generate_response(
                tokenizer=tokenizer,
                model=model,
                prompt=teacher_prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p
            )

            is_valid, invalid_reason = validate_teacher_response(
                text=teacher_response,
                gold_answer=example["answer"]
            )

        except Exception as e:
            teacher_response = ""
            is_valid = False
            invalid_reason = f"generation_error: {repr(e)}"

        output_record = build_output_record(
            example=example,
            teacher_prompt=teacher_prompt,
            teacher_response=teacher_response,
            is_valid=is_valid,
            invalid_reason=invalid_reason
        )

        append_jsonl(output_record, output_path)

        total += 1

        if is_valid:
            valid_count += 1
        else:
            invalid_count += 1

    print("\n[DONE] Teacher rationale üretimi tamamlandı.")
    print(f"[INFO] Yeni işlenen örnek: {total}")
    print(f"[INFO] Geçerli çıktı: {valid_count}")
    print(f"[INFO] Geçersiz çıktı: {invalid_count}")
    print(f"[INFO] Atlanan/resume: {skipped_count}")
    print(f"[INFO] Çıktı dosyası: {output_path}")


if __name__ == "__main__":
    main()