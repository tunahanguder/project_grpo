"""
08_generate_similar_questions.py

Amaç:
- OpenBookQA original sorularını okumak
- Teacher model ile her soru için benzer bir çoktan seçmeli soru üretmek
- Benzer soru aynı kavramı ölçmeli ama kelime kelime kopya olmamalı
- Çıktıyı JSONL olarak kaydetmek

mB için kullanılacak:
original question + similar question
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


VALID_ANSWERS = {"A", "B", "C", "D"}


def read_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    examples = []
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def append_jsonl(example: Dict[str, Any], file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(example, ensure_ascii=False) + "\n")


def load_existing_ids(output_path: Path) -> set:
    existing_ids = set()

    if not output_path.exists():
        return existing_ids

    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
                existing_ids.add(item["id"])
            except Exception:
                continue

    return existing_ids


def format_choices(choices: Dict[str, str]) -> str:
    return "\n".join([f"{k}) {choices[k]}" for k in ["A", "B", "C", "D"]])


def build_teacher_prompt(example: Dict[str, Any]) -> str:
    question = example["question"]
    choices = format_choices(example["choices"])
    answer = example["answer"]

    prompt = f"""
Aşağıdaki çoktan seçmeli fen/reasoning sorusuna kavramsal olarak benzer yeni bir soru üret.

Orijinal soru:
{question}

Orijinal seçenekler:
{choices}

Orijinal doğru cevap: {answer}

Görev:
- Aynı temel kavramı veya bilgiyi ölçen yeni bir çoktan seçmeli soru yaz.
- Soruyu kelime kelime yeniden yazma; yeni bir bağlam veya nesne kullan.
- 4 seçenek üret: A, B, C, D.
- Yalnızca bir doğru cevap olsun.
- Seçenekler kısa ve açık olsun.
- Doğru cevabı belirt.
- Türkçe çıktı üret.

Çıktıyı kesinlikle aşağıdaki JSON formatında ver.
JSON dışında hiçbir açıklama yazma.

{{
  "similar_question": "...",
  "similar_choices": {{
    "A": "...",
    "B": "...",
    "C": "...",
    "D": "..."
  }},
  "similar_answer": "A"
}}
""".strip()

    return prompt


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    Teacher cevabından JSON nesnesini çıkarır.

    Model bazen ```json ... ``` yazabilir.
    Bu yüzden önce code fence temizliyoruz.
    """

    if text is None:
        return None

    cleaned = text.strip()

    cleaned = cleaned.replace("```json", "").replace("```JSON", "").replace("```", "").strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start == -1 or end == -1 or end <= start:
        return None

    json_text = cleaned[start:end + 1]

    try:
        return json.loads(json_text)
    except Exception:
        return None


def validate_similar_record(obj: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    if not isinstance(obj, dict):
        return False, "not_dict"

    if "similar_question" not in obj:
        return False, "missing_similar_question"

    if "similar_choices" not in obj:
        return False, "missing_similar_choices"

    if "similar_answer" not in obj:
        return False, "missing_similar_answer"

    question = str(obj["similar_question"]).strip()
    choices = obj["similar_choices"]
    answer = str(obj["similar_answer"]).strip().upper()

    if question == "":
        return False, "empty_question"

    if not isinstance(choices, dict):
        return False, "choices_not_dict"

    for label in ["A", "B", "C", "D"]:
        if label not in choices:
            return False, f"missing_choice_{label}"

        if str(choices[label]).strip() == "":
            return False, f"empty_choice_{label}"

    if answer not in VALID_ANSWERS:
        return False, "invalid_answer"

    choice_texts = [str(choices[label]).strip() for label in ["A", "B", "C", "D"]]

    if len(set(choice_texts)) != 4:
        return False, "duplicate_choices"

    return True, None


def normalize_similar_obj(obj: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "similar_question": str(obj["similar_question"]).strip(),
        "similar_choices": {
            "A": str(obj["similar_choices"]["A"]).strip(),
            "B": str(obj["similar_choices"]["B"]).strip(),
            "C": str(obj["similar_choices"]["C"]).strip(),
            "D": str(obj["similar_choices"]["D"]).strip(),
        },
        "similar_answer": str(obj["similar_answer"]).strip().upper(),
    }


def load_teacher_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map={"": 0},
        trust_remote_code=True,
    )

    model.eval()

    return tokenizer, model


@torch.inference_mode()
def generate_text(
    tokenizer,
    model,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "Sen çoktan seçmeli veri setleri için kaliteli, tutarlı ve "
                "JSON formatında benzer soru üreten bir eğitim verisi üreticisisin."
            ),
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer([text], return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id,
    )

    input_len = inputs["input_ids"].shape[1]
    new_tokens = generated_ids[0][input_len:]

    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def build_output_record(
    original: Dict[str, Any],
    teacher_prompt: str,
    teacher_response: str,
    parsed_obj: Optional[Dict[str, Any]],
    is_valid: bool,
    invalid_reason: Optional[str],
) -> Dict[str, Any]:

    record = {
        "id": original["id"],
        "source": original.get("source"),
        "subset": original.get("subset"),
        "split": original.get("split"),
        "split_role": original.get("split_role"),

        "original_question": original["question"],
        "original_choices": original["choices"],
        "original_answer": original["answer"],

        "teacher_prompt": teacher_prompt,
        "teacher_response": teacher_response,

        "is_valid": is_valid,
        "invalid_reason": invalid_reason,
    }

    if is_valid and parsed_obj is not None:
        normalized = normalize_similar_obj(parsed_obj)
        record.update(normalized)
    else:
        record["similar_question"] = None
        record["similar_choices"] = None
        record["similar_answer"] = None

    return record


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Original OpenBookQA dosyası.",
    )

    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Üretilecek similar question JSONL dosyası.",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-14B-Instruct-AWQ",
        help="Teacher model.",
    )

    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--top_p", type=float, default=0.9)

    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input_file)
    output_path = Path(args.output_file)

    print("[INFO] Similar question üretimi başlıyor...")
    print(f"[INFO] Input: {input_path}")
    print(f"[INFO] Output: {output_path}")
    print(f"[INFO] Teacher model: {args.model_name}")

    examples = read_jsonl(input_path)

    if args.limit is not None:
        examples = examples[:args.limit]
        print(f"[INFO] Limit aktif: {len(examples)} örnek")

    existing_ids = set()

    if args.resume:
        existing_ids = load_existing_ids(output_path)
        print(f"[INFO] Resume aktif. Mevcut örnek sayısı: {len(existing_ids)}")

    tokenizer, model = load_teacher_model(args.model_name)

    valid_count = 0
    invalid_count = 0
    skipped_count = 0

    for original in tqdm(examples, desc="Generating similar questions"):
        example_id = original["id"]

        if args.resume and example_id in existing_ids:
            skipped_count += 1
            continue

        teacher_prompt = build_teacher_prompt(original)

        try:
            teacher_response = generate_text(
                tokenizer=tokenizer,
                model=model,
                prompt=teacher_prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )

            parsed_obj = extract_json_object(teacher_response)

            if parsed_obj is None:
                is_valid = False
                invalid_reason = "json_parse_failed"
            else:
                is_valid, invalid_reason = validate_similar_record(parsed_obj)

        except Exception as e:
            teacher_response = ""
            parsed_obj = None
            is_valid = False
            invalid_reason = f"generation_error: {repr(e)}"

        record = build_output_record(
            original=original,
            teacher_prompt=teacher_prompt,
            teacher_response=teacher_response,
            parsed_obj=parsed_obj,
            is_valid=is_valid,
            invalid_reason=invalid_reason,
        )

        append_jsonl(record, output_path)

        if is_valid:
            valid_count += 1
        else:
            invalid_count += 1

    print("\n[DONE] Similar question üretimi tamamlandı.")
    print(f"[INFO] Geçerli: {valid_count}")
    print(f"[INFO] Geçersiz: {invalid_count}")
    print(f"[INFO] Atlanan/resume: {skipped_count}")
    print(f"[INFO] Çıktı: {output_path}")


if __name__ == "__main__":
    main()