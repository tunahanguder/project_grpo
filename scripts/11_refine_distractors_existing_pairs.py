
import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


VALID_ANSWERS = {"A", "B", "C", "D"}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def format_choices(choices: Dict[str, str]) -> str:
    return "\n".join([f"{k}) {choices[k]}" for k in ["A", "B", "C", "D"]])


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None

    text = text.strip()
    text = re.sub(r"^```json", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"^```", "", text).strip()
    text = re.sub(r"```$", "", text).strip()

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        return None

    try:
        return json.loads(text[start:end + 1])
    except Exception:
        return None


def normalize_completion(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        if not completion:
            return ""
        first = completion[0]
        if isinstance(first, dict) and "content" in first:
            return str(first["content"])
        return str(first)
    if isinstance(completion, dict) and "content" in completion:
        return str(completion["content"])
    return str(completion)


def extract_single_answer(text: str) -> Optional[str]:
    if not text:
        return None

    patterns = [
        r"nihai\s+cevap\s*:\s*([ABCD])",
        r"cevap\s*:\s*([ABCD])",
        r"answer\s*:\s*([ABCD])",
    ]

    matches = []
    for pattern in patterns:
        for m in re.finditer(pattern, text, flags=re.IGNORECASE):
            matches.append(m.group(1).upper())

    return matches[-1] if matches else None


def validate_choices(obj: Dict[str, Any], original: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if "choices_1" not in obj or "choices_2" not in obj:
        return None

    if not isinstance(obj["choices_1"], dict) or not isinstance(obj["choices_2"], dict):
        return None

    for key in ["A", "B", "C", "D"]:
        if key not in obj["choices_1"] or key not in obj["choices_2"]:
            return None

    answer_1 = str(original["answer_1"]).strip().upper()
    answer_2 = str(original["answer_2"]).strip().upper()

    if answer_1 not in VALID_ANSWERS or answer_2 not in VALID_ANSWERS:
        return None

    choices_1 = {k: str(obj["choices_1"][k]).strip() for k in ["A", "B", "C", "D"]}
    choices_2 = {k: str(obj["choices_2"][k]).strip() for k in ["A", "B", "C", "D"]}

    # Doğru şık metnini koruyoruz. Teacher yanlışlıkla değiştirirse geri alıyoruz.
    choices_1[answer_1] = str(original["choices_1"][answer_1]).strip()
    choices_2[answer_2] = str(original["choices_2"][answer_2]).strip()

    return {
        "id": original.get("id", ""),
        "source_id": original.get("id", ""),
        "question_1": original["question_1"],
        "choices_1": choices_1,
        "answer_1": answer_1,
        "question_2": original["question_2"],
        "choices_2": choices_2,
        "answer_2": answer_2,
        "original_choices_1": original["choices_1"],
        "original_choices_2": original["choices_2"],
    }


def build_teacher_prompt(example: Dict[str, Any], attempt: int) -> List[Dict[str, str]]:
    answer_1 = str(example["answer_1"]).strip().upper()
    answer_2 = str(example["answer_2"]).strip().upper()

    correct_1 = example["choices_1"][answer_1]
    correct_2 = example["choices_2"][answer_2]

    user_text = f"""
Aşağıdaki iki çoktan seçmeli soru için SADECE yanlış seçenekleri güçlendir.

Kesin kurallar:
- Soru 1 ve Soru 2 metinlerini değiştirme.
- Doğru cevap harflerini değiştirme.
- Doğru cevap metinlerini değiştirme.
- Sadece yanlış seçenekleri yeniden yaz.
- Yanlış seçenekler doğru cevaba semantik olarak yakın, yüzeysel olarak makul, fakat kesinlikle yanlış olmalı.
- Bariz saçma, çok kısa, alakasız veya kolay elenen çeldirici yazma.
- Seçeneklerin uzunluk ve biçimleri birbirine yakın olsun.
- Türkçe yaz.

Zorluk seviyesi:
- Bu attempt = {attempt}
- Attempt arttıkça çeldiricileri daha güçlü ve daha yanıltıcı yap.

[SORU 1]
{example["question_1"]}

Mevcut seçenekler 1:
{format_choices(example["choices_1"])}

Doğru cevap 1: {answer_1}
Doğru cevap 1 metni: {correct_1}

[SORU 2]
{example["question_2"]}

Mevcut seçenekler 2:
{format_choices(example["choices_2"])}

Doğru cevap 2: {answer_2}
Doğru cevap 2 metni: {correct_2}

Sadece şu JSON formatında cevap ver:

{{
  "choices_1": {{
    "A": "...",
    "B": "...",
    "C": "...",
    "D": "..."
  }},
  "choices_2": {{
    "A": "...",
    "B": "...",
    "C": "...",
    "D": "..."
  }},
  "distractor_rationale": "Yanlış seçenekleri nasıl güçlendirdiğini bir cümleyle açıkla."
}}
""".strip()

    return [
        {
            "role": "system",
            "content": (
                "Sen çoktan seçmeli sorularda güçlü çeldirici tasarlayan uzman bir veri üreticisisin. "
                "Görevin doğru cevabı koruyup yanlış seçenekleri kavramsal olarak zorlaştırmaktır."
            ),
        },
        {"role": "user", "content": user_text},
    ]


def build_student_prompt(question: str, choices: Dict[str, str]) -> List[Dict[str, str]]:
    user_text = f"""
Aşağıdaki çoktan seçmeli soruyu çöz.

Soru:
{question}

Seçenekler:
{format_choices(choices)}

Son satırda mutlaka şu formatı kullan:
Nihai cevap: <A/B/C/D>
""".strip()

    return [
        {"role": "system", "content": "Sen çoktan seçmeli soruları çözen uzman bir yapay zekâ modelisin."},
        {"role": "user", "content": user_text},
    ]


def generate_text(model, tokenizer, messages, max_new_tokens, temperature, top_p, do_sample=True) -> str:
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    gen = outputs[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True)


def student_solve_pair(student_model, student_tokenizer, pair: Dict[str, Any], max_new_tokens: int):
    out1 = generate_text(
        student_model,
        student_tokenizer,
        build_student_prompt(pair["question_1"], pair["choices_1"]),
        max_new_tokens=max_new_tokens,
        temperature=0.1,
        top_p=0.95,
        do_sample=False,
    )

    out2 = generate_text(
        student_model,
        student_tokenizer,
        build_student_prompt(pair["question_2"], pair["choices_2"]),
        max_new_tokens=max_new_tokens,
        temperature=0.1,
        top_p=0.95,
        do_sample=False,
    )

    pred1 = extract_single_answer(out1)
    pred2 = extract_single_answer(out2)

    return pred1, pred2, out1, out2


def is_hard_enough(pred1, pred2, gold1, gold2, accept_if_one_wrong=True) -> bool:
    c1 = pred1 == gold1
    c2 = pred2 == gold2

    if accept_if_one_wrong:
        return not (c1 and c2)

    return (not c1) and (not c2)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)

    parser.add_argument("--teacher_model", type=str, required=True)
    parser.add_argument("--student_model", type=str, required=True)

    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max_attempts", type=int, default=3)

    parser.add_argument("--teacher_max_new_tokens", type=int, default=900)
    parser.add_argument("--teacher_temperature", type=float, default=0.8)
    parser.add_argument("--teacher_top_p", type=float, default=0.95)

    parser.add_argument("--student_max_new_tokens", type=int, default=256)
    parser.add_argument("--accept_if_one_wrong", action="store_true")
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    input_path = Path(args.input_file)
    output_path = Path(args.output_file)

    if args.overwrite and output_path.exists():
        output_path.unlink()

    examples = read_jsonl(input_path)
    if args.limit is not None:
        examples = examples[:args.limit]

    print("[INFO] Mevcut soru çiftlerinde çeldirici güçlendirme başlıyor...")
    print(f"[INFO] Input: {input_path}")
    print(f"[INFO] Output: {output_path}")
    print(f"[INFO] Örnek sayısı: {len(examples)}")

    print("[INFO] Teacher yükleniyor...")
    teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher_model, trust_remote_code=True)
    teacher_model = AutoModelForCausalLM.from_pretrained(
        args.teacher_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    if teacher_tokenizer.pad_token is None:
        teacher_tokenizer.pad_token = teacher_tokenizer.eos_token

    print("[INFO] Student yükleniyor...")
    student_tokenizer = AutoTokenizer.from_pretrained(args.student_model, trust_remote_code=True)
    student_model = AutoModelForCausalLM.from_pretrained(
        args.student_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    if student_tokenizer.pad_token is None:
        student_tokenizer.pad_token = student_tokenizer.eos_token

    accepted = 0
    fallback = 0
    failed = 0

    for idx, ex in enumerate(tqdm(examples, desc="Çeldirici güçlendirme")):
        best_pair = None
        accepted_pair = None

        for attempt in range(1, args.max_attempts + 1):
            teacher_out = generate_text(
                teacher_model,
                teacher_tokenizer,
                build_teacher_prompt(ex, attempt),
                max_new_tokens=args.teacher_max_new_tokens,
                temperature=args.teacher_temperature,
                top_p=args.teacher_top_p,
                do_sample=True,
            )

            obj = extract_json_object(teacher_out)
            if obj is None:
                continue

            pair = validate_choices(obj, ex)
            if pair is None:
                continue

            pair["generation_attempt"] = attempt
            pair["teacher_raw_rationale"] = obj.get("distractor_rationale", "")

            pred1, pred2, raw1, raw2 = student_solve_pair(
                student_model,
                student_tokenizer,
                pair,
                max_new_tokens=args.student_max_new_tokens,
            )

            pair["student_pred_1"] = pred1
            pair["student_pred_2"] = pred2
            pair["student_correct_1"] = pred1 == pair["answer_1"]
            pair["student_correct_2"] = pred2 == pair["answer_2"]
            pair["student_raw_1"] = raw1
            pair["student_raw_2"] = raw2

            best_pair = pair

            if is_hard_enough(
                pred1,
                pred2,
                pair["answer_1"],
                pair["answer_2"],
                accept_if_one_wrong=args.accept_if_one_wrong,
            ):
                accepted_pair = pair
                break

        if accepted_pair is not None:
            accepted_pair["strong_distractor_status"] = "accepted_student_failed"
            append_jsonl(output_path, accepted_pair)
            accepted += 1

        elif best_pair is not None:
            best_pair["strong_distractor_status"] = "fallback_student_solved"
            append_jsonl(output_path, best_pair)
            fallback += 1

        else:
            failed_row = dict(ex)
            failed_row["strong_distractor_status"] = "failed_generation_kept_original"
            append_jsonl(output_path, failed_row)
            failed += 1

    print("[DONE] Çeldirici güçlendirme tamamlandı.")
    print(f"[INFO] Accepted hard: {accepted}")
    print(f"[INFO] Fallback solved: {fallback}")
    print(f"[INFO] Failed kept original: {failed}")
    print(f"[INFO] Output: {output_path}")


if __name__ == "__main__":
    main()
PY