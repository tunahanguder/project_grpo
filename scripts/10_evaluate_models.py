import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


VALID_ANSWERS = {"A", "B", "C", "D"}


def read_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    examples = []

    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))

    return examples


def format_choices(choices_dict: Dict[str, str]) -> str:
    return "\n".join([f"{k}) {choices_dict[k]}" for k in ["A", "B", "C", "D"]])


def normalize_gold(ans: Any) -> str:
    return str(ans).strip().upper()


def extract_single_answer(text: str) -> Optional[str]:
    if not text:
        return None

    text = str(text).strip()

    patterns = [
        r"nihai\s+cevap\s*:\s*([ABCD])",
        r"doğru\s+cevap\s*:\s*([ABCD])",
        r"cevap\s*:\s*([ABCD])",
        r"final\s+answer\s*:\s*([ABCD])",
        r"answer\s*:\s*([ABCD])",
    ]

    matches = []

    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            matches.append(match.group(1).upper())

    return matches[-1] if matches else None


def extract_dual_answers(text: str) -> Tuple[Optional[str], Optional[str]]:
    if not text:
        return None, None

    text = str(text).strip()

    patterns_1 = [
        r"nihai\s+cevap\s*1\s*:\s*([ABCD])",
        r"cevap\s*1\s*:\s*([ABCD])",
        r"final\s+answer\s*1\s*:\s*([ABCD])",
        r"answer\s*1\s*:\s*([ABCD])",
    ]

    patterns_2 = [
        r"nihai\s+cevap\s*2\s*:\s*([ABCD])",
        r"cevap\s*2\s*:\s*([ABCD])",
        r"final\s+answer\s*2\s*:\s*([ABCD])",
        r"answer\s*2\s*:\s*([ABCD])",
    ]

    matches_1 = []
    matches_2 = []

    for pattern in patterns_1:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            matches_1.append(match.group(1).upper())

    for pattern in patterns_2:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            matches_2.append(match.group(1).upper())

    ans_1 = matches_1[-1] if matches_1 else None
    ans_2 = matches_2[-1] if matches_2 else None

    return ans_1, ans_2


def build_single_prompt(question: str, choices: Dict[str, str]):
    user_text = f"""
Aşağıdaki çoktan seçmeli soruyu çöz.

Soru:
{question}

Seçenekler:
{format_choices(choices)}

Kurallar:
- Önce <think> ve </think> etiketleri arasında kısa bir akıl yürütme yap.
- Gereksiz tekrar yapma.
- Son satırda mutlaka şu formatı kullan:
Nihai cevap: <A/B/C/D>
""".strip()

    return [
        {
            "role": "system",
            "content": (
                "Sen mantıksal akıl yürütme yaparak çoktan seçmeli soruları "
                "kısa ve doğru şekilde çözen uzman bir yapay zekâ modelisin."
            ),
        },
        {
            "role": "user",
            "content": user_text,
        },
    ]


def build_dual_prompt(
    q1: str,
    c1: Dict[str, str],
    q2: str,
    c2: Dict[str, str],
):
    user_text = f"""
Aşağıdaki iki soruyu, yani Orijinal ve Benzer soruyu, tutarlı bir mantıkla çöz.

[SORU 1 - ORİJİNAL]
{q1}

Seçenekler 1:
{format_choices(c1)}

[SORU 2 - BENZER]
{q2}

Seçenekler 2:
{format_choices(c2)}

Kurallar:
- Önce <think> ve </think> etiketleri arasında kısa bir akıl yürütme yap.
- Her iki sorunun ortak mantığını bulmaya çalış.
- Gereksiz tekrar yapma.
- Düşünme süreci bittikten sonra en alt satırlarda mutlaka şu iki formatı kullan:
Nihai cevap 1: <A/B/C/D>
Nihai cevap 2: <A/B/C/D>
- Bu iki cevaptan sonra başka hiçbir şey yazma.
""".strip()

    return [
        {
            "role": "system",
            "content": (
                "Sen benzer çoktan seçmeli soruları tutarlı ve kısa akıl yürütmeyle "
                "çözen uzman bir yapay zekâ modelisin."
            ),
        },
        {
            "role": "user",
            "content": user_text,
        },
    ]


def generate_text(model, tokenizer, messages, max_new_tokens: int) -> str:
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    input_device = next(model.parameters()).device
    inputs = {k: v.to(input_device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]

    return tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)

    parser.add_argument(
        "--mode",
        type=str,
        choices=["single", "dual"],
        required=True,
        help="SFT ve mA için single, mB_1 için dual kullanılır.",
    )

    parser.add_argument("--max_new_tokens", type=int, default=350)

    args = parser.parse_args()

    print(f"\n[INFO] Model yükleniyor: {args.model_path}")
    print(f"[INFO] Test file: {args.test_file}")
    print(f"[INFO] Output file: {args.output_file}")
    print(f"[INFO] Mode: {args.mode}")
    print(f"[INFO] max_new_tokens: {args.max_new_tokens}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    model.eval()

    examples = read_jsonl(Path(args.test_file))

    details = []

    counts = {
        "total": len(examples),
        "orig_correct": 0,
        "sim_correct": 0,
        "both_correct": 0,
        "orig_only_correct": 0,
        "sim_only_correct": 0,
        "neither_correct": 0,
        "format_fail_1": 0,
        "format_fail_2": 0,
    }

    for ex in tqdm(examples, desc="Test seti değerlendiriliyor"):
        gold_1 = normalize_gold(ex["answer_1"])
        gold_2 = normalize_gold(ex["answer_2"])

        pred_1 = None
        pred_2 = None

        output_1 = ""
        output_2 = ""
        output_dual = ""

        if args.mode == "single":
            messages_1 = build_single_prompt(
                ex["question_1"],
                ex["choices_1"],
            )

            output_1 = generate_text(
                model=model,
                tokenizer=tokenizer,
                messages=messages_1,
                max_new_tokens=args.max_new_tokens,
            )

            pred_1 = extract_single_answer(output_1)

            messages_2 = build_single_prompt(
                ex["question_2"],
                ex["choices_2"],
            )

            output_2 = generate_text(
                model=model,
                tokenizer=tokenizer,
                messages=messages_2,
                max_new_tokens=args.max_new_tokens,
            )

            pred_2 = extract_single_answer(output_2)

        elif args.mode == "dual":
            messages_dual = build_dual_prompt(
                ex["question_1"],
                ex["choices_1"],
                ex["question_2"],
                ex["choices_2"],
            )

            output_dual = generate_text(
                model=model,
                tokenizer=tokenizer,
                messages=messages_dual,
                max_new_tokens=args.max_new_tokens,
            )

            pred_1, pred_2 = extract_dual_answers(output_dual)

        c1 = pred_1 == gold_1
        c2 = pred_2 == gold_2

        if pred_1 not in VALID_ANSWERS:
            counts["format_fail_1"] += 1

        if pred_2 not in VALID_ANSWERS:
            counts["format_fail_2"] += 1

        if c1:
            counts["orig_correct"] += 1

        if c2:
            counts["sim_correct"] += 1

        if c1 and c2:
            counts["both_correct"] += 1
        elif c1 and not c2:
            counts["orig_only_correct"] += 1
        elif not c1 and c2:
            counts["sim_only_correct"] += 1
        else:
            counts["neither_correct"] += 1

        details.append(
            {
                "id": ex.get("id", ""),
                "gold_1": gold_1,
                "gold_2": gold_2,
                "pred_1": pred_1,
                "pred_2": pred_2,
                "correct_1": c1,
                "correct_2": c2,
                "both_correct": c1 and c2,
                "mode": args.mode,
                "output_1": output_1,
                "output_2": output_2,
                "output_dual": output_dual,
            }
        )

    total = counts["total"]

    metrics = {
        "total": total,
        "mode": args.mode,
        "model_path": args.model_path,
        "test_file": args.test_file,
        "max_new_tokens": args.max_new_tokens,
        "orig_accuracy": counts["orig_correct"] / total if total else 0.0,
        "similar_accuracy": counts["sim_correct"] / total if total else 0.0,
        "both_correct_accuracy": counts["both_correct"] / total if total else 0.0,
        "orig_only_rate": counts["orig_only_correct"] / total if total else 0.0,
        "sim_only_rate": counts["sim_only_correct"] / total if total else 0.0,
        "neither_correct_rate": counts["neither_correct"] / total if total else 0.0,
        "format_fail_1_rate": counts["format_fail_1"] / total if total else 0.0,
        "format_fail_2_rate": counts["format_fail_2"] / total if total else 0.0,
        "counts": counts,
    }

    print("\n" + "=" * 60)
    print(f"--- {args.mode.upper()} MOD TEST SONUÇLARI ---")
    print("=" * 60)
    print(f"Toplam örnek: {total}")
    print(f"Orijinal Soru Doğruluğu      : %{metrics['orig_accuracy'] * 100:.2f}")
    print(f"Benzer Soru Doğruluğu        : %{metrics['similar_accuracy'] * 100:.2f}")
    print(f"İkisi Birden Doğru           : %{metrics['both_correct_accuracy'] * 100:.2f}")
    print(f"Sadece Orijinal Doğru        : %{metrics['orig_only_rate'] * 100:.2f}")
    print(f"Sadece Benzer Doğru          : %{metrics['sim_only_rate'] * 100:.2f}")
    print(f"İkisi de Yanlış              : %{metrics['neither_correct_rate'] * 100:.2f}")
    print(f"Format Fail 1                : %{metrics['format_fail_1_rate'] * 100:.2f}")
    print(f"Format Fail 2                : %{metrics['format_fail_2_rate'] * 100:.2f}")
    print("=" * 60 + "\n")

    output = {
        "metrics": metrics,
        "details": details,
    }

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Sonuçlar kaydedildi: {output_path}")


if __name__ == "__main__":
    main()