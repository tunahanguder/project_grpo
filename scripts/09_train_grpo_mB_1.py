import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import Dataset
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer


VALID_ANSWERS = {"A", "B", "C", "D"}

# main() içinde args.max_completion_length ile güncellenecek.
MAX_COMPLETION_LENGTH = 700


def read_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    examples = []

    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))

    return examples


def format_choices(choices: Dict[str, str]) -> str:
    return "\n".join([f"{k}) {choices[k]}" for k in ["A", "B", "C", "D"]])


def build_messages_prompt(example: Dict[str, Any]) -> List[Dict[str, str]]:
    user_text = f"""
Aşağıdaki iki soruyu, yani Orijinal ve Benzer soruyu, tutarlı bir mantıkla çöz.

[SORU 1 - ORİJİNAL]
{example["question_1"]}

Seçenekler 1:
{format_choices(example["choices_1"])}

[SORU 2 - BENZER]
{example["question_2"]}

Seçenekler 2:
{format_choices(example["choices_2"])}

Kurallar:
- Önce <think> ve </think> etiketleri arasında kısa bir akıl yürütme yap.
- Her iki sorunun ortak mantığını bulmaya çalış.
- <think> içinde gereksiz tekrar yapma.
- <think> içinde en fazla 6 kısa madde kullan.
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
                "çözen uzman bir yapay zekâ modelisin. Gereksiz uzun açıklama yapmazsın."
            ),
        },
        {
            "role": "user",
            "content": user_text,
        },
    ]


def extract_dual_answers(text: str) -> Tuple[Optional[str], Optional[str]]:
    if text is None:
        return None, None

    text = str(text).strip()

    patterns_1 = [
        r"nihai\s+cevap\s*1\s*:\s*([ABCD])",
        r"cevap\s*1\s*:\s*([ABCD])",
        r"answer\s*1\s*:\s*([ABCD])",
        r"final\s+answer\s*1\s*:\s*([ABCD])",
    ]

    patterns_2 = [
        r"nihai\s+cevap\s*2\s*:\s*([ABCD])",
        r"cevap\s*2\s*:\s*([ABCD])",
        r"answer\s*2\s*:\s*([ABCD])",
        r"final\s+answer\s*2\s*:\s*([ABCD])",
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


def normalize_completion(completion: Any) -> str:
    if isinstance(completion, str):
        return completion

    if isinstance(completion, list):
        if len(completion) == 0:
            return ""

        first = completion[0]

        if isinstance(first, dict) and "content" in first:
            return str(first["content"])

        return str(first)

    if isinstance(completion, dict) and "content" in completion:
        return str(completion["content"])

    return str(completion)


def estimate_token_count(text: str) -> float:
    """
    Hızlı yaklaşık token hesabı.
    Gerçek tokenizer tokenı değildir ama reward içinde pratik ve hızlıdır.
    """
    return len(text) / 4.0

REWARD_TOKENIZER = None


def set_reward_tokenizer(tokenizer):
    global REWARD_TOKENIZER
    REWARD_TOKENIZER = tokenizer


def count_completion_tokens(text: str) -> int:
    global REWARD_TOKENIZER

    if REWARD_TOKENIZER is not None:
        return len(
            REWARD_TOKENIZER.encode(
                text,
                add_special_tokens=False,
            )
        )

    return int(estimate_token_count(text))

def paired_answer_format_length_reward_func(
    completions,
    answer_1=None,
    answer_2=None,
    **kwargs,
):
    """
    Hiyerarşik mB reward tasarımı:

    1. Önce uzunluk kontrol edilir.
       token_count >= 450 ise reward = -0.2

    2. token_count < 450 ise format kontrol edilir.
       Format bozuksa reward = 0.1

    3. Format doğruysa:
       format reward = +0.2

    4. Format doğruysa doğruluk kontrol edilir.
       Sadece biri doğruysa +0.4
       İkisi doğruysa +1.0

    Maksimum reward = 1.2
    """
    rewards = []

    token_limit = 450

    for completion, gold_1, gold_2 in zip(completions, answer_1, answer_2):
        text = normalize_completion(completion)

        token_count = count_completion_tokens(text)

        # 1) Önce uzunluk kontrolü.
        if token_count >= token_limit:
            rewards.append(-0.2)
            continue

        pred_1, pred_2 = extract_dual_answers(text)

        gold_1 = str(gold_1).strip().upper()
        gold_2 = str(gold_2).strip().upper()

        has_valid_dual_format = (
            pred_1 in VALID_ANSWERS and pred_2 in VALID_ANSWERS
        )

        # 2) Kısa ama format bozuksa teselli ödülü.
        if not has_valid_dual_format:
            rewards.append(0.1)
            continue

        correct_1 = pred_1 == gold_1
        correct_2 = pred_2 == gold_2

        reward = 0.2  # format reward

        # 3) Doğruluk reward'u.
        if correct_1 and correct_2:
            reward += 1.0
        elif correct_1 or correct_2:
            reward += 0.4

        rewards.append(float(reward))

    return rewards


def build_dataset(input_file: Path, limit: Optional[int] = None) -> Dataset:
    raw_examples = read_jsonl(input_file)

    if limit is not None:
        raw_examples = raw_examples[:limit]

    records = []

    for ex in raw_examples:
        records.append(
            {
                "id": ex.get("id", ""),
                "prompt": build_messages_prompt(ex),
                "answer_1": ex["answer_1"],
                "answer_2": ex["answer_2"],
            }
        )

    return Dataset.from_list(records)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/models/grpo_mB_1_v2")

    parser.add_argument("--limit", type=int, default=None)

    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=5e-6)

    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)

    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--max_completion_length", type=int, default=700)

    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--beta", type=float, default=0.04)

    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main():
    global MAX_COMPLETION_LENGTH

    args = parse_args()
    MAX_COMPLETION_LENGTH = args.max_completion_length

    print("[INFO] mB_1_v2 GRPO eğitimi başlıyor...")
    print(f"[INFO] SFT checkpoint: {args.model_name_or_path}")
    print(f"[INFO] Train file: {args.train_file}")
    print(f"[INFO] Output dir: {args.output_dir}")
    print(f"[INFO] num_generations: {args.num_generations}")
    print(f"[INFO] max_completion_length: {args.max_completion_length}")

    train_dataset = build_dataset(
        input_file=Path(args.train_file),
        limit=args.limit,
    )

    print(f"[INFO] GRPO train örnek sayısı: {len(train_dataset)}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    set_reward_tokenizer(tokenizer)

    reward_funcs = [paired_answer_format_length_reward_func]

    grpo_args = GRPOConfig(
        output_dir=args.output_dir,

        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,

        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,

        num_generations=args.num_generations,
        generation_batch_size=args.num_generations,

        max_completion_length=args.max_completion_length,

        temperature=args.temperature,
        top_p=args.top_p,
        beta=args.beta,

        logging_steps=args.logging_steps,

        save_strategy="no",

        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),

        report_to=["tensorboard"],
        remove_unused_columns=False,

        seed=args.seed,
    )

    trainer = GRPOTrainer(
        model=args.model_name_or_path,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=grpo_args,
        train_dataset=train_dataset,
    )

    trainer.train()

    print("[INFO] mB_1_v2 final modeli kaydediliyor...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("[DONE] mB_1_v2 GRPO eğitimi tamamlandı.")
    print(f"[INFO] Çıktı klasörü: {args.output_dir}")


if __name__ == "__main__":
    main()