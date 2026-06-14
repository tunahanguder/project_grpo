import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from datasets import Dataset
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer


VALID_ANSWERS = {"A", "B", "C", "D"}

# Reward içinde kullanılacak global değer.
# main() içinde args.max_completion_length ile güncellenecek.
MAX_COMPLETION_LENGTH = 350


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
Aşağıdaki çoktan seçmeli soruyu çöz.

Soru:
{example["question"]}

Seçenekler:
{format_choices(example["choices"])}

Kurallar:
- Önce <think> ve </think> etiketleri arasında kısa bir akıl yürütme yap.
- <think> içinde en fazla 4 kısa madde kullan.
- Gereksiz tekrar yapma.
- Düşünme süreci bittikten sonra son satırda mutlaka şu formatı kullan:
Nihai cevap: <A/B/C/D>
""".strip()

    return [
        {
            "role": "system",
            "content": (
                "Sen çoktan seçmeli soruları kısa ve mantıklı akıl yürütmeyle çözen "
                "uzman bir yapay zekâ modelisin. Gereksiz uzun açıklama yapmazsın."
            ),
        },
        {
            "role": "user",
            "content": user_text,
        },
    ]


def extract_final_answer(text: str) -> Optional[str]:
    if text is None:
        return None

    text = str(text).strip()

    # Sadece ana cevap formatı sayılabilecek kalıplar.
    # Açıklama içindeki "B seçeneği doğrudur" gibi ifadeleri bilerek almıyoruz.
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

    if not matches:
        return None

    # Birden fazla ana cevap formatı varsa en sondakini al.
    return matches[-1]


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
    TRL reward fonksiyonunda tokenizer'a erişmek zahmetli olduğu için
    pratikte len(text) / 4 yaklaşımını kullanıyoruz.
    """
    return len(text) / 4.0


def answer_format_length_reward_func(
    completions,
    answer=None,
    **kwargs,
):
    """
    mA_v2 hiyerarşik tek-soru reward tasarımı:

    1. token_count >= 300 ise reward = -0.2
    2. token_count < 300 ama format bozuksa reward = 0.1
    3. format doğruysa +0.2
    4. format doğru ve cevap doğruysa +1.0

    Maksimum reward = 1.2
    """
    rewards = []

    # Tek soru için 350 max length kullanıyoruz.
    # 300 token üstünü overlong kabul ediyoruz.
    token_limit = 300

    for completion, gold_answer in zip(completions, answer):
        text = normalize_completion(completion)
        token_count = count_completion_tokens(text)

        # 1) Önce uzunluk kontrolü.
        if token_count >= token_limit:
            rewards.append(-0.2)
            continue

        pred_answer = extract_final_answer(text)
        gold_answer = str(gold_answer).strip().upper()

        has_valid_format = pred_answer in VALID_ANSWERS

        # 2) Kısa ama format bozuksa teselli ödülü.
        if not has_valid_format:
            rewards.append(0.1)
            continue

        # 3) Format doğruysa format ödülü.
        reward = 0.2

        # 4) Format doğru ve cevap doğruysa doğruluk ödülü.
        if pred_answer == gold_answer:
            reward += 1.0

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
                "id": ex["id"],
                "prompt": build_messages_prompt(ex),
                "answer": ex["answer"],
            }
        )

    return Dataset.from_list(records)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/models/grpo_mA_v2")

    parser.add_argument("--limit", type=int, default=None)

    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=5e-6)

    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)

    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--max_completion_length", type=int, default=350)

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

    print("[INFO] mA_v2 GRPO eğitimi başlıyor...")
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

    reward_funcs = [answer_format_length_reward_func]

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

    print("[INFO] mA_v2 final modeli kaydediliyor...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("[DONE] mA_v2 GRPO eğitimi tamamlandı.")
    print(f"[INFO] Çıktı klasörü: {args.output_dir}")


if __name__ == "__main__":
    main()