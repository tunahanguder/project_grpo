"""
06_train_sft.py

Bu dosyanın amacı:
- data/processed/final/sft_train.jsonl dosyasını okumak
- Student modeli SFT ile eğitmek
- Varsayılan olarak full fine-tuning yapmak
- Gerekirse LoRA/QLoRA modunda çalışmak
- Eğitilmiş modeli / adapter'ı outputs/models/sft altına kaydetmek

Önemli:
- Prompt kısmında doğru cevap yoktur.
- Cevap yalnızca assistant response içinde bulunur.
- Loss sadece assistant cevabı üzerinde hesaplanır.
  Yani model kullanıcı prompt'unu ezberlemek yerine cevabı üretmeyi öğrenir.
"""

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    set_seed,
)

try:
    from peft import (
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
    )
except ImportError:
    LoraConfig = None
    get_peft_model = None
    prepare_model_for_kbit_training = None


IGNORE_INDEX = -100


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


def split_train_validation(
    examples: List[Dict[str, Any]],
    val_ratio: float,
    seed: int,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    SFT train dosyasından küçük bir validation bölümü ayırır.

    Elimizde teacher-generated ayrı bir SFT dev dosyası olmadığı için
    burada final SFT train içinden küçük bir oran ayırıyoruz.

    val_ratio=0 ise validation kullanılmaz.
    """

    if val_ratio <= 0:
        return examples, []

    rng = random.Random(seed)
    shuffled = list(examples)
    rng.shuffle(shuffled)

    val_size = int(len(shuffled) * val_ratio)

    if val_size <= 0:
        return examples, []

    val_examples = shuffled[:val_size]
    train_examples = shuffled[val_size:]

    return train_examples, val_examples


def get_torch_dtype(dtype_name: str) -> torch.dtype:
    """
    Komut satırından gelen dtype stringini torch dtype'a çevirir.
    """

    if dtype_name == "bf16":
        return torch.bfloat16

    if dtype_name == "fp16":
        return torch.float16

    if dtype_name == "fp32":
        return torch.float32

    raise ValueError(f"Desteklenmeyen dtype: {dtype_name}")


def supports_bf16() -> bool:
    """
    GPU bf16 destekliyor mu?
    RTX A6000 Ampere olduğu için genelde bf16 desteği vardır.
    Yine de güvenli kontrol yapıyoruz.
    """

    if not torch.cuda.is_available():
        return False

    return torch.cuda.is_bf16_supported()


def build_fallback_text(messages: List[Dict[str, str]]) -> str:
    """
    Eğer tokenizer chat_template desteklemezse basit bir metin oluşturur.
    """

    system_text = messages[0]["content"]
    user_text = messages[1]["content"]
    assistant_text = messages[2]["content"]

    return (
        f"Sistem: {system_text}\n\n"
        f"Kullanıcı:\n{user_text}\n\n"
        f"Asistan:\n{assistant_text}"
    )


def build_fallback_prompt_text(messages: List[Dict[str, str]]) -> str:
    """
    Fallback durumda assistant cevabı hariç prompt metni.
    """

    system_text = messages[0]["content"]
    user_text = messages[1]["content"]

    return (
        f"Sistem: {system_text}\n\n"
        f"Kullanıcı:\n{user_text}\n\n"
        f"Asistan:\n"
    )


class ChatSFTDataset(Dataset):
    """
    Chat formatındaki SFT verisini tokenize eder.

    Her örnekte:
    - messages = [system, user, assistant]
    vardır.

    Eğitimde loss yalnızca assistant cevabında hesaplanır.
    Bunun için prompt tokenları label tarafında -100 yapılır.
    """

    def __init__(
        self,
        examples: List[Dict[str, Any]],
        tokenizer,
        max_seq_length: int,
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self) -> int:
        return len(self.examples)

    def _format_with_chat_template(self, messages: List[Dict[str, str]]) -> tuple[str, str]:
        """
        full_text:
            system + user + assistant response

        prompt_text:
            system + user + assistant generation başlangıcı

        prompt_text uzunluğu kadar label maskelenir.
        """

        prompt_messages = messages[:2]

        if self.tokenizer.chat_template is not None:
            prompt_text = self.tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            full_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )

            return prompt_text, full_text

        prompt_text = build_fallback_prompt_text(messages)
        full_text = build_fallback_text(messages)

        return prompt_text, full_text

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        example = self.examples[index]
        messages = example["messages"]

        prompt_text, full_text = self._format_with_chat_template(messages)

        prompt_ids = self.tokenizer(
            prompt_text,
            add_special_tokens=False,
        )["input_ids"]

        full_ids = self.tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_seq_length,
        )["input_ids"]

        if len(full_ids) == 0:
            full_ids = [self.tokenizer.eos_token_id]

        labels = list(full_ids)

        prompt_len = min(len(prompt_ids), len(labels))

        for i in range(prompt_len):
            labels[i] = IGNORE_INDEX

        # Eğer truncation yüzünden assistant kısmı tamamen kesildiyse,
        # en azından son birkaç token üzerinde loss oluşsun diye güvenlik önlemi.
        if all(label == IGNORE_INDEX for label in labels):
            keep = min(8, len(labels))
            for i in range(len(labels) - keep, len(labels)):
                labels[i] = full_ids[i]

        attention_mask = [1] * len(full_ids)

        return {
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class DataCollatorForCausalLM:
    """
    Batch içindeki değişken uzunluklu örnekleri pad eder.

    input_ids pad token ile,
    attention_mask 0 ile,
    labels ise -100 ile pad edilir.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        max_length = max(feature["input_ids"].shape[0] for feature in features)

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        pad_token_id = self.tokenizer.pad_token_id

        for feature in features:
            input_ids = feature["input_ids"]
            attention_mask = feature["attention_mask"]
            labels = feature["labels"]

            pad_length = max_length - input_ids.shape[0]

            if pad_length > 0:
                input_ids = torch.cat(
                    [
                        input_ids,
                        torch.full((pad_length,), pad_token_id, dtype=torch.long),
                    ]
                )

                attention_mask = torch.cat(
                    [
                        attention_mask,
                        torch.zeros((pad_length,), dtype=torch.long),
                    ]
                )

                labels = torch.cat(
                    [
                        labels,
                        torch.full((pad_length,), IGNORE_INDEX, dtype=torch.long),
                    ]
                )

            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(labels)

        return {
            "input_ids": torch.stack(batch_input_ids),
            "attention_mask": torch.stack(batch_attention_mask),
            "labels": torch.stack(batch_labels),
        }


def load_tokenizer(model_name: str):
    """
    Tokenizer yükler.
    Qwen modellerinde pad_token bazen None olabilir.
    Bu durumda eos_token pad olarak atanır.
    """

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    return tokenizer


def load_model_full(
    model_name: str,
    dtype: torch.dtype,
):
    """
    Full fine-tuning için modeli yükler.

    Burada quantization kullanmıyoruz.
    Tüm parametreler eğitilebilir olur.
    """

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    model.config.use_cache = False

    return model


def load_model_lora(
    model_name: str,
    dtype: torch.dtype,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
):
    """
    LoRA/QLoRA fallback modu.

    Model 4-bit yüklenir ve sadece LoRA adapter parametreleri eğitilir.
    """

    if LoraConfig is None:
        raise ImportError(
            "peft kurulu değil. Lora modu için şu kurulmalı: pip install peft bitsandbytes"
        )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True,
    )

    model.config.use_cache = False

    model = prepare_model_for_kbit_training(model)

    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3.5-4B",
        help=(
            "Student model adı. Qwen3.5-4B AutoModelForCausalLM ile sorun çıkarırsa "
            "Qwen/Qwen3-4B-Instruct-2507 denenebilir."
        ),
    )

    parser.add_argument(
        "--train_file",
        type=str,
        default="data/processed/final/sft_train.jsonl",
        help="Final SFT eğitim dosyası.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/models/sft",
        help="SFT model çıktılarının kaydedileceği klasör.",
    )

    parser.add_argument(
        "--training_mode",
        type=str,
        default="full",
        choices=["full", "lora"],
        help="full: tüm parametreleri eğitir; lora: LoRA/QLoRA fallback.",
    )

    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--num_train_epochs",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.03,
    )

    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--save_steps",
        type=int,
        default=250,
    )

    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.03,
        help="Train dosyasından ayrılacak küçük validation oranı. 0 yapılırsa eval yok.",
    )

    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Smoke test için ilk N örneği kullanır.",
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "bf16", "fp16", "fp32"],
    )

    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Bellek tasarrufu için gradient checkpointing açar.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    # LoRA parametreleri
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    train_path = Path(args.train_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] SFT eğitimi başlıyor...")
    print(f"[INFO] Model: {args.model_name}")
    print(f"[INFO] Training mode: {args.training_mode}")
    print(f"[INFO] Train file: {train_path}")
    print(f"[INFO] Output dir: {output_dir}")

    examples = read_jsonl(train_path)

    if args.max_train_samples is not None:
        examples = examples[: args.max_train_samples]
        print(f"[INFO] Smoke test aktif. Kullanılan örnek sayısı: {len(examples)}")
    else:
        print(f"[INFO] Toplam SFT örnek sayısı: {len(examples)}")

    train_examples, eval_examples = split_train_validation(
        examples=examples,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    print(f"[INFO] Train örnek sayısı: {len(train_examples)}")
    print(f"[INFO] Eval örnek sayısı: {len(eval_examples)}")

    if args.dtype == "auto":
        if supports_bf16():
            dtype = torch.bfloat16
            use_bf16 = True
            use_fp16 = False
            print("[INFO] dtype auto -> bf16 kullanılacak.")
        else:
            dtype = torch.float16
            use_bf16 = False
            use_fp16 = True
            print("[INFO] dtype auto -> fp16 kullanılacak.")
    else:
        dtype = get_torch_dtype(args.dtype)
        use_bf16 = dtype == torch.bfloat16
        use_fp16 = dtype == torch.float16

    tokenizer = load_tokenizer(args.model_name)

    if args.training_mode == "full":
        model = load_model_full(
            model_name=args.model_name,
            dtype=dtype,
        )
    else:
        model = load_model_lora(
            model_name=args.model_name,
            dtype=dtype,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        )

    if args.gradient_checkpointing:
        print("[INFO] Gradient checkpointing aktif.")
        model.gradient_checkpointing_enable()

    train_dataset = ChatSFTDataset(
        examples=train_examples,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
    )

    eval_dataset = None

    if len(eval_examples) > 0:
        eval_dataset = ChatSFTDataset(
            examples=eval_examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
        )

    data_collator = DataCollatorForCausalLM(tokenizer)

    optim_name = "adamw_torch" if args.training_mode == "full" else "paged_adamw_8bit"

    training_args = TrainingArguments(
        output_dir=str(output_dir),

        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,

        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",

        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,

        bf16=use_bf16,
        fp16=use_fp16,

        optim=optim_name,
        report_to=["tensorboard"],

        remove_unused_columns=False,
        dataloader_num_workers=2,

        save_strategy="steps",
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=args.save_steps if eval_dataset is not None else None,

        gradient_checkpointing=args.gradient_checkpointing,
    )

    trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    processing_class=tokenizer,
    )

    trainer.train()

    print("[INFO] Model kaydediliyor...")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    print("[DONE] SFT eğitimi tamamlandı.")
    print(f"[INFO] Çıktı klasörü: {output_dir}")


if __name__ == "__main__":
    main()