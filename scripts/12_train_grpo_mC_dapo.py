"""
09_train_grpo_mC_dapo.py

mC: mB reward + DAPO-inspired training improvements
- mB reward fonksiyonu korunur.
- Clip-Higher: epsilon=0.2, epsilon_high=0.28.
- DAPO token-level loss: loss_type="dapo".
- Dynamic sampling/retry: grup içi filter_score tekdüzeyse aynı prompt grubu yeniden üretilir.

Not:
- Bu script güncel TRL gerektirir. GRPOConfig içinde epsilon_high ve loss_type bulunmalıdır.
- Dynamic sampling kontrolü pratikte tek GPU / tek process kullanım için tasarlanmıştır.
- Mevcut overlong/length reward tasarımı korunmuştur; yeni bir overlong shaping eklenmemiştir.
"""

import argparse
import inspect
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import Dataset
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer


VALID_ANSWERS = {"A", "B", "C", "D"}

# main() içinde args.max_completion_length ile güncellenecek.
MAX_COMPLETION_LENGTH = 700

# mB'deki çalışan uzunluk yorumu korunuyor.
REWARD_TOKEN_LIMIT = 450


@dataclass
class DynamicSamplingState:
    """Reward fonksiyonu ile Trainer subclass arasında hafif durum paylaşımı."""

    last_filter_scores: List[int] = field(default_factory=list)
    last_rewards: List[float] = field(default_factory=list)

    reward_calls: int = 0
    generated_groups: int = 0
    informative_groups_seen: int = 0
    uniform_0_groups_seen: int = 0
    uniform_1_groups_seen: int = 0
    uniform_2_groups_seen: int = 0

    trainer_attempt_batches: int = 0
    trainer_kept_batches: int = 0
    trainer_retry_batches: int = 0
    trainer_max_retry_reached: int = 0

    def reset_last(self) -> None:
        self.last_filter_scores = []
        self.last_rewards = []

    def record_reward_call(self, rewards: List[float], filter_scores: List[int], num_generations: int) -> None:
        self.reward_calls += 1
        self.last_rewards = list(rewards)
        self.last_filter_scores = list(filter_scores)

        groups = group_values(filter_scores, num_generations)
        self.generated_groups += len(groups)

        for g in groups:
            if is_informative_scores(g):
                self.informative_groups_seen += 1
            else:
                uniform_value = int(g[0]) if g else -1
                if uniform_value == 0:
                    self.uniform_0_groups_seen += 1
                elif uniform_value == 1:
                    self.uniform_1_groups_seen += 1
                elif uniform_value == 2:
                    self.uniform_2_groups_seen += 1

    def summary(self) -> Dict[str, Any]:
        d = asdict(self)
        # Son batch'e ait ham listeleri özet dosyada gereksiz büyütmeyelim.
        d.pop("last_filter_scores", None)
        d.pop("last_rewards", None)
        if self.generated_groups > 0:
            d["dynamic_keep_ratio_seen"] = self.informative_groups_seen / self.generated_groups
        else:
            d["dynamic_keep_ratio_seen"] = 0.0
        return d


DYNAMIC_STATE = DynamicSamplingState()


# -----------------------------------------------------------------------------
# Veri / prompt yardımcıları
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# Cevap çıkarımı / reward yardımcıları
# -----------------------------------------------------------------------------

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


def compute_mb_reward_and_filter_score(
    completion: Any,
    gold_1: Any,
    gold_2: Any,
    token_limit: int,
) -> Tuple[float, int]:
    """
    mB reward korunur.

    filter_score sadece dynamic sampling kararı içindir:
      0 = ikisi de yanlış / format bozuk / overlong
      1 = yalnızca biri doğru
      2 = ikisi de doğru
    """
    text = normalize_completion(completion)
    token_count = count_completion_tokens(text)

    # mB'deki çalışan uzunluk yorumu aynen korunuyor.
    if token_count >= token_limit:
        return -0.2, 0

    pred_1, pred_2 = extract_dual_answers(text)

    gold_1 = str(gold_1).strip().upper()
    gold_2 = str(gold_2).strip().upper()

    has_valid_dual_format = pred_1 in VALID_ANSWERS and pred_2 in VALID_ANSWERS

    if not has_valid_dual_format:
        return 0.1, 0

    correct_1 = pred_1 == gold_1
    correct_2 = pred_2 == gold_2

    reward = 0.2  # format reward

    if correct_1 and correct_2:
        reward += 1.0
    elif correct_1 or correct_2:
        reward += 0.4

    filter_score = int(correct_1) + int(correct_2)

    return float(reward), int(filter_score)


def group_values(values: List[Any], group_size: int) -> List[List[Any]]:
    if group_size <= 0:
        return []
    return [values[i : i + group_size] for i in range(0, len(values), group_size) if len(values[i : i + group_size]) == group_size]


def is_informative_scores(scores: List[int]) -> bool:
    if not scores:
        return False
    return min(scores) < max(scores)


def paired_answer_format_length_reward_func(
    completions,
    answer_1=None,
    answer_2=None,
    **kwargs,
):
    """
    Hiyerarşik mB reward tasarımı korunmuştur.

    Ek olarak mC dynamic sampling için filter_score hesaplanır:
      filter_score = int(original_correct) + int(similar_correct)
    Bu skor reward olarak kullanılmaz; sadece grup bilgilendirici mi diye bakılır.
    """
    rewards = []
    filter_scores = []

    # TRL reward kwargs içinden loglama hook'larını alalım.
    log_metric = kwargs.get("log_metric", None)
    trainer_state = kwargs.get("trainer_state", None)

    for completion, gold_1, gold_2 in zip(completions, answer_1, answer_2):
        reward, filter_score = compute_mb_reward_and_filter_score(
            completion=completion,
            gold_1=gold_1,
            gold_2=gold_2,
            token_limit=REWARD_TOKEN_LIMIT,
        )
        rewards.append(float(reward))
        filter_scores.append(int(filter_score))

    # num_generations'a doğrudan erişemediğimiz durumlar için state'ten değil, batch yapısından ilerliyoruz.
    # main() içinde num_generations global olarak set edilmiyor; reward fonksiyonuna TRL bunu vermez.
    # Bu nedenle DYNAMIC_NUM_GENERATIONS globali kullanılır.
    DYNAMIC_STATE.record_reward_call(
        rewards=rewards,
        filter_scores=filter_scores,
        num_generations=DYNAMIC_NUM_GENERATIONS,
    )

    if callable(log_metric):
        groups = group_values(filter_scores, DYNAMIC_NUM_GENERATIONS)
        if groups:
            informative = sum(1 for g in groups if is_informative_scores(g))
            uniform_0 = sum(1 for g in groups if (not is_informative_scores(g) and g[0] == 0))
            uniform_1 = sum(1 for g in groups if (not is_informative_scores(g) and g[0] == 1))
            uniform_2 = sum(1 for g in groups if (not is_informative_scores(g) and g[0] == 2))
            log_metric("dynamic/informative_group_ratio", informative / len(groups))
            log_metric("dynamic/uniform_0_groups", float(uniform_0))
            log_metric("dynamic/uniform_1_groups", float(uniform_1))
            log_metric("dynamic/uniform_2_groups", float(uniform_2))

        # Bazı TRL sürümlerinde trainer_state None olabilir; sorun değil.
        _ = trainer_state

    return rewards


# reward fonksiyonunun grup boyutunu bilmesi için global.
DYNAMIC_NUM_GENERATIONS = 4


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# Dynamic Sampling Trainer
# -----------------------------------------------------------------------------

class DynamicSamplingGRPOTrainer(GRPOTrainer):
    """
    GRPOTrainer üzerine mC dynamic sampling/retry katmanı.

    Mantık:
    - TRL önce aynı prompt için G completion üretir ve reward hesaplar.
    - Reward fonksiyonu her completion için filter_score üretir.
    - Grup içi filter_score tekdüzeyse bu grup bilgilendirici değildir.
    - Bilgilendirici değilse aynı generation batch yeniden üretilir.
    - max retry sonunda hâlâ tekdüzeyse batch bırakılır; bu durumda avantaj zaten sıfıra yakın/sıfır olur.

    Not:
    - Bu, stock TRL içinde pratik bir requeue/retry uyarlamasıdır.
    - Tam DAPO buffer replacement için TRL trainer'ın daha derin patchlenmesi gerekir.
    """

    def __init__(
        self,
        *args,
        dynamic_sampling_enabled: bool = True,
        dynamic_max_retries: int = 3,
        dynamic_require_all_groups: bool = True,
        **kwargs,
    ):
        self.dynamic_sampling_enabled = dynamic_sampling_enabled
        self.dynamic_max_retries = int(dynamic_max_retries)
        self.dynamic_require_all_groups = bool(dynamic_require_all_groups)
        super().__init__(*args, **kwargs)

    def _current_group_decision(self) -> Tuple[bool, Dict[str, Any]]:
        scores = list(DYNAMIC_STATE.last_filter_scores)
        groups = group_values(scores, self.num_generations)

        if not groups:
            return True, {
                "num_groups": 0,
                "informative_groups": 0,
                "uniform_0": 0,
                "uniform_1": 0,
                "uniform_2": 0,
                "reason": "no_scores_available_keep_batch",
            }

        informative_flags = [is_informative_scores(g) for g in groups]
        informative_count = sum(int(x) for x in informative_flags)

        uniform_0 = sum(1 for g in groups if (not is_informative_scores(g) and g[0] == 0))
        uniform_1 = sum(1 for g in groups if (not is_informative_scores(g) and g[0] == 1))
        uniform_2 = sum(1 for g in groups if (not is_informative_scores(g) and g[0] == 2))

        if self.dynamic_require_all_groups:
            keep = informative_count == len(groups)
        else:
            keep = informative_count > 0

        info = {
            "num_groups": len(groups),
            "informative_groups": informative_count,
            "uniform_0": uniform_0,
            "uniform_1": uniform_1,
            "uniform_2": uniform_2,
            "reason": "keep" if keep else "retry_uniform_group_detected",
        }
        return keep, info

    def _log_dynamic_metric(self, name: str, value: float) -> None:
        """TRL sürüm farklarına karşı güvenli metrik loglama."""
        try:
            mode = "train" if self.model.training else "eval"
            self._metrics[mode][name].append(float(value))
        except Exception:
            # Eğitim loglaması bozulmasın.
            pass

    def _generate_and_score_completions(self, inputs):
        # Dynamic sampling sadece eğitimde uygulanır.
        if (not self.dynamic_sampling_enabled) or (not self.model.training):
            return super()._generate_and_score_completions(inputs)

        # Bu dynamic state tek-process varsayımıyla güvenlidir.
        try:
            if self.accelerator.num_processes != 1:
                raise RuntimeError(
                    "mC dynamic sampling bu scriptte tek GPU/tek process için tasarlandı. "
                    "Multi-GPU için grup skorlarının process'ler arasında ayrıca gather edilmesi gerekir."
                )
        except AttributeError:
            pass

        last_output = None
        last_info = None

        for attempt in range(self.dynamic_max_retries + 1):
            DYNAMIC_STATE.reset_last()
            DYNAMIC_STATE.trainer_attempt_batches += 1

            output = super()._generate_and_score_completions(inputs)
            keep, info = self._current_group_decision()

            last_output = output
            last_info = info

            self._log_dynamic_metric("dynamic/attempt", float(attempt))
            self._log_dynamic_metric("dynamic/batch_num_groups", float(info["num_groups"]))
            self._log_dynamic_metric("dynamic/batch_informative_groups", float(info["informative_groups"]))
            self._log_dynamic_metric("dynamic/batch_uniform_0", float(info["uniform_0"]))
            self._log_dynamic_metric("dynamic/batch_uniform_1", float(info["uniform_1"]))
            self._log_dynamic_metric("dynamic/batch_uniform_2", float(info["uniform_2"]))

            if keep:
                DYNAMIC_STATE.trainer_kept_batches += 1
                self._log_dynamic_metric("dynamic/kept_batch", 1.0)
                return output

            if attempt < self.dynamic_max_retries:
                DYNAMIC_STATE.trainer_retry_batches += 1
                self._log_dynamic_metric("dynamic/retry_batch", 1.0)
                continue

            # Max retry doldu: batch'i döndür. Uniform reward grubunda avantaj zaten öğrenme sinyalini zayıflatır/sıfırlar.
            DYNAMIC_STATE.trainer_max_retry_reached += 1
            self._log_dynamic_metric("dynamic/max_retry_reached", 1.0)
            return output

        # Teorik fallback.
        if last_output is None:
            return super()._generate_and_score_completions(inputs)
        _ = last_info
        return last_output


# -----------------------------------------------------------------------------
# Argümanlar / Config
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/models/grpo_mC_dapo")

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

    # DAPO Clip-Higher değerleri: makaledeki/pratik reçetedeki değerler.
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--epsilon_high", type=float, default=0.28)
    parser.add_argument("--loss_type", type=str, default="dapo")

    # Dynamic sampling/retry.
    parser.add_argument("--dynamic_sampling", action="store_true", default=True)
    parser.add_argument("--no_dynamic_sampling", action="store_false", dest="dynamic_sampling")
    parser.add_argument("--dynamic_max_retries", type=int, default=3)
    parser.add_argument(
        "--dynamic_require_all_groups",
        action="store_true",
        default=True,
        help="Batch içinde birden çok prompt grubu varsa hepsinin bilgilendirici olmasını ister. per_device_train_batch_size=1 için fark etmez.",
    )
    parser.add_argument(
        "--dynamic_keep_if_any_group_informative",
        action="store_false",
        dest="dynamic_require_all_groups",
        help="Batch içinde en az bir grup bilgilendiriciyse batch'i kabul eder.",
    )

    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def assert_dapo_trl_support() -> None:
    sig = inspect.signature(GRPOConfig)
    params = sig.parameters

    missing = []
    for name in ["epsilon_high", "loss_type"]:
        if name not in params:
            missing.append(name)

    if missing:
        raise RuntimeError(
            "Bu mC scripti güncel TRL gerektiriyor. GRPOConfig içinde eksik parametre(ler): "
            f"{missing}\n\n"
            "Colab'da şu komutları çalıştırıp runtime'ı yeniden başlat:\n"
            "  !pip install -U trl transformers accelerate peft\n"
            "Eğer hâlâ görünmezse:\n"
            "  !pip install -U git+https://github.com/huggingface/trl.git\n"
        )


def make_grpo_config(args) -> GRPOConfig:
    """Sürüm farklarına karşı GRPOConfig'e sadece desteklenen argümanları gönderir."""
    sig = inspect.signature(GRPOConfig)
    supported = set(sig.parameters.keys())

    cfg = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_generations": args.num_generations,
        "max_completion_length": args.max_completion_length,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "beta": args.beta,
        "epsilon": args.epsilon,
        "epsilon_high": args.epsilon_high,
        "loss_type": args.loss_type,
        "logging_steps": args.logging_steps,
        "save_strategy": "no",
        "bf16": torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        "fp16": torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        "report_to": ["tensorboard"],
        "remove_unused_columns": False,
        "seed": args.seed,
    }

    # Eski/yeni TRL sürüm farkları için opsiyonel.
    if "generation_batch_size" in supported:
        cfg["generation_batch_size"] = args.num_generations

    # Kullanıcının mevcut overlong shaping'i bozulmasın diye mask_truncated_completions eklemiyoruz.

    filtered = {k: v for k, v in cfg.items() if k in supported}
    return GRPOConfig(**filtered)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    global MAX_COMPLETION_LENGTH, DYNAMIC_NUM_GENERATIONS

    args = parse_args()
    MAX_COMPLETION_LENGTH = args.max_completion_length
    DYNAMIC_NUM_GENERATIONS = args.num_generations

    assert_dapo_trl_support()

    print("[INFO] mC DAPO-esinli GRPO eğitimi başlıyor...")
    print(f"[INFO] SFT checkpoint: {args.model_name_or_path}")
    print(f"[INFO] Train file: {args.train_file}")
    print(f"[INFO] Output dir: {args.output_dir}")
    print(f"[INFO] num_generations: {args.num_generations}")
    print(f"[INFO] max_completion_length: {args.max_completion_length}")
    print(f"[INFO] mB reward token limit korunuyor: {REWARD_TOKEN_LIMIT}")
    print(f"[INFO] DAPO Clip-Higher: epsilon={args.epsilon}, epsilon_high={args.epsilon_high}")
    print(f"[INFO] DAPO loss_type: {args.loss_type}")
    print(f"[INFO] Dynamic sampling: {args.dynamic_sampling}")
    print(f"[INFO] Dynamic max retries: {args.dynamic_max_retries}")

    if args.per_device_train_batch_size != 1:
        print(
            "[UYARI] Bu scriptte dynamic sampling en temiz per_device_train_batch_size=1 iken çalışır. "
            "Batch içinde birden çok prompt grubu varsa --dynamic_require_all_groups ayarı devreye girer."
        )

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

    grpo_args = make_grpo_config(args)

    trainer = DynamicSamplingGRPOTrainer(
        model=args.model_name_or_path,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=grpo_args,
        train_dataset=train_dataset,
        dynamic_sampling_enabled=args.dynamic_sampling,
        dynamic_max_retries=args.dynamic_max_retries,
        dynamic_require_all_groups=args.dynamic_require_all_groups,
    )

    trainer.train()

    print("[INFO] mC final modeli kaydediliyor...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stats_path = output_dir / "dynamic_sampling_stats.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(DYNAMIC_STATE.summary(), f, ensure_ascii=False, indent=2)

    print("[DONE] mC DAPO-esinli GRPO eğitimi tamamlandı.")
    print(f"[INFO] Çıktı klasörü: {args.output_dir}")
    print(f"[INFO] Dynamic sampling istatistikleri: {stats_path}")


if __name__ == "__main__":
    main()
