"""
09_train_grpo_mD_teacher_gaussian_dapo.py

mD: Teacher-Guided Gaussian Process-Supervised DAPO-GRPO

Tasarım:
- mC checkpoint üzerine inşa edilir.
- Güçlü çeldiricili paired train set ile eğitilir.
- mB outcome reward mantığı korunur.
- DAPO bileşenleri kullanılır:
  * Dynamic sampling / retry
  * Clip-Higher: epsilon=0.2, epsilon_high=0.28
  * DAPO token-level loss: loss_type="dapo"
- Yeni process supervision:
  * <think> bloğu 4 yapılandırılmış adıma ayrılır.
  * Her adım için hedef token aralığı vardır.
  * Aralık dışındaki adım teacher'a sorulmadan -0.2 alır.
  * Aralık içindeki adım teacher score ile puanlanır.
  * Teacher score, step uzunluğunun aralık merkezine yakınlığına göre Gaussian katsayıyla çarpılır.

Önemli notlar:
- Online 32B teacher VRAM tüketimini artırır. İlk önce --limit 8 veya --limit 16 ile smoke test önerilir.
- Teacher, aynı prompt için üretilen G completion'ı tek çağrıda puanlayacak şekilde tasarlanmıştır.
- Bu script tek GPU / tek process için pratik bir uyarlamadır. Multi-GPU dynamic sampling için ek gather gerekir.
"""

import argparse
import hashlib
import inspect
import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

try:
    from transformers import BitsAndBytesConfig
except Exception:  # pragma: no cover
    BitsAndBytesConfig = None


VALID_ANSWERS = {"A", "B", "C", "D"}

# main() içinde güncellenecek.
MAX_COMPLETION_LENGTH = 500
DYNAMIC_NUM_GENERATIONS = 4

# mB'deki çalışan global uzunluk yorumu korunuyor.
REWARD_TOKEN_LIMIT = 450

# Teacher / process global nesneleri.
REWARD_TOKENIZER = None
TEACHER_SCORER = None
PROCESS_CONFIG = None


# =============================================================================
# Durum / log sınıfları
# =============================================================================

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
        d.pop("last_filter_scores", None)
        d.pop("last_rewards", None)
        d["dynamic_keep_ratio_seen"] = (
            self.informative_groups_seen / self.generated_groups if self.generated_groups else 0.0
        )
        return d


@dataclass
class ProcessRewardState:
    """Teacher + Gaussian process reward istatistikleri."""

    completions_seen: int = 0
    valid_format_completions: int = 0
    global_overlong_count: int = 0
    format_fail_count: int = 0
    severe_structure_fail_count: int = 0

    process_reward_sum: float = 0.0
    gaussian_inside_steps: int = 0
    gaussian_outside_steps: int = 0
    missing_steps: int = 0

    teacher_group_calls: int = 0
    teacher_cache_hits: int = 0
    teacher_parse_failures: int = 0
    teacher_disabled_calls: int = 0

    def summary(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.valid_format_completions:
            d["avg_process_reward_valid_format"] = self.process_reward_sum / self.valid_format_completions
        else:
            d["avg_process_reward_valid_format"] = 0.0
        total_steps = self.gaussian_inside_steps + self.gaussian_outside_steps + self.missing_steps
        d["process_total_steps_seen"] = total_steps
        if total_steps:
            d["gaussian_inside_step_ratio"] = self.gaussian_inside_steps / total_steps
            d["gaussian_outside_step_ratio"] = self.gaussian_outside_steps / total_steps
            d["missing_step_ratio"] = self.missing_steps / total_steps
        else:
            d["gaussian_inside_step_ratio"] = 0.0
            d["gaussian_outside_step_ratio"] = 0.0
            d["missing_step_ratio"] = 0.0
        return d


DYNAMIC_STATE = DynamicSamplingState()
PROCESS_STATE = ProcessRewardState()


# =============================================================================
# Process config
# =============================================================================

@dataclass
class StepSpec:
    name: str
    label: str
    min_tokens: int
    max_tokens: int

    @property
    def center(self) -> float:
        return (self.min_tokens + self.max_tokens) / 2.0

    def sigma(self, sigma_divisor: float) -> float:
        width = max(1.0, float(self.max_tokens - self.min_tokens))
        return max(1.0, width / sigma_divisor)


@dataclass
class ProcessConfig:
    process_weight: float = 0.2
    step_outside_penalty: float = -0.2
    teacher_parse_fallback_score: float = 0.0
    gaussian_sigma_divisor: float = 4.0
    severe_missing_steps: int = 2
    hard_structure_gate: bool = True
    disable_teacher_scoring: bool = False
    teacher_max_new_tokens: int = 256

    step_specs: Dict[str, StepSpec] = field(default_factory=dict)


# =============================================================================
# Veri / prompt yardımcıları
# =============================================================================

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
    """
    mD promptu mB/mC dual promptunun process-supervised sürümüdür.
    Yapılandırılmış 4 step, teacher ve Gaussian process reward'un güvenilir çalışması için istenir.
    """
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
- Önce <think> ve </think> etiketleri arasında kısa ve yapılandırılmış bir akıl yürütme yap.
- <think> içinde tam olarak şu 4 adımı kullan:
1. Ortak mantık:
2. Soru 1 gerekçesi:
3. Soru 2 gerekçesi:
4. Son kontrol:
- Her adımı kısa, ilgili ve seçeneklerle bağlantılı yaz.
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
                "Sen benzer çoktan seçmeli soruları tutarlı, kısa ve yapılandırılmış "
                "akıl yürütmeyle çözen uzman bir yapay zekâ modelisin. Gereksiz uzun açıklama yapmazsın."
            ),
        },
        {"role": "user", "content": user_text},
    ]


# =============================================================================
# Cevap / step çıkarımı
# =============================================================================

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

    matches_1, matches_2 = [], []
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


def set_reward_tokenizer(tokenizer) -> None:
    global REWARD_TOKENIZER
    REWARD_TOKENIZER = tokenizer


def estimate_token_count(text: str) -> int:
    return max(0, int(len(text) / 4.0))


def count_completion_tokens(text: str) -> int:
    global REWARD_TOKENIZER
    if REWARD_TOKENIZER is not None:
        return len(REWARD_TOKENIZER.encode(text, add_special_tokens=False))
    return estimate_token_count(text)


def extract_think_block(text: str) -> Optional[str]:
    if not text:
        return None
    m = re.search(r"<think>(.*?)</think>", text, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return None
    return m.group(1).strip()


def extract_reasoning_steps(text: str) -> Dict[str, str]:
    """
    <think> bloğundan 4 step'i çıkarır.
    Öncelik numbered step formatıdır:
      1. Ortak mantık:
      2. Soru 1 gerekçesi:
      3. Soru 2 gerekçesi:
      4. Son kontrol:
    """
    think = extract_think_block(text)
    result = {"step_1": "", "step_2": "", "step_3": "", "step_4": ""}
    if not think:
        return result

    # Normalize line endings.
    think = think.replace("\r\n", "\n")

    # Başlıkları yakala. Her başlığın başlangıcını bulup bir sonraki başlığa kadar al.
    heading_pattern = re.compile(
        r"(?im)^\s*(?:[-*]\s*)?(?P<num>[1-4])\s*[\.)-]\s*(?P<title>[^:\n]{0,80})\s*:\s*"
    )
    matches = list(heading_pattern.finditer(think))

    if matches:
        for idx, m in enumerate(matches):
            num = m.group("num")
            start = m.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(think)
            content = think[start:end].strip()
            result[f"step_{num}"] = content
        return result

    # Fallback: anahtar kelime başlıklarını numarasız yakala.
    key_patterns = [
        ("step_1", r"(?im)^\s*(?:ortak\s+mantık|common\s+logic)\s*:\s*"),
        ("step_2", r"(?im)^\s*(?:soru\s*1\s+gerekçesi|orijinal\s+gerekçe|question\s*1)\s*:\s*"),
        ("step_3", r"(?im)^\s*(?:soru\s*2\s+gerekçesi|benzer\s+gerekçe|question\s*2)\s*:\s*"),
        ("step_4", r"(?im)^\s*(?:son\s+kontrol|final\s+check)\s*:\s*"),
    ]

    spans = []
    for step_key, pat in key_patterns:
        mm = re.search(pat, think)
        if mm:
            spans.append((mm.start(), mm.end(), step_key))
    spans.sort(key=lambda x: x[0])

    if spans:
        for idx, (_, start_content, step_key) in enumerate(spans):
            end = spans[idx + 1][0] if idx + 1 < len(spans) else len(think)
            result[step_key] = think[start_content:end].strip()
        return result

    # Son fallback: satırları 4 parçaya bölme yok; bunu severe fail olarak bırakmak daha güvenli.
    return result


# =============================================================================
# mB outcome reward ve dynamic filter score
# =============================================================================

def compute_mb_outcome_reward_and_filter_score(
    completion: Any,
    gold_1: Any,
    gold_2: Any,
    token_limit: int,
) -> Tuple[float, int, Dict[str, Any]]:
    """
    mB reward mantığı korunur.

    filter_score sadece dynamic sampling kararı içindir:
      0 = ikisi de yanlış / format bozuk / overlong
      1 = yalnızca biri doğru
      2 = ikisi de doğru
    """
    text = normalize_completion(completion)
    token_count = count_completion_tokens(text)

    meta = {
        "status": "ok",
        "token_count": token_count,
        "pred_1": None,
        "pred_2": None,
        "correct_1": False,
        "correct_2": False,
        "valid_format": False,
    }

    if token_count >= token_limit:
        meta["status"] = "global_overlong"
        return -0.2, 0, meta

    pred_1, pred_2 = extract_dual_answers(text)
    meta["pred_1"] = pred_1
    meta["pred_2"] = pred_2

    gold_1 = str(gold_1).strip().upper()
    gold_2 = str(gold_2).strip().upper()

    has_valid_dual_format = pred_1 in VALID_ANSWERS and pred_2 in VALID_ANSWERS
    meta["valid_format"] = bool(has_valid_dual_format)

    if not has_valid_dual_format:
        meta["status"] = "format_fail"
        return 0.1, 0, meta

    correct_1 = pred_1 == gold_1
    correct_2 = pred_2 == gold_2
    meta["correct_1"] = bool(correct_1)
    meta["correct_2"] = bool(correct_2)

    reward = 0.2  # format reward
    if correct_1 and correct_2:
        reward += 1.0
    elif correct_1 or correct_2:
        reward += 0.4

    filter_score = int(correct_1) + int(correct_2)
    return float(reward), int(filter_score), meta


# =============================================================================
# Gaussian step gate / process reward
# =============================================================================

def gaussian_value(length: int, spec: StepSpec, sigma_divisor: float) -> float:
    sigma = spec.sigma(sigma_divisor)
    return float(math.exp(-((float(length) - spec.center) ** 2) / (2.0 * sigma * sigma)))


def compute_step_gaussian_info(steps: Dict[str, str], cfg: ProcessConfig) -> Dict[str, Dict[str, Any]]:
    info: Dict[str, Dict[str, Any]] = {}
    for step_key, spec in cfg.step_specs.items():
        content = steps.get(step_key, "") or ""
        length = count_completion_tokens(content)
        missing = len(content.strip()) == 0
        inside = (not missing) and (spec.min_tokens <= length <= spec.max_tokens)
        g = gaussian_value(length, spec, cfg.gaussian_sigma_divisor) if inside else 0.0
        info[step_key] = {
            "length": int(length),
            "missing": bool(missing),
            "inside": bool(inside),
            "gaussian": float(g),
            "min_tokens": spec.min_tokens,
            "max_tokens": spec.max_tokens,
            "center": spec.center,
        }
    return info


def is_severe_structure_fail(steps: Dict[str, str], cfg: ProcessConfig) -> bool:
    missing = sum(1 for k in cfg.step_specs.keys() if not (steps.get(k, "") or "").strip())
    return missing >= cfg.severe_missing_steps


# =============================================================================
# Teacher scorer
# =============================================================================

class TeacherStepScorer:
    def __init__(
        self,
        model_name_or_path: str,
        max_new_tokens: int = 256,
        load_in_4bit: bool = False,
        device_map: str = "auto",
        trust_remote_code: bool = True,
        cache_file: Optional[Path] = None,
        system_prompt: Optional[str] = None,
    ):
        self.model_name_or_path = model_name_or_path
        self.max_new_tokens = int(max_new_tokens)
        self.cache_file = cache_file
        self.cache: Dict[str, Dict[str, Dict[str, float]]] = {}
        self.system_prompt = system_prompt or (
            "Sen çoktan seçmeli soru çözümlerinde akıl yürütme adımlarını nesnel biçimde "
            "puanlayan bir process reward değerlendiricisisin. Sadece geçerli JSON döndürürsün."
        )

        if self.cache_file and self.cache_file.exists():
            try:
                with self.cache_file.open("r", encoding="utf-8") as f:
                    self.cache = json.load(f)
                print(f"[INFO] Teacher cache yüklendi: {self.cache_file} ({len(self.cache)} kayıt)")
            except Exception as e:
                print(f"[UYARI] Teacher cache okunamadı: {e}")
                self.cache = {}

        print(f"[INFO] Teacher tokenizer yükleniyor: {model_name_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

        quantization_config = None
        if load_in_4bit:
            if BitsAndBytesConfig is None:
                raise RuntimeError(
                    "--teacher_load_in_4bit kullanıldı ama BitsAndBytesConfig import edilemedi. "
                    "bitsandbytes/transformers kurulumunu kontrol et."
                )
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        print(f"[INFO] Teacher model yükleniyor: {model_name_or_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype if quantization_config is None else None,
            quantization_config=quantization_config,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )
        self.model.eval()
        print("[INFO] Teacher model hazır.")

    def save_cache(self) -> None:
        if not self.cache_file:
            return
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with self.cache_file.open("w", encoding="utf-8") as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[UYARI] Teacher cache kaydedilemedi: {e}")

    @staticmethod
    def _json_from_text(text: str) -> Optional[Dict[str, Any]]:
        if not text:
            return None
        text = text.strip()
        # Markdown fence varsa temizle.
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"\s*```$", "", text).strip()

        try:
            return json.loads(text)
        except Exception:
            pass

        # İlk JSON obje aralığını bulmayı dene.
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except Exception:
                return None
        return None

    @staticmethod
    def _normalize_teacher_scores(raw: Dict[str, Any], n: int, fallback_score: float) -> Dict[str, Dict[str, float]]:
        normalized: Dict[str, Dict[str, float]] = {}
        for i in range(1, n + 1):
            ckey = f"completion_{i}"
            cdata = raw.get(ckey, {}) if isinstance(raw, dict) else {}
            if not isinstance(cdata, dict):
                cdata = {}
            normalized[ckey] = {}
            for step_idx in range(1, 5):
                skey = f"step_{step_idx}"
                val = cdata.get(skey, fallback_score)
                try:
                    fval = float(val)
                except Exception:
                    fval = float(fallback_score)
                # Teacher puanını 0-1 aralığına sıkıştır.
                fval = max(0.0, min(1.0, fval))
                normalized[ckey][skey] = fval
        return normalized

    def build_teacher_prompt(
        self,
        example: Dict[str, Any],
        completion_texts: List[str],
        step_infos: List[Dict[str, Dict[str, Any]]],
        steps_list: List[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        q1 = example.get("question_1", "")
        q2 = example.get("question_2", "")
        c1 = example.get("choices_1", {}) or {}
        c2 = example.get("choices_2", {}) or {}
        a1 = example.get("answer_1", "")
        a2 = example.get("answer_2", "")

        completions_block = []
        for idx, text in enumerate(completion_texts, start=1):
            steps = steps_list[idx - 1]
            ginfo = step_infos[idx - 1]
            step_block = []
            for sidx in range(1, 5):
                skey = f"step_{sidx}"
                step_text = steps.get(skey, "") or ""
                length = ginfo.get(skey, {}).get("length", 0)
                inside = ginfo.get(skey, {}).get("inside", False)
                step_block.append(
                    f"{skey} | token_len={length} | gaussian_range_ok={inside}\n{step_text}"
                )
            completions_block.append(
                f"[COMPLETION {idx}]\n" + "\n\n".join(step_block) + "\n[FINAL OUTPUT]\n" + text
            )

        user_text = f"""
Aşağıda iki çoktan seçmeli soru, altın cevaplar ve aynı öğrenci modelinin aynı soru çifti için ürettiği {len(completion_texts)} farklı çözüm verilmiştir.

Görevin:
Her completion içindeki 4 reasoning adımını 0.0 ile 1.0 arasında puanla.

Puanlama ölçütleri:
- 1.0: Adım açık, ilgili, mantıksal olarak doğru, seçeneklerle bağlantılı ve final cevapla uyumlu.
- 0.5: Adım kısmen ilgili ama eksik, yüzeysel veya belirsiz.
- 0.0: Adım boş, ilgisiz, yanlış, çeldiricileri karıştırıyor veya final cevapla çelişiyor.

Dikkat:
- Altın cevapları kullanabilirsin; bu bir eğitim reward değerlendirmesidir.
- Uzunluk/gaussian_range_ok bilgisini sadece yardımcı bağlam olarak gör. Esas puanın adımın mantıksal kalitesi ve ilgisine göre olmalı.
- Sadece geçerli JSON döndür. Açıklama yazma.

[SORU 1 - ORİJİNAL]
{q1}

Seçenekler 1:
{format_choices(c1) if isinstance(c1, dict) and c1 else c1}

Altın cevap 1: {a1}

[SORU 2 - BENZER]
{q2}

Seçenekler 2:
{format_choices(c2) if isinstance(c2, dict) and c2 else c2}

Altın cevap 2: {a2}

Beklenen JSON şeması:
{{
  "completion_1": {{"step_1": 0.0, "step_2": 0.0, "step_3": 0.0, "step_4": 0.0}},
  "completion_2": {{"step_1": 0.0, "step_2": 0.0, "step_3": 0.0, "step_4": 0.0}}
}}

Çözümler:
{chr(10).join(completions_block)}
""".strip()

        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_text},
        ]

    def score_group(
        self,
        example: Dict[str, Any],
        completion_texts: List[str],
        step_infos: List[Dict[str, Dict[str, Any]]],
        steps_list: List[Dict[str, str]],
        fallback_score: float,
    ) -> Dict[str, Dict[str, float]]:
        # Cache key: soru id + cevap metinleri.
        raw_key = json.dumps(
            {
                "id": example.get("id", ""),
                "a1": example.get("answer_1", ""),
                "a2": example.get("answer_2", ""),
                "completions": completion_texts,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        cache_key = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
        if cache_key in self.cache:
            PROCESS_STATE.teacher_cache_hits += 1
            return self.cache[cache_key]

        messages = self.build_teacher_prompt(example, completion_texts, step_infos, steps_list)
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192)

        # device_map='auto' için input'u ilk parametre cihazına taşımak yeterli olur.
        try:
            input_device = next(self.model.parameters()).device
        except StopIteration:
            input_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs = {k: v.to(input_device) for k, v in inputs.items()}

        PROCESS_STATE.teacher_group_calls += 1
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        gen_ids = outputs[0][inputs["input_ids"].shape[1] :]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        raw_json = self._json_from_text(text)

        if raw_json is None:
            PROCESS_STATE.teacher_parse_failures += 1
            scores = self._normalize_teacher_scores({}, len(completion_texts), fallback_score)
        else:
            scores = self._normalize_teacher_scores(raw_json, len(completion_texts), fallback_score)

        self.cache[cache_key] = scores
        return scores


# =============================================================================
# Reward helpers
# =============================================================================

def group_values(values: List[Any], group_size: int) -> List[List[Any]]:
    if group_size <= 0:
        return []
    return [
        values[i : i + group_size]
        for i in range(0, len(values), group_size)
        if len(values[i : i + group_size]) == group_size
    ]


def is_informative_scores(scores: List[int]) -> bool:
    if not scores:
        return False
    return min(scores) < max(scores)


def safe_get_list(kwargs: Dict[str, Any], key: str, n: int, default: Any = None) -> List[Any]:
    val = kwargs.get(key, None)
    if val is None:
        return [default for _ in range(n)]
    if isinstance(val, list):
        if len(val) == n:
            return val
        if len(val) == 1 and n > 1:
            return val * n
        # Uzunluk farklıysa güvenli doldur.
        return (val + [default] * n)[:n]
    return [val for _ in range(n)]


def build_example_from_group(kwargs: Dict[str, Any], start: int, group_len: int) -> Dict[str, Any]:
    """Reward kwargs içindeki dataset kolonlarından grup için tek örnek çıkarır."""
    def pick(key: str, default: Any = "") -> Any:
        vals = safe_get_list(kwargs, key, start + group_len, default)
        if start < len(vals):
            return vals[start]
        return default

    return {
        "id": pick("id", ""),
        "question_1": pick("question_1", ""),
        "question_2": pick("question_2", ""),
        "choices_1": pick("choices_1", {}),
        "choices_2": pick("choices_2", {}),
        "answer_1": pick("answer_1", ""),
        "answer_2": pick("answer_2", ""),
    }


def compute_process_rewards_for_group(
    example: Dict[str, Any],
    completion_texts: List[str],
    cfg: ProcessConfig,
) -> List[float]:
    """
    Bir prompt grubundaki G completion için process reward hesaplar.
    Teacher tek çağrıda tüm grubu puanlar.
    """
    global TEACHER_SCORER

    steps_list: List[Dict[str, str]] = []
    step_infos: List[Dict[str, Dict[str, Any]]] = []
    severe_fails: List[bool] = []

    for text in completion_texts:
        steps = extract_reasoning_steps(text)
        info = compute_step_gaussian_info(steps, cfg)
        severe = is_severe_structure_fail(steps, cfg)
        steps_list.append(steps)
        step_infos.append(info)
        severe_fails.append(severe)

    # Teacher scoring kapalıysa fallback skorlarla ilerle.
    if cfg.disable_teacher_scoring or TEACHER_SCORER is None:
        PROCESS_STATE.teacher_disabled_calls += 1
        teacher_scores = {
            f"completion_{i}": {f"step_{j}": float(cfg.teacher_parse_fallback_score) for j in range(1, 5)}
            for i in range(1, len(completion_texts) + 1)
        }
    else:
        teacher_scores = TEACHER_SCORER.score_group(
            example=example,
            completion_texts=completion_texts,
            step_infos=step_infos,
            steps_list=steps_list,
            fallback_score=cfg.teacher_parse_fallback_score,
        )

    process_rewards: List[float] = []
    for cidx, _ in enumerate(completion_texts, start=1):
        if severe_fails[cidx - 1] and cfg.hard_structure_gate:
            PROCESS_STATE.severe_structure_fail_count += 1
            process_rewards.append(float(cfg.step_outside_penalty))
            # Missing adımları logla.
            for skey in cfg.step_specs.keys():
                if step_infos[cidx - 1][skey]["missing"]:
                    PROCESS_STATE.missing_steps += 1
            continue

        step_rewards = []
        ckey = f"completion_{cidx}"
        for sidx in range(1, 5):
            skey = f"step_{sidx}"
            sinfo = step_infos[cidx - 1][skey]

            if sinfo["missing"]:
                PROCESS_STATE.missing_steps += 1
                step_rewards.append(float(cfg.step_outside_penalty))
                continue

            if not sinfo["inside"]:
                PROCESS_STATE.gaussian_outside_steps += 1
                step_rewards.append(float(cfg.step_outside_penalty))
                continue

            PROCESS_STATE.gaussian_inside_steps += 1
            tscore = teacher_scores.get(ckey, {}).get(skey, cfg.teacher_parse_fallback_score)
            try:
                tscore = float(tscore)
            except Exception:
                tscore = float(cfg.teacher_parse_fallback_score)
            tscore = max(0.0, min(1.0, tscore))
            step_rewards.append(float(sinfo["gaussian"]) * tscore)

        if step_rewards:
            process_rewards.append(float(sum(step_rewards) / len(step_rewards)))
        else:
            process_rewards.append(float(cfg.step_outside_penalty))

    return process_rewards


def teacher_gaussian_process_reward_func(
    completions,
    answer_1=None,
    answer_2=None,
    **kwargs,
):
    """
    mD reward:
      R_total = R_mB_outcome + process_weight * R_teacher_gaussian

    Hiyerarşi:
    - Global overlong: -0.2, process'e geçme.
    - Format fail: 0.1, process'e geçme.
    - Format doğruysa outcome + process.
    """
    global PROCESS_CONFIG
    cfg = PROCESS_CONFIG
    if cfg is None:
        raise RuntimeError("PROCESS_CONFIG set edilmemiş.")

    n = len(completions)
    answer_1 = answer_1 or [None] * n
    answer_2 = answer_2 or [None] * n

    rewards: List[float] = [0.0] * n
    filter_scores: List[int] = [0] * n

    # Önce mB outcome ve format/overlong durumlarını hesapla.
    valid_for_process_indices = set()
    completion_texts = [normalize_completion(c) for c in completions]

    for i, (completion, gold_1, gold_2) in enumerate(zip(completions, answer_1, answer_2)):
        PROCESS_STATE.completions_seen += 1
        outcome_reward, filter_score, meta = compute_mb_outcome_reward_and_filter_score(
            completion=completion,
            gold_1=gold_1,
            gold_2=gold_2,
            token_limit=REWARD_TOKEN_LIMIT,
        )
        rewards[i] = float(outcome_reward)
        filter_scores[i] = int(filter_score)

        if meta["status"] == "global_overlong":
            PROCESS_STATE.global_overlong_count += 1
            continue
        if meta["status"] == "format_fail":
            PROCESS_STATE.format_fail_count += 1
            continue

        PROCESS_STATE.valid_format_completions += 1
        valid_for_process_indices.add(i)

    # G completion'lık gruplar halinde teacher + Gaussian process reward.
    for start in range(0, n, DYNAMIC_NUM_GENERATIONS):
        end = min(start + DYNAMIC_NUM_GENERATIONS, n)
        if end - start <= 0:
            continue

        group_indices = list(range(start, end))
        group_texts = [completion_texts[i] for i in group_indices]
        example = build_example_from_group(kwargs, start=start, group_len=len(group_indices))

        # Sadece formatı doğru olan completion'lar process bonus alacak.
        # Teacher'a yine tüm group_texts'i veriyoruz; skorlar indeksle eşleşiyor.
        process_rewards = compute_process_rewards_for_group(
            example=example,
            completion_texts=group_texts,
            cfg=cfg,
        )

        for local_idx, global_idx in enumerate(group_indices):
            if global_idx not in valid_for_process_indices:
                continue
            p_reward = process_rewards[local_idx] if local_idx < len(process_rewards) else 0.0
            PROCESS_STATE.process_reward_sum += float(p_reward)
            rewards[global_idx] = float(rewards[global_idx] + cfg.process_weight * p_reward)

    DYNAMIC_STATE.record_reward_call(
        rewards=rewards,
        filter_scores=filter_scores,
        num_generations=DYNAMIC_NUM_GENERATIONS,
    )

    # Opsiyonel TRL metric hook.
    log_metric = kwargs.get("log_metric", None)
    if callable(log_metric):
        try:
            log_metric("process/avg_reward", sum(rewards) / max(1, len(rewards)))
        except Exception:
            pass

    return rewards


# =============================================================================
# Dataset
# =============================================================================

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
                "question_1": ex["question_1"],
                "choices_1": ex["choices_1"],
                "answer_1": ex["answer_1"],
                "question_2": ex["question_2"],
                "choices_2": ex["choices_2"],
                "answer_2": ex["answer_2"],
            }
        )
    return Dataset.from_list(records)


# =============================================================================
# Dynamic Sampling Trainer
# =============================================================================

class DynamicSamplingGRPOTrainer(GRPOTrainer):
    """
    mC'deki dynamic sampling/retry katmanı.
    Reward fonksiyonu filter_score üretir; grup tekdüzeyse aynı batch yeniden üretilir.
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

        keep = informative_count == len(groups) if self.dynamic_require_all_groups else informative_count > 0
        return keep, {
            "num_groups": len(groups),
            "informative_groups": informative_count,
            "uniform_0": uniform_0,
            "uniform_1": uniform_1,
            "uniform_2": uniform_2,
            "reason": "keep" if keep else "retry_uniform_group_detected",
        }

    def _log_dynamic_metric(self, name: str, value: float) -> None:
        try:
            mode = "train" if self.model.training else "eval"
            self._metrics[mode][name].append(float(value))
        except Exception:
            pass

    def _generate_and_score_completions(self, inputs):
        if (not self.dynamic_sampling_enabled) or (not self.model.training):
            return super()._generate_and_score_completions(inputs)

        try:
            if self.accelerator.num_processes != 1:
                raise RuntimeError(
                    "mD dynamic sampling bu scriptte tek GPU/tek process için tasarlandı. "
                    "Multi-GPU için grup skorlarının process'ler arasında ayrıca gather edilmesi gerekir."
                )
        except AttributeError:
            pass

        last_output = None
        for attempt in range(self.dynamic_max_retries + 1):
            DYNAMIC_STATE.reset_last()
            DYNAMIC_STATE.trainer_attempt_batches += 1

            output = super()._generate_and_score_completions(inputs)
            keep, info = self._current_group_decision()
            last_output = output

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

            DYNAMIC_STATE.trainer_max_retry_reached += 1
            self._log_dynamic_metric("dynamic/max_retry_reached", 1.0)
            return output

        return last_output if last_output is not None else super()._generate_and_score_completions(inputs)


# =============================================================================
# Argümanlar / config
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--teacher_model_name_or_path", type=str, required=True)
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/models/grpo_mD_teacher_gaussian_dapo")

    parser.add_argument("--limit", type=int, default=None)

    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=2e-6)

    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)

    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--max_completion_length", type=int, default=500)

    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--beta", type=float, default=0.1)

    # DAPO.
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--epsilon_high", type=float, default=0.28)
    parser.add_argument("--loss_type", type=str, default="dapo")

    # Dynamic sampling.
    parser.add_argument("--dynamic_sampling", action="store_true", default=True)
    parser.add_argument("--no_dynamic_sampling", action="store_false", dest="dynamic_sampling")
    parser.add_argument("--dynamic_max_retries", type=int, default=3)
    parser.add_argument("--dynamic_require_all_groups", action="store_true", default=True)
    parser.add_argument("--dynamic_keep_if_any_group_informative", action="store_false", dest="dynamic_require_all_groups")

    # Process reward.
    parser.add_argument("--process_weight", type=float, default=0.2)
    parser.add_argument("--step_outside_penalty", type=float, default=-0.2)
    parser.add_argument("--gaussian_sigma_divisor", type=float, default=4.0)
    parser.add_argument("--severe_missing_steps", type=int, default=2)
    parser.add_argument("--no_hard_structure_gate", action="store_true", help="Severe step structure fail durumunda process'i direkt -0.2 yapma; step bazlı ceza uygula.")
    parser.add_argument("--teacher_parse_fallback_score", type=float, default=0.0)

    # Varsayılan step aralıkları.
    parser.add_argument("--step1_min", type=int, default=25)
    parser.add_argument("--step1_max", type=int, default=55)
    parser.add_argument("--step2_min", type=int, default=25)
    parser.add_argument("--step2_max", type=int, default=65)
    parser.add_argument("--step3_min", type=int, default=25)
    parser.add_argument("--step3_max", type=int, default=65)
    parser.add_argument("--step4_min", type=int, default=10)
    parser.add_argument("--step4_max", type=int, default=35)

    # Teacher.
    parser.add_argument("--teacher_max_new_tokens", type=int, default=256)
    parser.add_argument("--teacher_load_in_4bit", action="store_true")
    parser.add_argument("--teacher_device_map", type=str, default="auto")
    parser.add_argument("--teacher_cache_file", type=str, default=None)
    parser.add_argument("--disable_teacher_scoring", action="store_true", help="Debug için teacher çağrısını kapatır; fallback score kullanır.")

    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def build_process_config(args) -> ProcessConfig:
    step_specs = {
        "step_1": StepSpec("step_1", "Ortak mantık", args.step1_min, args.step1_max),
        "step_2": StepSpec("step_2", "Soru 1 gerekçesi", args.step2_min, args.step2_max),
        "step_3": StepSpec("step_3", "Soru 2 gerekçesi", args.step3_min, args.step3_max),
        "step_4": StepSpec("step_4", "Son kontrol", args.step4_min, args.step4_max),
    }
    return ProcessConfig(
        process_weight=args.process_weight,
        step_outside_penalty=args.step_outside_penalty,
        teacher_parse_fallback_score=args.teacher_parse_fallback_score,
        gaussian_sigma_divisor=args.gaussian_sigma_divisor,
        severe_missing_steps=args.severe_missing_steps,
        hard_structure_gate=not args.no_hard_structure_gate,
        disable_teacher_scoring=args.disable_teacher_scoring,
        teacher_max_new_tokens=args.teacher_max_new_tokens,
        step_specs=step_specs,
    )


def assert_dapo_trl_support() -> None:
    sig = inspect.signature(GRPOConfig)
    params = sig.parameters
    missing = [name for name in ["epsilon_high", "loss_type"] if name not in params]
    if missing:
        raise RuntimeError(
            "Bu mD scripti güncel TRL gerektiriyor. GRPOConfig içinde eksik parametre(ler): "
            f"{missing}\n\n"
            "Colab'da şu komutları çalıştırıp runtime'ı yeniden başlat:\n"
            "  !pip install -U trl transformers accelerate peft bitsandbytes\n"
            "Eğer hâlâ görünmezse:\n"
            "  !pip install -U git+https://github.com/huggingface/trl.git\n"
        )


def make_grpo_config(args) -> GRPOConfig:
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
    if "generation_batch_size" in supported:
        cfg["generation_batch_size"] = args.num_generations

    filtered = {k: v for k, v in cfg.items() if k in supported}
    return GRPOConfig(**filtered)


# =============================================================================
# Main
# =============================================================================

def main():
    global MAX_COMPLETION_LENGTH, DYNAMIC_NUM_GENERATIONS, PROCESS_CONFIG, TEACHER_SCORER

    args = parse_args()
    MAX_COMPLETION_LENGTH = args.max_completion_length
    DYNAMIC_NUM_GENERATIONS = args.num_generations
    PROCESS_CONFIG = build_process_config(args)

    assert_dapo_trl_support()

    print("[INFO] mD Teacher-Guided Gaussian Process-Supervised DAPO-GRPO eğitimi başlıyor...")
    print(f"[INFO] Student/model checkpoint: {args.model_name_or_path}")
    print(f"[INFO] Teacher model: {args.teacher_model_name_or_path}")
    print(f"[INFO] Train file: {args.train_file}")
    print(f"[INFO] Output dir: {args.output_dir}")
    print(f"[INFO] num_generations: {args.num_generations}")
    print(f"[INFO] max_completion_length: {args.max_completion_length}")
    print(f"[INFO] mB reward token limit korunuyor: {REWARD_TOKEN_LIMIT}")
    print(f"[INFO] DAPO Clip-Higher: epsilon={args.epsilon}, epsilon_high={args.epsilon_high}")
    print(f"[INFO] DAPO loss_type: {args.loss_type}")
    print(f"[INFO] Dynamic sampling: {args.dynamic_sampling}, max_retries={args.dynamic_max_retries}")
    print(f"[INFO] Process weight: {args.process_weight}")
    print(f"[INFO] Teacher 4-bit: {args.teacher_load_in_4bit}")
    print(f"[INFO] Teacher disabled/debug: {args.disable_teacher_scoring}")

    if args.per_device_train_batch_size != 1:
        print("[UYARI] Teacher-group scoring ve dynamic sampling en temiz per_device_train_batch_size=1 iken çalışır.")

    train_dataset = build_dataset(Path(args.train_file), limit=args.limit)
    print(f"[INFO] GRPO train örnek sayısı: {len(train_dataset)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    set_reward_tokenizer(tokenizer)

    reward_funcs = [teacher_gaussian_process_reward_func]
    grpo_args = make_grpo_config(args)

    # Önce student/trainer yüklenir; sonra teacher yüklenir.
    # Bu, device_map='auto' teacher'ın GPU belleğini student'tan önce kapmasını azaltır.
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

    if not args.disable_teacher_scoring:
        cache_file = Path(args.teacher_cache_file) if args.teacher_cache_file else Path(args.output_dir) / "teacher_process_cache.json"
        TEACHER_SCORER = TeacherStepScorer(
            model_name_or_path=args.teacher_model_name_or_path,
            max_new_tokens=args.teacher_max_new_tokens,
            load_in_4bit=args.teacher_load_in_4bit,
            device_map=args.teacher_device_map,
            trust_remote_code=True,
            cache_file=cache_file,
        )
    else:
        TEACHER_SCORER = None
        print("[UYARI] Teacher scoring kapalı. Process reward fallback score ile hesaplanacak.")

    trainer.train()

    print("[INFO] mD final modeli kaydediliyor...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if TEACHER_SCORER is not None:
        TEACHER_SCORER.save_cache()

    stats = {
        "dynamic_sampling": DYNAMIC_STATE.summary(),
        "process_reward": PROCESS_STATE.summary(),
        "process_config": {
            "process_weight": PROCESS_CONFIG.process_weight,
            "step_outside_penalty": PROCESS_CONFIG.step_outside_penalty,
            "gaussian_sigma_divisor": PROCESS_CONFIG.gaussian_sigma_divisor,
            "severe_missing_steps": PROCESS_CONFIG.severe_missing_steps,
            "hard_structure_gate": PROCESS_CONFIG.hard_structure_gate,
            "teacher_parse_fallback_score": PROCESS_CONFIG.teacher_parse_fallback_score,
            "step_specs": {
                k: {
                    "label": v.label,
                    "min_tokens": v.min_tokens,
                    "max_tokens": v.max_tokens,
                    "center": v.center,
                    "sigma": v.sigma(PROCESS_CONFIG.gaussian_sigma_divisor),
                }
                for k, v in PROCESS_CONFIG.step_specs.items()
            },
        },
        "args": vars(args),
    }
    stats_path = output_dir / "mD_teacher_gaussian_process_stats.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("[DONE] mD Teacher-Guided Gaussian Process-Supervised DAPO-GRPO eğitimi tamamlandı.")
    print(f"[INFO] Çıktı klasörü: {args.output_dir}")
    print(f"[INFO] İstatistikler: {stats_path}")


if __name__ == "__main__":
    main()
