GRPO-Based Multiple-Choice Reasoning Experiments
This repository contains the code, data preparation scripts, evaluation utilities, and a guided Colab notebook for experiments on reinforcement learning with verifiable rewards in multiple-choice question answering.
The project studies OpenBookQA-style reasoning and compares several GRPO-based training variants under original/similar question consistency settings. The main motivation is to evaluate whether a model can answer an original multiple-choice question correctly and remain consistent when the same underlying concept is tested through a semantically similar question.
---
Overview
The repository includes:
dataset download, normalization, splitting, and JSONL preparation scripts,
teacher-generated rationales for supervised fine-tuning,
teacher-generated similar questions for paired training and evaluation,
GRPO training scripts for multiple model variants,
evaluation scripts for single-question and paired-question settings,
a guided Colab notebook: `notebooks/project_grpo_guided.ipynb`.
The general workflow is:
```text
SFT baseline
    ↓
mA: standard GRPO
    ↓
mB: paired consistency GRPO
    ↓
mC: DAPO-inspired GRPO
    ↓
mD: teacher-guided Gaussian process reward
```
---
Model Variants
Variant	Description
SFT	Supervised fine-tuned baseline trained on teacher-generated rationales.
mA	Standard GRPO with outcome reward on single-question multiple-choice reasoning.
mB_v1	Pair Consistency GRPO; original and similar questions are solved jointly.
mB_v2	Paired GRPO with refined, stronger distractors.
mC	DAPO-inspired GRPO with dynamic sampling/retry, Clip-Higher, and DAPO-style loss when supported by TRL.
mD	Teacher-guided Gaussian process reward with structured process supervision over the reasoning block.
---
Repository Structure
```text
project_grpo/
├── notebooks/
│   └── project_grpo_guided.ipynb
├── scripts/
│   ├── 01_download_datasets.py
│   ├── 02_normalize_datasets.py
│   ├── 03_create_splits.py
│   ├── 04_generate_sft_rationales.py
│   ├── 05_build_sft_dataset.py
│   ├── 06_train_sft.py
│   ├── 06a_build_paired_datasets.py
│   ├── 07_train_grpo_mA.py
│   ├── 07a_extract_single_from_pairs.py
│   ├── 08_generate_similar_questions.py
│   ├── 09_train_grpo_mB_1.py
│   ├── 10_evaluate_models.py
│   ├── 11_refine_distractors_existing_pairs.py
│   ├── 12_train_grpo_mC_dapo.py
│   └── 13_train_grpo_mD_teacher_gaussian_dapo.py
├── data/
│   ├── processed/
│   │   ├── normalized/
│   │   ├── splits/
│   │   └── final/
│   └── teacher_outputs/
├── outputs/
│   └── evaluation/
├── README.md
└── .gitignore
```
Model checkpoints are not included in the repository because they are large. Evaluation JSON files may be kept for reproducibility.
---
Dataset Pipeline
The main data pipeline is:
```text
raw datasets
    ↓ 01_download_datasets.py
raw JSONL files
    ↓ 02_normalize_datasets.py
normalized A/B/C/D multiple-choice files
    ↓ 03_create_splits.py
SFT pool and GRPO train/test splits
    ↓ 04_generate_sft_rationales.py
teacher rationales
    ↓ 05_build_sft_dataset.py
final SFT dataset
```
For paired GRPO experiments, a second branch is used:
```text
OpenBookQA original split
    +
teacher-generated similar questions
    ↓ 06a_build_paired_datasets.py
clean paired dataset with shuffled similar-question choices
    ↓ 07a_extract_single_from_pairs.py
single-question mA-compatible dataset
```
---
Paired Dataset Construction
Teacher generation may fail for some examples or produce invalid similar-question records. These records are removed after merging original and similar-question files.
In the included data:
```text
openbookqa_grpo_train_1000.jsonl
+
openbookqa_grpo_train_1000_similar.jsonl
→ openbook_train_paired_shuffled.jsonl
# 993 valid train pairs
→ openbook_train_993.jsonl
# single-question train file for mA
```
Similarly:
```text
openbookqa_test_500.jsonl
+
openbookqa_test_500_similar.jsonl
→ openbookqa_test_paired_shuffled.jsonl
# 493 valid test pairs
→ openbook_test_493.jsonl
# single-question test file for mA
```
During paired dataset construction, the choices of the similar question are shuffled and `answer_2` is updated accordingly. This reduces the risk of exploiting answer-letter correlations between the original and similar questions.
---
Evaluation Setting
mA is trained as a single-question model. However, for consistency analysis it is evaluated using the paired test file in `single` mode.
In this setting, the evaluator reads each pair but asks the two questions separately:
```text
question_1 → solved as a single question
question_2 → solved as a single question
```
Thus, mA does not receive the original and similar question together in the same prompt.
mB, mC, and mD use `dual` mode, where the original and similar questions are provided together in one prompt.
---
Installation
A Colab environment with a recent GPU is recommended.
```bash
pip install -U transformers accelerate datasets peft bitsandbytes trl
```
For mC and mD, a recent TRL version is required because the scripts use DAPO-related configuration options such as `epsilon_high` and `loss_type`.
If these options are not available in the installed TRL version, install the latest development version:
```bash
pip install -U git+https://github.com/huggingface/trl.git
```
---
Guided Notebook
The recommended entry point is:
```text
notebooks/project_grpo_guided.ipynb
```
The notebook documents the complete workflow, including data preparation, SFT construction, paired dataset generation, training commands, and evaluation commands.
---
Training Commands
SFT
```bash
python scripts/06_train_sft.py   --train_file data/processed/final/sft_train.jsonl   --output_dir outputs/models/sft_qwen35_4b_full
```
mA: Standard GRPO
```bash
python scripts/07_train_grpo_mA.py   --model_name_or_path outputs/models/sft_qwen35_4b_full   --train_file data/processed/final/openbook_train_993.jsonl   --output_dir outputs/models/grpo_mA_full
```
mB: Pair Consistency GRPO
```bash
python scripts/09_train_grpo_mB_1.py   --model_name_or_path outputs/models/sft_qwen35_4b_full   --train_file data/processed/final/openbook_train_paired_shuffled.jsonl   --output_dir outputs/models/grpo_mB_1
```
mC: DAPO-Inspired GRPO
```bash
python scripts/12_train_grpo_mC_dapo.py   --model_name_or_path outputs/models/grpo_mB_1   --train_file data/processed/final/openbook_train_paired_shuffled.jsonl   --output_dir outputs/models/grpo_mC_dapo
```
mD: Teacher-Guided Gaussian Process Reward
```bash
python scripts/13_train_grpo_mD_teacher_gaussian_dapo.py   --model_name_or_path outputs/models/grpo_mC_dapo   --teacher_model_name_or_path Qwen/Qwen2.5-32B-Instruct   --train_file data/processed/final/openbook_train_paired_shuffled.jsonl   --output_dir outputs/models/grpo_mD_teacher_gaussian_dapo
```
---
Evaluation Commands
Single-question evaluation:
```bash
python scripts/10_evaluate_models.py   --model_path outputs/models/grpo_mA_full   --test_file data/processed/final/openbookqa_test_paired_shuffled.jsonl   --output_file outputs/evaluation/mA_eval_single.json   --mode single
```
Paired-question evaluation:
```bash
python scripts/10_evaluate_models.py   --model_path outputs/models/grpo_mB_1   --test_file data/processed/final/openbookqa_test_paired_shuffled.jsonl   --output_file outputs/evaluation/mB_eval_dual.json   --mode dual
```
---
Reproducibility Notes
Random seeds are fixed where applicable.
Similar-question generation depends on the teacher model and sampling settings.
Large model checkpoints are intentionally excluded from the repository.
Evaluation JSON files can be used to inspect model outputs and reported metrics.
---
License
No license has been selected yet. Add a license before public reuse or redistribution.
---
Acknowledgement
This project was developed as part of an experimental study on GRPO-style reinforcement learning for multiple-choice reasoning and semantic consistency.
