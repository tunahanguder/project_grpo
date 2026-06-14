GRPO-Based Multiple-Choice Reasoning Experiments
This repository contains the code, data preparation scripts, evaluation utilities, and a guided Colab notebook for studying reinforcement learning with verifiable rewards (RLVR) on multiple-choice question answering (MCQA).
The project focuses on OpenBookQA-style reasoning and compares several GRPO-based training variants under original/similar question consistency settings. The core idea is simple:
> A model should not only answer an original multiple-choice question correctly, but should also remain consistent when the same concept is tested through a semantically similar question.
---
Project Overview
The repository includes:
dataset download, normalization, splitting, and JSONL preparation scripts,
teacher-generated rationales for SFT,
teacher-generated similar questions for paired evaluation,
GRPO training scripts for several model variants,
evaluation scripts for single-question and paired-question testing,
a guided Colab notebook: `notebooks/project_grpo_guided.ipynb`.
The experiments use a student model initialized from an SFT checkpoint, then apply GRPO-style training with different reward designs.
---
Model Variants
Variant	Name	Main idea
SFT	Supervised fine-tuned baseline	Student model trained on teacher-generated rationales
mA	Standard GRPO	Outcome reward on a single original question
mB_v1	Pair Consistency GRPO	Jointly solves original and similar questions; rewards paired correctness
mB_v2	Strong Distractor GRPO	Uses refined, stronger distractors for paired training
mC	DAPO-inspired GRPO	Adds dynamic sampling/retry, Clip-Higher, and DAPO-style loss when supported by TRL
mD	Teacher-Guided Gaussian Process Reward	Adds structured process supervision over the `<think>` block using Gaussian step-length weighting and optional teacher scoring
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
`outputs/models/` is intentionally excluded from version control because trained model checkpoints are large. Evaluation JSON files can be kept for reproducibility.
---
Dataset Pipeline
The data pipeline is organized as follows:
```text
raw datasets
    ↓ 01_download_datasets.py
raw JSONL files
    ↓ 02_normalize_datasets.py
normalized A/B/C/D MCQA files
    ↓ 03_create_splits.py
SFT pool + GRPO train/test splits
    ↓ 04_generate_sft_rationales.py
teacher rationales
    ↓ 05_build_sft_dataset.py
final SFT dataset
```
For paired GRPO experiments, a second branch creates similar-question pairs:
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
Why 1000 Becomes 993 and 500 Becomes 493
Teacher generation does not always produce a valid similar question. Invalid records are filtered after the original and similar datasets are merged.
In the included data:
```text
openbookqa_grpo_train_1000.jsonl
+
openbookqa_grpo_train_1000_similar.jsonl
→ openbook_train_paired_shuffled.jsonl
# 993 valid pairs
→ openbook_train_993.jsonl
# mA single-question train file
```
Similarly:
```text
openbookqa_test_500.jsonl
+
openbookqa_test_500_similar.jsonl
→ openbookqa_test_paired_shuffled.jsonl
# 493 valid pairs
→ openbook_test_493.jsonl
# mA single-question test file
```
During paired dataset construction, Question 2 choices are shuffled and `answer_2` is updated accordingly. This reduces the risk that the model exploits answer-letter correlations between the original and similar questions.
---
Important Evaluation Detail
mA is a single-question model, but it is evaluated on the paired test dataset in `single` mode.
That means:
```text
paired test file:
  question_1 = original question
  question_2 = similar question

single evaluation mode:
  ask question_1 alone
  ask question_2 alone
```
So mA does not receive both questions in the same prompt. It solves each question separately.
mB, mC, and mD use `dual` mode, where the original and similar questions are provided together in one prompt.
---
Installation
A typical Colab setup is recommended.
```bash
pip install -U transformers accelerate datasets peft bitsandbytes trl
```
For mC and mD, a recent TRL version is required because the scripts use DAPO-related configuration options such as `epsilon_high` and `loss_type`.
If needed:
```bash
pip install -U git+https://github.com/huggingface/trl.git
```
---
Running the Guided Notebook
The recommended entry point is:
```text
notebooks/project_grpo_guided.ipynb
```
The notebook explains the pipeline step by step and includes commands for:
dataset download,
normalization,
split creation,
SFT rationale generation,
SFT dataset construction,
paired dataset construction,
mA-compatible single dataset extraction,
SFT training,
GRPO training variants,
evaluation.
---
Training Overview
SFT
```bash
python scripts/06_train_sft.py \
  --train_file data/processed/final/sft_train.jsonl \
  --output_dir outputs/models/sft_qwen35_4b_full
```
mA: Standard GRPO
```bash
python scripts/07_train_grpo_mA.py \
  --model_name_or_path outputs/models/sft_qwen35_4b_full \
  --train_file data/processed/final/openbook_train_993.jsonl \
  --output_dir outputs/models/grpo_mA_full
```
mB: Pair Consistency GRPO
```bash
python scripts/09_train_grpo_mB_1.py \
  --model_name_or_path outputs/models/sft_qwen35_4b_full \
  --train_file data/processed/final/openbook_train_paired_shuffled.jsonl \
  --output_dir outputs/models/grpo_mB_1
```
mC: DAPO-Inspired GRPO
```bash
python scripts/12_train_grpo_mC_dapo.py \
  --model_name_or_path outputs/models/grpo_mB_1 \
  --train_file data/processed/final/openbook_train_paired_shuffled.jsonl \
  --output_dir outputs/models/grpo_mC_dapo
```
mD: Teacher-Guided Gaussian Process Reward
```bash
python scripts/13_train_grpo_mD_teacher_gaussian_dapo.py \
  --model_name_or_path outputs/models/grpo_mC_dapo \
  --teacher_model_name_or_path Qwen/Qwen2.5-32B-Instruct \
  --train_file data/processed/final/openbook_train_paired_shuffled.jsonl \
  --output_dir outputs/models/grpo_mD_teacher_gaussian_dapo
```
---
Evaluation
Single mode evaluates each question independently.
```bash
python scripts/10_evaluate_models.py \
  --model_path outputs/models/grpo_mA_full \
  --test_file data/processed/final/openbookqa_test_paired_shuffled.jsonl \
  --output_file outputs/evaluation/mA_eval_single.json \
  --mode single
```
Dual mode evaluates original and similar questions together.
```bash
python scripts/10_evaluate_models.py \
  --model_path outputs/models/grpo_mB_1 \
  --test_file data/processed/final/openbookqa_test_paired_shuffled.jsonl \
  --output_file outputs/evaluation/mB_eval_dual.json \
  --mode dual
```
---
Notes on Reproducibility
Random seed is fixed where applicable.
Similar-question generation uses a teacher model, so exact reproduction may depend on model version and sampling settings.
Large trained checkpoints are not included in the repository.
Evaluation result JSON files can be used to inspect reported metrics and model outputs.
---
License
No license has been selected yet. Add a license before public reuse or redistribution.
---
Acknowledgement
This project was developed as part of an experimental study on GRPO-style reinforcement learning for multiple-choice reasoning and semantic consistency.
