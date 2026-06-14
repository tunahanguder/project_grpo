# GRPO-Based Multiple-Choice Reasoning Experiments

This repository contains a compact experimental pipeline for studying reinforcement learning with verifiable rewards on multiple-choice question answering (MCQA). The project focuses on OpenBookQA-style reasoning and compares several GRPO-based training variants under original/similar question consistency settings.

The core idea is simple: a model should not only answer an original multiple-choice question correctly, but should also remain consistent when the same concept is tested through a semantically similar question.

## Project Overview

The repository includes:

* dataset download, normalization, splitting, and final JSONL preparation scripts,
* teacher-generated rationales for SFT,
* teacher-generated similar questions for paired evaluation,
* GRPO training scripts for several model variants,
* evaluation scripts for single-question and paired-question testing,
* a guided Colab notebook: `project\_grpo\_guided.ipynb`.

The experiments use a student model initialized from an SFT checkpoint, then apply GRPO-style training with different reward designs.

## Model Variants

|Variant|Name|Main idea|
|-|-|-|
|SFT|Supervised fine-tuned baseline|Student model trained on teacher-generated rationales.|
|mA|Standard GRPO|Outcome reward on a single original question.|
|mB\_v1|Pair Consistency GRPO|Jointly solves original and similar questions; rewards paired correctness.|
|mB\_v2|Strong Distractor GRPO|Uses refined, stronger distractors for paired training.|
|mC|DAPO-inspired GRPO|Adds dynamic sampling/retry, Clip-Higher, and DAPO-style loss when supported by TRL.|
|mD|Teacher-Guided Gaussian Process Reward|Adds structured process supervision over the `<think>` block using Gaussian step-length weighting and optional teacher scoring.|

## Repository Structure

```text
project\_grpo/
├── project\_grpo\_guided.ipynb
├── requirements.txt
├── scripts/
│   ├── 01\_download\_datasets.py
│   ├── 02\_normalize\_datasets.py
│   ├── 03\_create\_splits.py
│   ├── 04\_generate\_sft\_rationales.py
│   ├── 05\_build\_sft\_dataset.py
│   ├── 06\_train\_sft.py
│   ├── 06a\_build\_paired\_datasets.py
│   ├── 07\_train\_grpo\_mA.py
│   ├── 07a\_extract\_single\_from\_pairs.py
│   ├── 08\_generate\_similar\_questions.py
│   ├── 09\_train\_grpo\_mB\_1.py
│   ├── 10\_evaluate\_models.py
│   ├── 11\_refine\_distractors\_existing\_pairs.py
│   ├── 12\_train\_grpo\_mC\_dapo.py
│   └── 13\_train\_grpo\_mD\_teacher\_gaussian\_dapo.py
├── data/
│   ├── raw/
│   ├── processed/
│   │   ├── normalized/
│   │   ├── splits/
│   │   └── final/
│   └── teacher\_outputs/
└── outputs/
    ├── evaluation/
    └── models/
```

`outputs/models/` is intentionally kept out of version control because trained model checkpoints are large. Evaluation JSON files can be kept for reproducibility.

## Dataset Pipeline

The data pipeline is organized as follows:

```text
raw datasets
  ↓ 01\_download\_datasets.py
raw JSONL files
  ↓ 02\_normalize\_datasets.py
normalized A/B/C/D MCQA files
  ↓ 03\_create\_splits.py
SFT pool + GRPO train/test splits
  ↓ 04\_generate\_sft\_rationales.py
teacher rationales
  ↓ 05\_build\_sft\_dataset.py
final SFT dataset
```

For paired GRPO experiments, a second branch creates similar-question pairs:

```text
OpenBookQA original split
  +
teacher-generated similar questions
  ↓ 06a\_build\_paired\_datasets.py
clean paired dataset with shuffled similar-question choices
  ↓ 07a\_extract\_single\_from\_pairs.py
single-question mA-compatible dataset
```

### Why 1000 becomes 993 and 500 becomes 493

Teacher generation does not always produce a valid similar question. Invalid records are filtered after the original and similar datasets are merged.

In the included data:

```text
openbookqa\_grpo\_train\_1000.jsonl
+ openbookqa\_grpo\_train\_1000\_similar.jsonl
→ openbook\_train\_paired\_shuffled.jsonl      # 993 valid pairs
→ openbook\_train\_993.jsonl                  # mA single-question train file

openbookqa\_test\_500.jsonl
+ openbookqa\_test\_500\_similar.jsonl
→ openbookqa\_test\_paired\_shuffled.jsonl     # 493 valid pairs
→ openbook\_test\_493.jsonl                   # mA single-question test file
```

During paired dataset construction, Question 2 choices are shuffled and `answer\_2` is updated accordingly. This reduces the risk that the model learns a shortcut such as copying the same answer letter across the original and similar questions.

## Installation

A GPU environment is strongly recommended. The experiments were designed for Colab/A100-like environments or a high-memory local GPU.

```bash
pip install -r requirements.txt
```

For the DAPO-style scripts, a recent TRL version is required. If `epsilon\_high` or `loss\_type="dapo"` is not available in your TRL installation, install the latest version:

```bash
pip install -U trl transformers accelerate peft bitsandbytes
# or, if needed:
pip install -U git+https://github.com/huggingface/trl.git
```

## Recommended Notebook Usage

The easiest way to reproduce the pipeline is to follow:

```text
project\_grpo\_guided.ipynb
```

The notebook is organized as a guided experiment runner. Expensive steps such as teacher generation and model training should be skipped if the corresponding output files already exist.

## Key Commands

### Build paired train/test datasets

```bash
python scripts/06a\_build\_paired\_datasets.py \\
  --original\_train data/processed/splits/openbookqa\_grpo\_train\_1000.jsonl \\
  --similar\_train data/teacher\_outputs/openbookqa\_grpo\_train\_1000\_similar.jsonl \\
  --output\_train data/processed/final/openbook\_train\_paired\_shuffled.jsonl \\
  --original\_test data/processed/splits/openbookqa\_test\_500.jsonl \\
  --similar\_test data/teacher\_outputs/openbookqa\_test\_500\_similar.jsonl \\
  --output\_test data/processed/final/openbookqa\_test\_paired\_shuffled.jsonl \\
  --seed 42
```

### Extract mA-compatible single-question files

```bash
python scripts/07a\_extract\_single\_from\_pairs.py \\
  --input\_train data/processed/final/openbook\_train\_paired\_shuffled.jsonl \\
  --output\_train data/processed/final/openbook\_train\_993.jsonl \\
  --input\_test data/processed/final/openbookqa\_test\_paired\_shuffled.jsonl \\
  --output\_test data/processed/final/openbook\_test\_493.jsonl
```

### Train mA

```bash
python scripts/07\_train\_grpo\_mA.py \\
  --model\_name\_or\_path outputs/models/sft\_qwen35\_4b\_full \\
  --train\_file data/processed/final/openbook\_train\_993.jsonl \\
  --output\_dir outputs/models/grpo\_mA\_v2 \\
  --num\_train\_epochs 1 \\
  --num\_generations 4 \\
  --max\_completion\_length 350
```

### Train mB

```bash
python scripts/09\_train\_grpo\_mB\_1.py \\
  --model\_name\_or\_path outputs/models/sft\_qwen35\_4b\_full \\
  --train\_file data/processed/final/openbook\_train\_paired\_shuffled.jsonl \\
  --output\_dir outputs/models/grpo\_mB\_1 \\
  --num\_train\_epochs 1 \\
  --num\_generations 4 \\
  --max\_completion\_length 700
```

### Evaluate a single-question model on the paired test set

```bash
python scripts/10\_evaluate\_models.py \\
  --model\_path outputs/models/grpo\_mA\_v2 \\
  --test\_file data/processed/final/openbookqa\_test\_paired\_shuffled.jsonl \\
  --output\_file outputs/evaluation/mA\_v2\_eval\_paired.json \\
  --mode single \\
  --max\_new\_tokens 350
```

### Evaluate a paired-question model

```bash
python scripts/10\_evaluate\_models.py \\
  --model\_path outputs/models/grpo\_mB\_1 \\
  --test\_file data/processed/final/openbookqa\_test\_paired\_shuffled.jsonl \\
  --output\_file outputs/evaluation/mB\_1\_v2\_eval\_paired.json \\
  --mode dual \\
  --max\_new\_tokens 700
```

## Included Evaluation Outputs

The repository includes JSON evaluation files under `outputs/evaluation/`. The main paired-test set contains 493 valid original/similar pairs.

|Model output file|Mode|Original Acc.|Similar Acc.|Both Correct|
|-|-:|-:|-:|-:|
|`sft\_eval\_paired.json`|single|91.08|82.35|76.67|
|`mA\_v2\_eval\_paired.json`|single|86.82|76.67|70.39|
|`mB\_1\_v2\_eval\_paired.json`|dual|91.28|88.44|84.18|
|`mB\_2\_v2\_eval\_paired.json`|dual|89.66|89.86|83.77|
|`mC\_dapo\_eval\_paired.json`|dual|91.48|89.66|86.41|
|`grpo\_mD\_process\_no\_teacher\_eval\_paired.json`|dual|93.31|89.25|85.80|

These results are included as reproducibility artifacts. Exact results may change with model versions, TRL versions, random seeds, decoding settings, and GPU environment.

## Notes on Reproducibility

* The main random seed used in the data split pipeline is `42`.
* The OpenBookQA test split is used only for final evaluation.
* ARC train data is used in the SFT pool; ARC validation is used in the SFT dev pool.
* Model checkpoints are not included in this repository.
* Teacher models such as Qwen2.5-72B-Instruct-AWQ and Qwen2.5-32B-Instruct may require high-memory GPUs.
* DAPO-related options depend on the installed TRL version.

## Files Usually Excluded from Git

Do not commit model weights or intermediate training checkpoints:

```text
outputs/models/
checkpoint-\*/
\*.safetensors
\*.bin
\*.pt
\*.pth
```

Temporary files such as notebook checkpoints, Python caches, TensorBoard logs, and compressed archives should also be excluded.

## License

Add a license before public release if this repository will be shared publicly.

## Acknowledgements

This project uses public MCQA datasets such as OpenBookQA and AI2 ARC, and builds experiments around GRPO/RLVR-style training for multiple-choice reasoning consistency.

#   p r o j e c t _ g r p o  
 