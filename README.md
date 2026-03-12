# Co-LoRA: Collaborative Model Personalization on Heterogeneous Multi-Modal Clients

<p align="center">
  <a href="https://openreview.net/forum?id=0g5Dk4Qfh0"><img src="https://img.shields.io/badge/ICLR_2026-Paper-blue" alt="Paper"></a>
  <a href="https://github.com/"><img src="https://img.shields.io/badge/License-MIT-green" alt="License"></a>
  <img src="https://img.shields.io/badge/Python-3.10-yellow" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.3.0-orange" alt="PyTorch">
</p>

> **Official repository** for [Co-LoRA: Collaborative Model Personalization on Heterogeneous Multi-Modal Clients (ICLR 2026)](https://openreview.net/forum?id=0g5Dk4Qfh0&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICLR.cc%2F2026%2FConference%2FAuthors%23your-submissions))

---

## 📋 Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Dataset Setup](#dataset-setup)
- [Federated Learning Configuration](#federated-learning-configuration)
- [Training](#training)
  - [Quick Start](#quick-start)
  - [Script Arguments](#script-arguments)
  - [Method-Specific Settings](#method-specific-settings)
  - [Dataset-Specific Settings](#dataset-specific-settings)
- [Evaluation](#evaluation)
- [Pretrained Checkpoints](#pretrained-checkpoints)

---

## Overview

**FedMosaic** is federated continual learning frameworks designed for heterogeneous multi-modal clients.
It is possible by the use of **Co-LoRA**, the dimension-agnostic LoRA that allows collaborative knowledge sharing among heterogeneous models. 
This repo supports training and evaluation of various federated learning baselines alongside our proposed method.

---

## Environment Setup

We recommend using a conda virtual environment.

```bash
conda create -n fcl2 python=3.10
conda activate fcl2

pip install transformers==4.47.1
pip install torch==2.3.0 torchvision --index-url https://download.pytorch.org/whl/cu118
pip install flash-attn==2.5.8 --no-build-isolation
pip install peft==0.14.0 bitsandbytes pandas kornia opencv-python timm \
    torch_optimizer easydict pycocoevalcap sentencepiece protobuf \
    trl==0.8.6 deepspeed==0.15.2 loguru captum POT jsonlines \
    numpy==1.26.4 accelerate==0.29.3 nevergrad
pip install -U scikit-learn
```

---

## Dataset Setup

Place all datasets under the `dataset/` folder at the project root.

Download all data for DRAKE from [here]().

---

## Federated Learning Configuration

Client tasks and model assignments are managed via a single JSON scenario file. For a summary of all scenarios used in the paper, see [docs/SCENARIO.md](docs/SCENARIO.md).

---

## Training

### Quick Start

```bash
bash train_VLM_CL.sh
```

### Script Arguments

Before running, edit the top section of `train_VLM_CL.sh` to configure your experiment. The key variables are:

#### General Settings

| Variable | Default | Description |
|---|---|---|
| `NOTE` | `"debug_fedmosaic"` | Experiment name used for logging and checkpoint saving |
| `MODE` | `"fedmosaic"` | Federated learning method (see [Method-Specific Settings](#method-specific-settings)) |

#### Federated Learning Settings

| Variable | Default | Description |
|---|---|---|
| `SCENARIO` | `DRAKE_hetero_llava_llama_1B_3B` | Scenario name defined in the `scenarios/` folder |
| `NUM_ROUNDS` | `5` | Number of federated communication rounds per task |
| `NUM_TASKS` | `4` | Number of tasks per client defined in the scenario JSON file |
| `NUM_CLIENTS` | `10` | Number of federated clients defined in the scenario JSOn file |
| `NUM_ITER` | `100` | Local training iterations per round (method-dependent, see below) |
| `IS_MULTIMODAL` | `True` | Enable multimodal (vision-language) training |

#### Optimization Settings

| Variable | Default | Description |
|---|---|---|
| `BATCHSIZE` | `4` | Gradient accumulation steps (effective total batch size) |
| `LR` | `2e-5` | Base learning rate for lora weights (A, B) |
| `MM_PROJECTOR_LR` | `5e-5` | Learning rate for other weights (e.g., Co-LoRA's P & Q) |
| `SCHED_NAME` | `"constant"` | LR scheduler type (`constant` or `cosine`) |

---

### Method-Specific Settings

Different federated methods require specific `NUM_ITER` and `BATCHSIZE` settings for fair comparison. Set `MODE` to one of the following and adjust accordingly:

| `MODE` | `NUM_ITER` | `BATCHSIZE` | Notes |
|---|---|---|---|
| `fedmosaic` | `94` | `4` | Set `USE_TASK_VECTOR=True` |
| `sft` / `fedavg` / `feddpa` | `100` | `4` | |
| `fedsim` / `takfl` | `75` | `4` | |
| `fedmkt` | `60` | `4` | |
| `ditto` / `perada` | `50` | `8` | |
| `feddat` | `43` | `8` | |

> ⚠️ `USE_TASK_VECTOR=True` should **only** be set for `fedmosaic`. Using them with other methods will cause errors.
> Methods that update local and global modules in local training steps require doubling the batch size to correctly update both modules with the same batch size to other methods when using `deepspeed`.
---

### Dataset-Specific Settings

Adjust learning rates and multimodal flags depending on the dataset used:

| Dataset | `LR` | `MM_PROJECTOR_LR` | `IS_MULTIMODAL` | `--lora_r` | `--lora_alpha` | `SCHED_NAME` | `sft NUM_ITER` |
|---|---|---|---|---|---|---|---|
| DRAKE / HFLB | `2e-5` | `5e-5` | `True` | `128` | `256` | `constant` | `100` |
| Fed-Scope / Fed-aya | `3e-4` | `5e-4` | `False` | `16` | `32` | `cosine` | `30` / `50` |
| Fed-LLM | `1e-4` | `5e-4` | `False` | `16` | `32` | `cosine` | `10`? |

---

## Evaluation

Default **Self** and **Others** evaluation is performed using:

```bash
bash eval_scripts/eval.sh
```

> ⚠️ Make sure `MODE` and `NOTE` in `eval_scripts/eval.sh` match the values used during training so that the correct checkpoints are loaded.

---

## Pretrained Checkpoints

For Co-LoRA, AB-aligned checkpoints are available on [Google Drive](https://drive.google.com/drive/folders/1VcqZMGiVuyS59AnuTJ_q5Ky6nVmDPWFZ?usp=drive_link).

To reproduce them from scratch:

```bash
bash train_abalign.sh
```

---

## 📄 Citation

If you find our work helpful in your research, please consider citing our paper. We'd really appreciate it! 🙏

```bibtex
@inproceedings{seo2026colora,
  title     = {Co-LoRA: Collaborative Model Personalization on Heterogeneous Multi-Modal Clients},
  author={Seo, Minhyuk and Kim, Taeheon and Lee, Hankook and Choi, Jonghyun and Tuytelaars, Tinne},
  booktitle = {The Fourteenth International Conference on Learning Representations (ICLR)},
  year      = {2026},
  url       = {https://openreview.net/forum?id=0g5Dk4Qfh0}
}
```

---

## 🙌 Acknowledgements

We sincerely thank the open-source community — this work builds on top of many excellent projects including [LLaVA](https://github.com/haotian-liu/LLaVA), [HuggingFace Transformers](https://github.com/huggingface/transformers), [PEFT](https://github.com/huggingface/peft), and [DeepSpeed](https://github.com/microsoft/DeepSpeed). We hope this codebase serves as a useful starting point for the federated and continual learning community, and we warmly welcome any questions, issues, or contributions. 😊
