# MST-CLIPIQA

This repository contains the PyTorch implementation of the paper:
**"Decoupling Semantics from Distortions: Multi-Scale Two-Stream Vision-Language Alignment for AI-Generated Image Quality Assessment"**

## Overview

MST-CLIPIQA is a multi-scale two-stream Vision-Language Model (VLM) framework that decouples global semantic understanding from local perceptual sensitivity using dual CLIP encoders with different patch sizes, fused via an information bottleneck-inspired gated mechanism.

### Core Contributions
- **Multi-Scale Two-Stream Feature Extraction (MSTFE)**: Uses dual frozen CLIP vision encoders (ViT-B/32 for coarse semantics and ViT-B/16 for fine details).
- **Gated Feature Fusion (GFF)**: Adaptively interpolates between coarse and fine features dimension-by-dimension.
- **Prediction Heads**: Supports both a Template-based Head (Variant A, no prompts) and a Prompt-Anchored Head (Variant B, with prompts).

## Requirements

- Python >= 3.8
- GPU with at least 12GB VRAM (to hold both ViT-B/32 and ViT-B/16 in memory)
- 16GB+ System RAM

Install the required dependencies:

```bash
pip install -r requirements.txt
```

*Note: Ensure your HuggingFace cache directory has sufficient space (~2GB) to download the `openai/clip-vit-base-patch32` and `openai/clip-vit-base-patch16` weights.*

## Dataset Preparation

The dataset should be structured with an image directory and a CSV file containing the annotations.
The CSV file must contain the following columns:
- `image_id`: The filename of the image (e.g., `image_0001.jpg`)
- `prompt`: The text prompt used to generate the image
- `mos`: The Mean Opinion Score (MOS) for quality assessment

## Usage

The repository provides a central CLI entry point `main.py` to run training and evaluation.

### Training

To train the model, use the `train` subcommand. You can choose between Variant A (Template-based, no prompts) and Variant B (Prompt-Anchored, with prompts).

**Train Variant A (Template-based, No Prompts):**
```bash
python main.py train --csv_file path/to/dataset.csv --img_dir path/to/images --variant A --batch_size 8 --epochs 20 --save_dir checkpoints
```

**Train Variant B (Prompt-Anchored, With Prompts):**
```bash
python main.py train --csv_file path/to/dataset.csv --img_dir path/to/images --variant B --batch_size 8 --epochs 20 --save_dir checkpoints
```

### Evaluation

To evaluate a trained model checkpoint on the test set, use the `eval` subcommand.

**Evaluate Variant A:**
```bash
python main.py eval --csv_file path/to/dataset.csv --img_dir path/to/images --variant A --checkpoint checkpoints/best_model.pth
```

**Evaluate Variant B:**
```bash
python main.py eval --csv_file path/to/dataset.csv --img_dir path/to/images --variant B --checkpoint checkpoints/best_model.pth
```

## Architecture Details

- **MSTFE**: Extracts `f_c` (coarse) and `f_f` (fine) features using frozen CLIP backbones.
- **GFF**: Fuses `f_c` and `f_f` into a unified representation `z` using a learned gating mechanism.
- **Heads**: 
  - **TemplateHead**: Computes cosine similarity between `z` and precomputed text embeddings of quality templates ("a photo with {q} quality").
  - **PromptHead**: Uses cross-attention between `z` and the text token embeddings of the generation prompt, with a zero-initialized residual connection.
- **Loss**: Optimizes a composite loss combining Mean Squared Error (MSE) and a Pairwise Margin Ranking Loss.

## Evaluation Metrics

The model is evaluated using:
- **SRCC** (Spearman Rank Correlation Coefficient)
- **PLCC** (Pearson Linear Correlation Coefficient)

SRCC on AGIQA-1K:
- Variant A (MST-CLIPIQA): ~0.8990
- Variant B (MST-CLIPIQA*): ~0.9091

## To do list



