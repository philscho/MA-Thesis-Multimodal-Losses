# An Exploration of Loss Functions in Multimodal Models and Their Impact on Downstream Performance

**Master's Thesis by Philipp Scholl · M.Sc. Informatik (Computer Science) · Grade: 1.6 · Publication date: 28.07.2025**

[[Thesis PDF]](./Master_thesis-Philipp_Scholl-Multimodal_Losses.pdf) [[Thesis presentation slides PDF]](./master_thesis_presentation.pdf)

A systematic ablation study of four loss functions — **CLIP**, **ITM**, **SimCLR**, and **MLM** — in a dual-stream Vision-Language model trained from scratch.

Each loss function defines how the model learns from data: **CLIP** aligns image and text embeddings contrastively, **ITM** classifies whether image-text pairs match, **SimCLR** applies contrastive learning within a single modality, and **MLM** predicts masked tokens in text sequences.

The study measures how each loss function and their combinations affect zero-shot classification, image-text retrieval, and linear probing across 16 benchmark datasets.

---

## Architecture

<img src="https://github.com/user-attachments/assets/3b3948cb-517c-403b-aa89-61623c2dc19a" width="50%">

    
**Encoders:** ViT-Base-Patch16-224 + BERT-Base-Uncased (trained from scratch)  
**Training data:** ~3.95M image-caption pairs (CC3M + COCO Captions + Visual Genome)  
**Configurations:** 12 loss combinations × 7 dataset fractions (5%–100%) ⇒ 72 experiments  (not all fractions were evaluated for each combination)  
**Tracking:** Weights & Biases

---

## Key Findings

| Finding | Detail |
|---------|--------|
| **CLIP dominates** | CLIP loss leads to better visual representations than SimCLR and better multimodal representations than ITM |
| **MLM is most beneficial addition** | CLIP+MLM consistently outperforms CLIP alone |
| **SimCLR hurts** | SimCLR degrades CLIP performance due to projection layer interference |
| **ITM is weak alone** | ITM without cross-attention cannot anchor vision-language alignment |

---

## Results

### Linear Probe (Top-1 Accuracy)

| Modell | Avg. (All Datasets) |
|--------|---------------|
| **CLIP+MLM** | **67.83%** |
| CLIP | 67.05% |
| CLIP+ITM | 66.96% |
| CLIP+ITM+MLM | 66.60% |
| CLIP+SimCLR+ITM | 64.60% |
| CLIP+ITM+SimCLR+MLM | 65.95% |
| CLIP+SimCLR+MLM | 65.38% |
| CLIP+SimCLR | 64.59% |
| SimCLR+ITM+MLM | 52.81% |
| SimCLR+ITM | 43.01% |
| SimCLR | 42.71% |
| ITM+MLM | 32.09% |

### Zero-Shot Classification (Top-1 Accuracy)

| Model | Avg. (All Datasets) |
|--------|--------------|
| **CLIP+MLM+ITM** | **26.91%** |
| CLIP+MLM | 26.62% |
| CLIP | 26.37% |
| CLIP+SimCLR+MLM | 26.02% |
| CLIP+SimCLR+MLM+ITM | 25.26% |
| CLIP+ITM | 25.22% |
| CLIP+SimCLR+ITM | 23.98% |
| CLIP+SimCLR | 23.57% |

### Image-Text Retrieval (Recall@1)

| Model | Image Retrieval (Avg.) | Text Retrieval (Avg.) |
|--------|----------------------|----------------------|
| **CLIP+MLM** | **32.05%** | 47.10% |
| **CLIP+SimCLR+MLM+ITM** | 31.22% | **47.19%** |
| CLIP+SimCLR+MLM | 29.76% | 44.78% |
| CLIP+MLM+ITM | 28.70% | 42.39% |
| CLIP | 27.68% | 42.26% |
| CLIP+ITM | 27.31% | 41.29% |
| CLIP+SimCLR | 26.63% | 41.05% |
| CLIP+SimCLR+ITM | 26.37% | 39.91% |

---

## Project Structure

```
.
├── src/
│   ├── model/
│   │   ├── model_module.py       # LitMML: PyTorch Lightning training module
│   │   └── utils.py              # Model and processor initialization
│   ├── data/
│   │   ├── data_module.py        # MyDataModule: multi-dataset data pipeline
│   │   └── datasets/             # Dataset wrappers (COCO, CC3M, ImageNet, ...)
│   ├── callbacks/
│   │   ├── linear_probe.py       # Linear probing evaluation callback
│   │   ├── zero_shot.py          # Zero-shot classification callback
│   │   └── utils.py              # Callback instantiation helpers
│   ├── utils/
│   │   ├── loss_functions.py     # NTXentLoss (SimCLR), CLIP loss, cosine similarity
│   │   ├── optimizer_and_scheduler.py
│   │   └── zero_shot_func.py     # Zero-shot classifier construction and evaluation
│   └── analysis/
│       ├── cka.py                # Centered Kernel Alignment (NumPy + CUDA)
│       ├── representations.py    # Feature extraction and layer-wise analysis
│       └── rsa.py                # Representational Similarity Analysis
├── scripts/
│   ├── train.py                  # Main training entry point (Hydra + Lightning)
│   ├── evaluate.py               # Zero-shot + linear probe evaluation pipeline
│   └── analyze_representations.py # CKA / RSA analysis
├── configs/                      # Hydra configuration tree
│   ├── model/, data/, loss/      # Component configs
│   ├── optimizer/, scheduler/    # Training configs
│   ├── callbacks/, logger/       # Evaluation and logging configs
│   └── checkpoints/              # Experiment checkpoint references
├── results/                      # Evaluation outputs (CSV)
│   ├── model_scores_zero-shot.csv
│   ├── model_scores_linear_probe.csv
│   └── model_scores_retrieval.csv
└── notebooks/
    └── analysis/                 # ITM head inspection, CKA visualizations
```

---

## Setup

**Requirements:** Python 3.10+, CUDA 11.7+

```bash
git clone https://github.com/philscho/MA-Thesis-Multimodal-Losses
cd MA-Thesis-Multimodal-Losses
pip install -r requirements.txt
```

---

### Data

Download and point configs to:
- [COCO Captions](https://cocodataset.org/) — train2017 + annotations
- [Conceptual Captions 3M](https://ai.google.com/research/ConceptualCaptions/)
- [Visual Genome](https://homes.cs.washington.edu/~ranjay/visualgenome/index.html) — region descriptions

Update `configs/paths/` with your local data paths.

---

## Usage

### Training

```bash
# Train CLIP + MLM on full dataset
python scripts/train.py loss=clip_mlm data=full_dataset

# Ablation: CLIP only, 20% of data
python scripts/train.py loss=clip data.fraction=0.2

# All loss combinations (requires cluster)
bash scripts/run_scripts.sh
```

### Evaluation

```bash
# Zero-shot classification + linear probing
python scripts/evaluate.py checkpoints=full_dataset_models

# Retrieval evaluation (uses CLIP-Benchmark)
python scripts/evaluate.py +custom_run=retrieval_flickr30k
```

### Representation Analysis

```bash
# Centered Kernel Alignment between model configurations
python scripts/analyze_representations.py
```

---

## Technical Stack

| Component | Library |
|-----------|---------|
| Vision encoder | ViT-Base-Patch16-224 via HuggingFace Transformers |
| Text encoder | BERT-Base-Uncased via HuggingFace Transformers |
| Training framework | PyTorch Lightning 2.2 |
| Configuration | Hydra + OmegaConf |
| Experiment tracking | Weights & Biases |
| Evaluation metrics | torchmetrics, scikit-learn |
| Representation analysis | Custom CKA (CUDA-accelerated) |

---

## Citation

```bibtex
@mastersthesis{scholl2025multimodal,
  title   = {An Exploration of Loss Functions in Multimodal Models
             and Their Impact on Downstream Performance},
  author  = {Philipp Scholl},
  year    = {2025},
  school  = {Goethe University Frankfurt},
  note    = {Grade: 1.6}
}
```
