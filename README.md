# An Exploration of Loss Functions in Multimodal Models and Their Impact on Downstream Performance

**Master's Thesis by Philipp Scholl · M.Sc. Informatik (Computer Science) · Grade: 1.6 · Publication date: 28.07.2025**

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
| **CLIP dominates** | CLIP loss leads to best results on all downstream tasks |
| **MLM adds value** | CLIP+MLM consistently outperforms CLIP alone (+1–2 pp on ImageNet zero-shot) |
| **SimCLR hurts** | SimCLR degrades CLIP performance due to projection layer interference |
| **ITM is weak alone** | ITM without cross-attention cannot anchor vision-language alignment |
| **Best combo** | CLIP+MLM achieves the best retrieval (R@1: 35.8% on Flickr30k) |

---

## Results

### Zero-Shot Classification — Top-1 Accuracy (full dataset)

| Loss Configuration | Caltech101 | CIFAR-10 | CIFAR-100 | ImageNet | Food101 |
|---|:-:|:-:|:-:|:-:|:-:|
| **CLIP + MLM** | **49.7%** | 49.9% | **22.1%** | **16.4%** | **15.6%** |
| CLIP + ITM + MLM | 47.8% | **59.0%** | **22.7%** | 16.2% | 13.8% |
| CLIP + ITM | 47.8% | 50.7% | 20.7% | 14.9% | 14.4% |
| **CLIP (baseline)** | 46.8% | 55.8% | 20.8% | 15.4% | 15.3% |
| CLIP + SimCLR | 46.4% | 48.8% | 18.7% | 14.7% | 15.1% |
| CLIP + SimCLR + MLM | 44.7% | 50.4% | 22.1% | 15.6% | 14.9% |
| SimCLR (no CLIP) | 1.5% | 11.3% | 1.3% | 0.1% | 1.2% |

### Image-Text Retrieval — Flickr30k (full dataset + augmentation)

| Loss Configuration | Image R@1 | Text R@1 | Image R@5 | Text R@5 |
|---|:-:|:-:|:-:|:-:|
| **CLIP + MLM** | **35.8%** | **51.1%** | 57.0% | 72.1% |
| CLIP + SimCLR + ITM + MLM | 34.9% | 51.6% | — | — |
| CLIP + ITM + MLM | 32.2% | 46.8% | — | — |
| **CLIP (baseline)** | 30.2% | 45.6% | 53.4% | 67.2% |
| CLIP + SimCLR | 28.7% | 44.6% | — | — |
| SimCLR (no CLIP) | 0.1% | 0.1% | — | — |

### Data Efficiency (CLIP baseline, Flickr30k IR@1)

| Training Data | 5% | 10% | 20% | 40% | 60% | 80% | 100% |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| IR@1 | 2.7% | 4.0% | 7.5% | 12.4% | 16.3% | 19.8% | 23.4% |

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
git clone https://github.com/philscho/multimodal-losses
cd multimodal-losses
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
