# Masked by Retraining: An Empirical Study of Neural Network Pruning

**Project Description**: An experimental PyTorch study comparing gradient-based and random weight pruning on MNIST to reveal the masking effect of post-pruning retraining.

## Motivation

Model compression is essential for deploying neural networks to edge devices with strict memory and compute constraints. Weight pruning iteratively removes parameters, reducing the model's footprint. A critical area of research involves finding the optimal pruning criterion (deciding _which_ weights to drop). However, evaluating the actual quality of these criteria is difficult because the standard pipeline involves retraining the network after pruning, which can obscure the underlying structural damage caused by the pruning step itself.

## Core Hypothesis

We hypothesize that post-pruning retraining acts as an equalizer that masks the true efficacy of the underlying pruning strategy. By evaluating pruned models _without_ the retraining safety net, we expect the difference in quality between an informed selection criterion (gradient-based importance) and an uninformed one (random selection) to become starkly apparent.

## Experimental Results

We trained a baseline CNN on MNIST to **98.85%** accuracy and applied iterative layer-wise pruning (removing 10% of remaining weights per iteration). We compared **Gradient-based Pruning** (importance = $|weight \times gradient|$) against **Random Pruning**.

| Experiment               | Condition              | Sparsity (%) | Gradient Accuracy | Random Accuracy  | Performance Gap |
| :----------------------- | :--------------------- | :----------- | :---------------- | :--------------- | :-------------- |
| **1. Push to Collapse**  | With Retraining        | 90%          | ~98.8%            | ~98.8%           | ~0.0%           |
|                          | With Retraining        | 100%         | 10.0% (collapse)  | 10.0% (collapse) | 0.0%            |
| **2. No-Retrain**        | No Retraining          | 50%          | **97.5%**         | 55.0%            | **+42.5%**      |
| **3. Improved Gradient** | No Retrain, 30 Batches | 60%          | **96.2%**         | 38.4%            | **+57.8%**      |

### Key Insight: The Masking Effect

Our results demonstrate that retraining hides poor weight selection. When weights are dropped randomly, critical neural pathways are destroyed. However, if the network is allowed to retrain, the remaining weights adapt and recover the lost performance, making random pruning appear nearly as effective as targeted gradient-based pruning (Experiment 1).

When we remove the retraining step (Experiment 2), the true signal emerges: gradient-based pruning intelligently preserves critical pathways, maintaining 97.5% accuracy even when half the network is removed. In contrast, random pruning collapses to 55%. This 42% gap proves that the gradient criterion provides a measurably superior structural prior. Furthermore, by improving the gradient estimation through a 30-batch accumulation window (Experiment 3), the gap widens significantly at higher sparsities.

## Repository Structure

```text
.
├── CONTRIBUTING.md
├── LICENSE
├── README.md
├── pyproject.toml
├── experiments/
│   ├── __init__.py
│   └── config.py          # Experiment presets
├── results/
│   └── summary.json       # Captured experimental metrics
├── model.py               # CNN Architecture
├── prune.py               # Pruning logic (Gradient & Random)
├── train.py               # Training and evaluation loops
└── utils.py               # Data loading and helpers
```

## Installation

This project uses [`uv`](https://github.com/astral-sh/uv) for fast, deterministic dependency management. Do not use `pip` or `conda`.

```bash
# Clone the repository
git clone https://github.com/yourusername/pruning-study.git
cd pruning-study

# Create a virtual environment and install dependencies
uv venv
source .venv/bin/activate
uv pip install -e .
```

## Running the Experiments

You can replicate the specific experimental conditions using the predefined configurations:

```bash
# 1. Run the collapse threshold experiment (with retraining)
python train.py --config experiments.config.collapse_experiment

# 2. Run the pure ablation study (no retraining)
python train.py --config experiments.config.no_retrain_experiment

# 3. Run the improved gradient estimation study
python train.py --config experiments.config.improved_gradient_experiment
```

## Limitations

- **Dataset Complexity**: MNIST is a simple, highly redundant dataset. Findings here provide a foundational intuition but do not automatically scale.
- **Model Scale**: The CNN architecture used in this study is small. Over-parameterized models (e.g., modern ResNets or Transformers) exhibit different plasticity dynamics.
- **Generalizability**: The observed 42% performance gap may compress or expand when applied to different architectures or more complex datasets like CIFAR-10 or ImageNet.

## Conclusion

Retraining is a powerful mechanism for recovering performance after network degradation. However, our findings suggest that to honestly evaluate the quality of a pruning criterion, researchers must ablate the retraining phase. Under strict constraints without retraining, gradient-based magnitude pruning is vastly superior to random baseline approaches.
