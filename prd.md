# Product Requirements Document (PRD)

## 1. Overview

This project aims to experimentally evaluate **gradient-based weight pruning** in a Convolutional Neural Network (CNN) trained on MNIST.

The goal is to measure how removing low-importance weights affects model performance and to analyze the trade-off between **model sparsity and accuracy**.

---

## 2. Objective

- Implement a CNN trained on MNIST
- Apply **iterative pruning using gradient-based importance**
- Measure:
  - Accuracy degradation
  - Loss behavior
  - Sparsity vs performance trade-off

---

## 3. Core Hypothesis

Weights with low importance (defined as `|weight × gradient|`) contribute less to the model’s performance and can be removed with minimal accuracy loss after retraining.

---

## 4. Scope

### Included

- CNN model training
- Gradient-based importance calculation
- Layer-wise weight pruning
- Iterative prune → retrain loop
- Performance tracking

### Excluded

- Large-scale datasets (CIFAR-10, ImageNet)
- GPU optimization
- Advanced pruning methods (structured pruning, MoE)

---

## 5. Technical Stack

- Python 3.x
- PyTorch
- torchvision
- matplotlib (for visualization)

---

## 6. Dataset

- MNIST
  - 60,000 training samples
  - 10,000 test samples
  - Grayscale images (28×28)

---

## 7. Model Architecture

Simple CNN:

- Conv2d (1 → 16, kernel=3, padding=1)

- ReLU

- MaxPool (2x2)

- Conv2d (16 → 32, kernel=3, padding=1)

- ReLU

- MaxPool (2x2)

- Flatten

- Linear (32×7×7 → 128)

- ReLU

- Linear (128 → 10)

---

## 8. Training Configuration

- Optimizer: Adam
- Learning rate: 1e-3
- Loss: CrossEntropyLoss
- Batch size: 64
- Initial training epochs: 5

---

## 9. Pruning Strategy

### Importance Metric

For each weight:

importance = |weight × gradient|

---

### Pruning Method

- Layer-wise pruning (each layer independently)
- Remove lowest importance weights per layer

---

### Iterative Pruning Setup

- Total iterations: 4
- Pruning per iteration: 10%
- Retraining after each pruning step: 2–3 epochs

---

## 10. Pruning Pipeline

1. Train baseline model
2. For each iteration:
   - Compute gradients (using 1–3 batches)
   - Calculate importance scores
   - Rank weights per layer
   - Prune lowest 10%
   - Retrain model
   - Evaluate performance

---

## 11. Evaluation Metrics

- Test Accuracy
- Training Loss
- Sparsity Level (% weights pruned)

---

## 12. Logging Requirements

Track after each iteration:

- Iteration number
- % weights pruned (cumulative)
- Test accuracy
- Loss

---

## 13. Output

### Required Outputs

- Console logs of training + pruning
- Final accuracy comparison
- Sparsity vs accuracy data

### Optional (Recommended)

- Plot:
  - X-axis: % weights pruned
  - Y-axis: Accuracy

---

## 14. Stretch Goals

- Compare with random pruning baseline
- Increase gradient estimation accuracy (multiple batches)
- Experiment with different pruning ratios

---

## 15. Success Criteria

- Model maintains reasonable accuracy (>90%) after moderate pruning (~30–40%)
- Clear relationship observed between sparsity and performance
- Results are reproducible and explainable

---

## 16. Risks

- Over-pruning → model collapse
- Poor gradient estimation → bad pruning decisions
- Insufficient retraining → degraded performance

---

## 17. Non-Goals

- Achieving state-of-the-art results
- Designing novel pruning algorithms
- Training large-scale models

---

## 18. Key Insight Target

Demonstrate that:

> Neural networks contain redundant parameters, and careful pruning can compress the model while preserving performance.
