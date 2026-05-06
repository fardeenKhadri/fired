# MNIST CNN Gradient-Based Pruning

This project implements **gradient-based weight pruning** in a Convolutional Neural Network (CNN) trained on the MNIST dataset using PyTorch. 

The primary goal is to experimentally evaluate how removing low-importance weights affects model performance and to analyze the trade-off between **model sparsity and accuracy**.

## Core Hypothesis

Weights with low importance (defined as `|weight × gradient|`) contribute less to the model’s performance and can be removed with minimal accuracy loss after retraining.

## Project Structure

- `pyproject.toml` - uv dependency and project configuration
- `model.py` - Contains the Simple CNN architecture
- `train.py` - Core training and retraining loops
- `prune.py` - Implements the iterative, layer-wise gradient pruning logic and mask enforcement
- `utils.py` - Dataset downloading, formatting, and model evaluation
- `main.py` - The execution entry point

## Getting Started

### Prerequisites

This project relies on [uv](https://github.com/astral-sh/uv) for lightning-fast Python dependency management. Make sure you have `uv` installed.

### Installation & Setup

1. **Sync the environment dependencies** (this will automatically create a `.venv` and install all required packages like `torch` and `matplotlib`):
   ```bash
   uv sync
   ```

2. *(Optional)* **Activate the virtual environment**:
   ```bash
   source .venv/bin/activate
   ```

### Running the Experiment

To execute the pipeline (which automatically handles initial training, iterative pruning, retraining, and evaluation), run:

```bash
uv run python main.py
```

*Note: The MNIST dataset will be automatically downloaded to a local `data/` directory upon the first run.*

### Outputs

- Detailed console logs tracking the iteration number, cumulative sparsity, training loss, and test accuracy.
- `sparsity_vs_accuracy.png`: A plot automatically saved to the root directory, visualizing the trade-off between the percentage of weights pruned and the resulting test accuracy.
