# Contributing

Thank you for your interest in contributing to this study.

## Running Code Locally
We use `uv` for environment management to ensure strict reproducibility.
1. Clone the repo.
2. Run `uv venv && source .venv/bin/activate`.
3. Run `uv pip install -e ".[dev]"`.

## Adding a New Pruning Method
If you wish to benchmark a new criterion (e.g., Hessian-based or Taylor expansion):
1. Navigate to `prune.py`.
2. Implement your logic in a new function that accepts a PyTorch model and a sparsity percentage.
3. Ensure the function returns a binary mask of the same dimensions as the model weights.
4. Add your method identifier to the experiment configurations in `experiments/config.py`.

## Code Style
This repository strictly enforces code styling to maintain readability:
- Use `black` for formatting.
- Use `ruff` for linting.
- Ensure any new code passes `ruff check .` and `black --check .` before submitting a Pull Request.
