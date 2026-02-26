RL env for reasoning on new math operators (e.g. `a ⊕ b = 2*a + 3*b + 1`).They are defined per-problem, and the model must evaluate expressions step by step, which impoves faithfulness of the reasoning (as the model cannont memoize the answers)

## Setup

```bash
uv sync
source .venv/bin/activate
export PYTHONPATH=.
```

Tests for env:

```bash
pytest tests/ -q
```

### Dataset Generation


```bash
python generate_datasets.py
```

Dataset on hf: [therem/novel-ops-reasoning](https://huggingface.co/datasets/therem/novel-ops-reasoning)

### Training


```bash
python train.py
```

### Eval


```bash
python eval.py
```

