# Artistic AI Experiments

## Setup

This project uses a standard `pyproject.toml` configuration. 
Any compatible Python package manager should work.
We recommend `uv` for a fast and reproducible setup:

```bash
uv sync
```

Our critic model uses the OpenAI API. Before running, set your API key:

```bash
export OPENAI_API_KEY="your_api_key_here"
```
See the [official guide](https://developers.openai.com/api/docs/quickstart#create-and-export-an-api-key) 
for details.

## Run

### Scalar Loop (XPO)

```bash
uv run python main.py \
  config config.toml \
  experiment.mode scalar \
  training.num_steps 10 \
  training.seed 42
```

### Textual Loop (Inference)

```bash
uv run python main.py \
  config config.toml \
  experiment.mode textual \
  training.num_steps 4 \
  training.seed 42
```

## Evaluation

```bash
uv run python human_eval/pairwise_cli.py \
  --textual result/textual \
  --scalar result/scalar \
  --output result/pairwise.csv
```

## Analysis

```bash
uv run python analysis/pairwise.py \
  result/pairwise.csv \
  --output result/analysis.csv
```
