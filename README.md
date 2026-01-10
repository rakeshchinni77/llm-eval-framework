# LLM Evaluation Framework

A **production-grade, CI-safe, Dockerized LLM evaluation framework** for evaluating Large Language Model (LLM) outputs using **multiple metrics**, **quality gates**, and **reproducible pipelines**.

This framework is designed for:
- Retrieval-Augmented Generation (RAG)
- Question Answering (QA)
- LLM benchmarking
- Research & production evaluation workflows

---

## Problem This Project Solves

Evaluating LLM outputs is non-trivial because:

- Single metrics (BLEU / ROUGE) are insufficient
- Human evaluation is subjective and non-scalable
- Results are often not reproducible
- CI/CD pipelines rarely enforce LLM quality
- Docker-based evaluation is frequently unsafe or broken

### This framework provides:

- **Multi-metric evaluation** (reference, embedding, and judge-based)
- **Deterministic dataset ↔ prediction alignment**
- **CI-enforced quality gates**
- **Docker-safe & local execution parity**
- **Extensible metric architecture**
- **Visualization-ready outputs**

---

## Architecture Overview

```

┌──────────────────────────┐
│ Dataset (.jsonl) │
└─────────────┬────────────┘
│
┌─────────────▼────────────┐
│ EvaluationRunner │
│ - Aligns data safely │
│ - Parallel execution │
└─────────────┬────────────┘
│
┌─────────────▼─────────────┐
│ Metrics Registry │
│ BLEU / ROUGE / BERTScore │
│ Faithfulness / Relevancy │
│ LLM Judge │
└─────────────┬─────────────┘
│
┌─────────────▼──────────┐
│ Aggregator │
│ mean / std / min / max │
└─────────────┬──────────┘
│
┌─────────────▼──────────────┐
│ Results & Visualizations │
│ JSON + PNG outputs │
└────────────────────────────┘

```
---

##  Project Structure

```

llm-eval-framework/
├── benchmarks/ # Evaluation datasets (.jsonl)
├── examples/ # Example configs & model outputs
│ ├── config.yaml
│ ├── config_ci.yaml
│ └── model_outputs/
├── src/llm_eval/
│ ├── cli/ # CLI entrypoint
│ ├── config/ # Config loading & schema
│ ├── data/ # Dataset loaders & validators
│ ├── evaluation/ # Evaluation runner & aggregation
│ ├── metrics/ # Metric implementations
│ ├── reporting/ # Result formatting
│ └── visualization/ # Histograms & radar charts
├── tests/unit/ # Unit & integration tests
├── screenshots/ # Generated PNG charts
├── results/ # Evaluation JSON outputs
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── poetry.lock
└── README.md

```
---

## CLI Usage (Local)

### 1️.Install dependencies
```bash
poetry install
```

### 2.Run evaluation
```bash
poetry run llm-eval run \
  --config examples/config.yaml \
  --output-dir results/
```
## Expected output
```
llm-eval v0.1.0
Dataset: benchmarks/rag_benchmark.jsonl
Models: ['model_a']
Output directory: results
Evaluation completed successfully
```
---

## Generated Outputs
### JSON
results/aggregates.json
results/raw_scores.json
### Visualizations (PNG)
Metric histograms
Radar chart (model comparison)
Stored in:
```
screenshots/
```
## Docker Usage
Build & run (recommended)
```bash
docker-compose up --abort-on-container-exit
```
### This guarantees:

Clean environment

Reproducible results

Identical behavior to CI

---

## Testing & Coverage

Run tests locally:
```bash
poetry run pytest tests/ --cov=llm_eval
```

Coverage

≥ 80% enforced

Unit + integration tests included

CLI tested end-to-end

---
## CI/CD Pipeline

GitHub Actions workflow:
```bash
.github/workflows/evaluation.yml
```

### CI steps:

1.Install dependencies

2.Run tests (coverage enforced)

3.Run evaluation using config_ci.yaml

4.Upload results & screenshots

5.Fail pipeline on quality gate breach

CI failure = evaluation failure

---

## Quality Gates

Quality gates are enforced in CI using config_ci.yaml.

Example:
```
quality_gates:
  thresholds:
    bleu.mean: 0.01
```
Local runs may disable gates for development convenience.

---

## Visualization

Generate charts after evaluation:
```bash
poetry run python src/llm_eval/visualization/histograms.py
poetry run python src/llm_eval/visualization/radar.py
```

---
### Reproducibility Guarantees

1.Fixed dependency versions (poetry.lock)

2.Dockerized execution

3.CI-safe config

4.Deterministic dataset alignment

5.CPU-only torch (no GPU / meta tensor issues)

---

## License

MIT License — free for academic and commercial use.
