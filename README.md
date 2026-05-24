# solum-bench

Comprehensive benchmark for evaluating LLM quality on regenerative agriculture, soil science, and ecological farming topics.

Built as a companion to [Solum](https://github.com/wesdottoday/solum), a regenerative agriculture AI model. solum-bench answers the question: **"How good is this LLM as a regenerative agriculture advisor?"**

## Question Bank

540 questions across 8 categories, 15 YAML files, covering the full breadth of regenerative agriculture knowledge with global coverage.

| Category | Questions | Types | Scoring |
|----------|-----------|-------|---------|
| **Factual** (soil, climate, ecology, crops, regen practices) | 210 | MCQ, short answer | Exact match, keyword rubric |
| **Reasoning** (causal chains, nutrient cycling, comparative) | 90 | Open-ended | LLM-as-judge |
| **Practical** (farmer advice, multi-constraint scenarios) | 70 | Scenario | Composite (LLM judge + safety) |
| **Data Interpretation** (real soil profiles, climate data, species) | 40 | Data-grounded | Numeric grounding + LLM judge |
| **Regional Adaptation** (7 global regions x 5 base questions) | 35 | Regional | LLM judge with regional rubrics |
| **Safety-Critical** (banned pesticides, contamination, harmful practices) | 35 | Scenario | Hard-fail on violations |
| **Completeness / Systems Thinking** | 30 | Open-ended | LLM judge (holistic rubric) |
| **Grounding / Citation** (must reference provided data) | 30 | Data-grounded | Numeric grounding + keyword rubric |

### Global Coverage

Questions span all climate zones and continents, including tropical Oxisols (Brazil, Congo), Vertisols (India, Sudan), Andisols (Japan, Ecuador), podzols (Scandinavia), chernozems (Ukraine), Sahel dryland, Mekong Delta, Mediterranean, and more.

Seven test regions aligned with the Solum training corpus: **Ohio** (humid continental), **Helsinki** (boreal), **Tokyo** (humid subtropical), **Nairobi** (tropical highland), **Mato Grosso** (tropical savanna), **Canterbury NZ** (maritime temperate), **Rajasthan** (arid/semi-arid).

## 12 Evaluation Dimensions

| Dimension | What it measures |
|-----------|-----------------|
| Factual Accuracy | Correct facts, right answers |
| Reasoning Depth | Causal WHY explanations, multi-step logic |
| Specificity | Real numbers, species names, local data |
| Completeness | Covers all asked aspects |
| Regional Adaptation | Location-appropriate advice |
| Data Grounding | Uses provided context data, no hallucinated values |
| Practical Actionability | Implementable recommendations with timing and rates |
| Safety | No harmful practices recommended (hard-fail) |
| Systems Thinking | Considers whole ecosystem, not reductionist |
| Coherence | Well-structured, internally consistent |
| Citation Fidelity | Claims traceable to context or ground truth |
| Context Retention | Multi-turn: maintains context without contradiction |

## Installation

```bash
pip install -e .

# With API provider support:
pip install -e ".[anthropic,openai,google]"
```

Requires Python 3.11+.

## Quick Start

### Using Claude Code (no API key needed)

The fastest way to run solum-bench if you have [Claude Code](https://docs.anthropic.com/en/docs/claude-code) installed:

```bash
# Run full benchmark through Claude Code CLI
solum-bench run --config configs/claude_code.yaml --skip-llm-judge

# Quick test with 10 questions
solum-bench run --config configs/claude_code.yaml --max-questions 10 --skip-llm-judge

# Just factual MCQs (fastest, cheapest)
solum-bench run --config configs/claude_code.yaml --categories factual --skip-llm-judge
```

This uses `claude -p` with tools disabled, so responses are pure model knowledge. Works with enterprise billing.

### Using API Keys

```bash
# Set your API key
export ANTHROPIC_API_KEY=sk-...

# Run against Claude via API
solum-bench run --config configs/frontier_models.yaml --skip-llm-judge
```

### View Results

```bash
# Terminal report
solum-bench report --run results/<run_dir>/ --format terminal

# Markdown report
solum-bench report --run results/<run_dir>/ --format markdown --output report.md
```

## Evaluation Without API Keys

For organizations with enterprise billing that don't have direct API keys, solum-bench supports a CSV export workflow for evaluation via Claude Code:

```bash
# 1. Run the benchmark (generates responses, applies fast scorers)
solum-bench run --config configs/claude_code.yaml --skip-llm-judge

# 2. Export responses for judging
solum-bench export-for-judging --run results/<run_dir>/ --output to_judge.csv

# 3. Open to_judge.csv in Claude Code and ask it to score each response
#    The CSV includes the question, response, rubric, and dimensions to score

# 4. Import scores back
solum-bench import-scores --scores scored.csv --run results/<run_dir>/

# 5. Generate report
solum-bench report --run results/<run_dir>/
```

## CLI Reference

```
solum-bench run              Run benchmark against configured models
solum-bench score            Re-score existing results with different judges
solum-bench compare          Compare results across multiple runs
solum-bench report           Generate benchmark reports (terminal/markdown/json)
solum-bench validate         Validate question bank integrity
solum-bench list-questions   List questions with optional filtering
solum-bench export-for-judging  Export for external LLM evaluation
solum-bench import-scores    Import externally-produced judge scores
```

### Key Options

```bash
--config        Model config YAML (required for run)
--categories    Filter: factual, reasoning, practical, safety, etc.
--regions       Filter: ohio, helsinki, tokyo, nairobi, mato_grosso, canterbury, rajasthan
--difficulties  Filter: easy, medium, hard
--max-questions Limit number of questions (for quick tests)
--skip-llm-judge  Use only fast deterministic scorers
--resume        Resume an interrupted run from checkpoint
--seed          Random seed for reproducibility (default: 42)
```

## Model Providers

| Provider | Config Key | Requirements |
|----------|-----------|--------------|
| Claude Code CLI | `claude_code` | `claude` on PATH |
| Anthropic API | `anthropic` | `ANTHROPIC_API_KEY` |
| OpenAI / OpenRouter | `openai` | `OPENAI_API_KEY` |
| Google Gemini | `google` | `GOOGLE_API_KEY` |
| vLLM (local GPU) | `vllm_local` | NVIDIA GPU + vLLM |
| HuggingFace (local) | `hf_local` | GPU + transformers |

Configure models in YAML:

```yaml
models:
  - name: claude-sonnet-via-cc
    provider: claude_code
    model: sonnet
```

## Scoring Methods

| Method | Speed | Use Case |
|--------|-------|----------|
| `exact_match` | Instant | MCQ, numeric answers |
| `keyword_rubric` | Instant | Required keywords + rubric criteria |
| `numeric_grounding` | Instant | Verifies cited numbers match context data |
| `safety_check` | Instant | Banned terms, required warnings (hard-fail) |
| `llm_judge` | Slow (API call) | Open-ended quality evaluation |
| `composite` | Mixed | Combines multiple methods |

## Project Structure

```
solum-bench/
├── src/solum_bench/
│   ├── cli.py              # Click CLI
│   ├── config.py           # Constants and defaults
│   ├── providers/          # Model providers (6 backends)
│   ├── questions/          # Schema, loader, validator
│   ├── scoring/            # 6 scoring methods + aggregator
│   ├── runner/             # Benchmark engine + session management
│   └── results/            # Storage, comparison, reports, export
├── question_bank/v1/       # 540 questions in 15 YAML files
├── configs/                # Model configuration YAMLs
└── tests/
```

## License

AGPL-3.0-only
