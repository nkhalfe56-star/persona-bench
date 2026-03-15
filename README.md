# 🤖 persona-bench

**Stress-test your AI app with synthetic users before you ship.**

persona-bench automatically generates 50+ diverse user personas, simulates realistic multi-turn conversations with your app, and scores every interaction across 5 dimensions — giving you a detailed failure report before your users find the bugs.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python) ![License](https://img.shields.io/badge/License-MIT-green) ![OpenAI](https://img.shields.io/badge/Powered%20by-OpenAI-black?logo=openai) ![FastAPI](https://img.shields.io/badge/Framework-FastAPI-teal?logo=fastapi)

---

## 🚀 The Problem

Every AI startup ships their chatbot and *then* discovers it fails for:
- Non-native English speakers
- Angry/frustrated users
- Adversarial testers who try to break it
- Edge cases no one thought to test

persona-bench catches these failures **before** you ship, automatically.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🎭 **50+ Synthetic Personas** | Auto-generated: angry, confused, elderly, adversarial, power user, non-native speaker, and more |
| 💬 **Multi-turn Simulation** | Each persona runs a full realistic conversation with your app, not just single prompts |
| 📊 **5-Dimension Scoring** | Consistency, Hallucination, Tone Calibration, Graceful Failure, Refusal Appropriateness |
| 🔥 **Failure Heatmap** | Visual heatmap: which persona archetypes expose which failure modes |
| ⚡ **Parallel Execution** | All personas run simultaneously via ThreadPoolExecutor — 50 personas in ~2 minutes |
| 🖥️ **Streamlit Dashboard** | Interactive dashboard with persona drilldown, radar charts, failing conversation explorer |
| 🔌 **Works With Anything** | Points at any REST API endpoint — OpenAI, LangChain, FastAPI, Flask, or your own backend |

---

## 🏗️ Architecture

```
persona-bench/
├── personabench/
│   ├── core.py                  # Main PersonaBench class (SDK entry point)
│   ├── schemas.py               # Pydantic models: Persona, Turn, ConversationResult, Report
│   ├── generator/
│   │   ├── persona_factory.py   # GPT-4o generates diverse realistic personas
│   │   └── conversation.py     # Simulates multi-turn conversations per persona
│   ├── runner/
│   │   ├── attack_runner.py    # Parallel execution engine (ThreadPoolExecutor)
│   │   └── scorer.py           # LLM-as-judge: scores each conversation
│   ├── scoring/
│   │   ├── consistency.py      # Are answers consistent across the conversation?
│   │   ├── hallucination.py    # Does the app invent facts?
│   │   ├── tone.py             # Does tone match what this persona needs?
│   │   └── failure_classifier.py # Classifies failure modes
│   ├── analysis/
│   │   ├── aggregator.py       # Builds heatmap data from all results
│   │   └── reporter.py        # Outputs JSON + Markdown report
│   ├── dashboard/
│   │   └── app.py             # Streamlit dashboard
│   └── cli.py                 # CLI: `pb run` and `pb dashboard`
├── tests/
│   └── test_aggregator.py
├── examples/
│   ├── mock_app.py             # Sample FastAPI app to test against
│   └── bench.yaml             # Sample config
├── .github/workflows/
│   └── test.yml               # CI/CD
├── requirements.txt
├── setup.py
└── .env.example
```

---

## ⚡ Quickstart

### 1. Install

```bash
git clone https://github.com/nkhalfe56-star/persona-bench.git
cd persona-bench
pip install -e .
cp .env.example .env
# Add your OPENAI_API_KEY to .env
```

### 2. Start your app (or use the mock)

```bash
uvicorn examples.mock_app:app --reload --port 8000
```

### 3. Run the benchmark

```bash
pb run \
  --app-url http://localhost:8000/chat \
  --description "A customer support chatbot for a SaaS billing tool" \
  --personas 20 \
  --turns 5
```

### 4. View the dashboard

```bash
pb dashboard
```

---

## 🐍 Python SDK Usage

```python
from personabench import PersonaBench

pb = PersonaBench(
    app_url="http://localhost:8000/chat",
    app_description="A customer support chatbot for a SaaS billing tool",
    openai_api_key="sk-..."  # or set OPENAI_API_KEY env var
)

report = pb.run(n_personas=20, turns_per_persona=5)

print(f"Overall Score: {report.overall_score}/10")
print(f"Hallucination Rate: {report.hallucination_rate:.1%}")
print(f"Top Failure Mode: {report.top_failure_mode}")

report.save("results/my_app_report.json")
```

---

## 📊 Scoring Dimensions

| Dimension | What It Checks | Scored By |
|---|---|---|
| **Consistency** | Same question = same answer across turns | GPT-4o-mini judge |
| **Hallucination** | Fabricated facts, invented stats, wrong claims | GPT-4o judge |
| **Tone Calibration** | Does tone match what this persona archetype needs? | GPT-4o-mini judge |
| **Graceful Failure** | Does the app fail gracefully when it does not know? | Rule-engine + LLM |
| **Refusal Appropriateness** | Does it refuse the right things and not over-refuse? | GPT-4o judge |

---

## 🎭 Persona Archetypes

| Archetype | Behavior |
|---|---|
| `angry` | Frustrated, aggressive tone, short fuse |
| `confused` | Does not understand instructions, asks vague questions |
| `adversarial` | Tries to jailbreak, extract system prompt, break the app |
| `elderly` | Formal, slow, needs very clear step-by-step answers |
| `non_native` | Non-fluent English, grammatical errors, unusual phrasing |
| `power_user` | Technical, wants detailed answers, impatient with vagueness |
| `impatient` | Very short messages, expects instant, concise answers |
| `verbose` | Over-explains, asks multiple questions at once |

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **OpenAI GPT-4o** — persona generation, simulation, LLM-as-judge scoring
- **FastAPI** — mock app example
- **Streamlit** — interactive dashboard
- **Pydantic v2** — strict data validation
- **ThreadPoolExecutor** — parallel persona simulation
- **Plotly** — heatmap and radar charts

---

## 🤝 Contributing

PRs welcome! Open an issue first to discuss what you would like to change.

---

## 📄 License

MIT — free to use, modify, and distribute.

---

*Built by [nkhalfe56-star](https://github.com/nkhalfe56-star)*
