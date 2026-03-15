"""
persona-bench Streamlit Dashboard.
Run with: streamlit run personabench/dashboard/app.py
Set RESULTS_FILE env var to point to your results JSON.
"""

import os
import json
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="persona-bench Dashboard",
    page_icon="🤖",
    layout="wide",
)

# ─── Load Results ────────────────────────────────────────────────────────────

def load_report(path: str) -> dict:
    with open(path) as f:
        return json.load(f)

results_file = os.environ.get("RESULTS_FILE", "results/latest.json")

st.title("🤖 persona-bench Dashboard")
st.caption("Stress-test your AI app with synthetic users before you ship.")

if not Path(results_file).exists():
    st.warning(
        f"No results found at `{results_file}`. "
        "Run `pb run` first, then reload this page."
    )
    st.code("pb run --app-url http://localhost:8000/chat --description 'My AI app' --personas 10")
    st.stop()

report = load_report(results_file)
results = report.get("results", [])

# ─── Summary Cards ───────────────────────────────────────────────────────────

def overall_score(results):
    if not results:
        return 0
    return round(sum(
        (r["scores"]["consistency"] + r["scores"]["hallucination"] +
         r["scores"]["tone_calibration"] + r["scores"]["graceful_failure"] +
         r["scores"]["refusal_appropriateness"]) / 5
        for r in results
    ) / len(results), 2)

def hallucination_rate(results):
    if not results:
        return 0.0
    flagged = sum(1 for r in results if "hallucination" in r.get("failure_modes", []))
    return flagged / len(results)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Overall Score", f"{overall_score(results)}/10")
col2.metric("Personas Tested", len(results))
col3.metric("Hallucination Rate", f"{hallucination_rate(results):.1%}")
col4.metric("App URL", report.get("app_url", "N/A"))

st.divider()

# ─── Failure Heatmap ─────────────────────────────────────────────────────────

st.subheader("🔥 Failure Heatmap: Archetype × Failure Mode")

from collections import defaultdict
failure_modes_all = [
    "off_topic_deflection", "contradictory_answer", "overly_verbose",
    "ignored_user_intent", "hallucination", "tone_mismatch",
    "over_refusal", "loop"
]
archetypes = sorted(set(r["persona"]["archetype"] for r in results))
matrix = defaultdict(lambda: defaultdict(int))
for r in results:
    arch = r["persona"]["archetype"]
    for mode in r.get("failure_modes", []):
        if mode != "none":
            matrix[arch][mode] += 1

heatmap_data = []
for arch in archetypes:
    row = [matrix[arch].get(mode, 0) for mode in failure_modes_all]
    heatmap_data.append(row)

fig_heatmap = px.imshow(
    heatmap_data,
    x=failure_modes_all,
    y=archetypes,
    color_continuous_scale="Reds",
    title="Failure Frequency by Archetype and Mode",
    aspect="auto",
)
fig_heatmap.update_xaxes(tickangle=30)
st.plotly_chart(fig_heatmap, use_container_width=True)

# ─── Radar Chart ─────────────────────────────────────────────────────────────

st.subheader("📊 Average Score Radar")

dims = ["consistency", "hallucination", "tone_calibration", "graceful_failure", "refusal_appropriateness"]
avg_scores = [
    round(sum(r["scores"][d] for r in results) / len(results), 2) if results else 0
    for d in dims
]

fig_radar = go.Figure(data=go.Scatterpolar(
    r=avg_scores + [avg_scores[0]],
    theta=dims + [dims[0]],
    fill="toself",
    line_color="royalblue",
))
fig_radar.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
    showlegend=False,
    title="App Quality Radar (avg across all personas)",
)
st.plotly_chart(fig_radar, use_container_width=True)

# ─── Per-Persona Table ───────────────────────────────────────────────────────

st.subheader("🎭 Per-Persona Results")

rows = []
for r in results:
    s = r["scores"]
    overall = round((s["consistency"] + s["hallucination"] + s["tone_calibration"] +
                     s["graceful_failure"] + s["refusal_appropriateness"]) / 5, 2)
    rows.append({
        "Name": r["persona"]["name"],
        "Archetype": r["persona"]["archetype"],
        "Overall": overall,
        "Consistency": s["consistency"],
        "Hallucination": s["hallucination"],
        "Tone": s["tone_calibration"],
        "Graceful Failure": s["graceful_failure"],
        "Refusal": s["refusal_appropriateness"],
        "Failure Modes": ", ".join(r.get("failure_modes", [])),
    })

df = pd.DataFrame(rows)
st.dataframe(df, use_container_width=True)

# ─── Conversation Drilldown ──────────────────────────────────────────────────

st.subheader("💬 Conversation Explorer")

persona_names = [r["persona"]["name"] for r in results]
selected = st.selectbox("Select a persona to inspect:", persona_names)
selected_result = next(r for r in results if r["persona"]["name"] == selected)

st.write(f"**Archetype:** {selected_result['persona']['archetype']}")
st.write(f"**Goal:** {selected_result['persona']['goal']}")
st.write(f"**Judge Notes:** {selected_result.get('judge_notes', 'N/A')}")

for turn in selected_result.get("turns", []):
    with st.expander(f"Turn {turn['turn_number']} — {turn.get('latency_ms', 0):.0f}ms"):
        st.markdown(f"**User:** {turn['user_message']}")
        st.markdown(f"**Assistant:** {turn['assistant_response']}")

# ─── Hallucination Table ─────────────────────────────────────────────────────

hallucinations = [
    {"Persona": r["persona"]["name"], "Example": ex}
    for r in results
    for ex in r.get("hallucination_examples", [])
]
if hallucinations:
    st.subheader("🚨 Hallucination Examples")
    st.dataframe(pd.DataFrame(hallucinations), use_container_width=True)
