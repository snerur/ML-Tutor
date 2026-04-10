"""
Page 5: Fairness Evaluation
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import utils.compat  # noqa: F401

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.fairness_utils import (
    compute_group_metrics,
    compute_fairness_metrics,
    get_fairness_interpretation,
)

st.set_page_config(page_title="Fairness Evaluation - ML Fairness Studio", layout="wide")

st.title("⚖️ Fairness Evaluation")

# ── Guard ─────────────────────────────────────────────────────────────────────
if st.session_state.get("model") is None:
    st.warning("⚠️ No trained model found. Please complete **🏋️ Model Training** first.")
    st.stop()
if st.session_state.get("X_test") is None:
    st.warning("⚠️ No test data found. Please complete **⚙️ Preprocessing** first.")
    st.stop()

model = st.session_state["model"]
X_test = st.session_state["X_test"]
y_test = np.asarray(st.session_state["y_test"])
protected_cols = st.session_state.get("protected_cols", [])
df = st.session_state.get("df")
X_test_raw = st.session_state.get("X_test_raw")

if not protected_cols:
    st.warning("⚠️ No protected attributes configured. Go to **📂 Data Upload** to set them.")
    st.stop()

y_pred = model.predict(X_test)
model_name = st.session_state.get("model_name", "Model")
st.markdown(f"**Model:** {model_name} | **Test samples:** {len(y_test):,}")

# ── Protected attribute selector ─────────────────────────────────────────────
st.markdown("---")
selected_attr = st.selectbox(
    "Evaluate fairness on protected attribute:",
    protected_cols,
    index=0,
)

# Get sensitive features for test set
sensitive_test = None
if X_test_raw is not None and selected_attr in X_test_raw.columns:
    sensitive_test = X_test_raw[selected_attr].reset_index(drop=True).values
elif st.session_state.get("sensitive_test") is not None and \
        st.session_state.get("sensitive_col_for_fairness") == selected_attr:
    sensitive_test = st.session_state["sensitive_test"]
elif df is not None and selected_attr in df.columns:
    # Reconstruct from original df (approximate - last n_test rows by index)
    st.warning(f"⚠️ Using approximation for `{selected_attr}` sensitive features.")
    n_test = len(y_test)
    sensitive_test = df[selected_attr].iloc[-n_test:].reset_index(drop=True).values

if sensitive_test is None:
    st.error(f"❌ Could not retrieve test-set values for `{selected_attr}`. "
             f"Make sure preprocessing was run with this attribute in the dataset.")
    st.stop()

# Ensure alignment
if len(sensitive_test) != len(y_test):
    st.error(f"Length mismatch: sensitive_test={len(sensitive_test)}, y_test={len(y_test)}")
    st.stop()

# ── Per-Group Metrics ─────────────────────────────────────────────────────────
st.markdown(f"## 📊 Per-Group Performance: `{selected_attr}`")
group_metrics_df = compute_group_metrics(y_test, y_pred, sensitive_test)
st.dataframe(
    group_metrics_df.style.format({
        "Accuracy": "{:.3f}",
        "Precision": "{:.3f}",
        "Recall": "{:.3f}",
        "F1 Score": "{:.3f}",
        "Selection Rate": "{:.3f}",
    }).background_gradient(subset=["Accuracy", "F1 Score"], cmap="RdYlGn"),
    use_container_width=True,
    hide_index=True,
)

# ── Visualizations ────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    fig_acc = px.bar(
        group_metrics_df,
        x="Group",
        y="Accuracy",
        color="Accuracy",
        color_continuous_scale="RdYlGn",
        title=f"Accuracy per {selected_attr} Group",
        text=group_metrics_df["Accuracy"].round(3).astype(str),
    )
    fig_acc.add_hline(
        y=group_metrics_df["Accuracy"].mean(),
        line_dash="dash",
        annotation_text="Mean",
        line_color="gray",
    )
    fig_acc.update_layout(height=380, showlegend=False, yaxis=dict(range=[0, 1]))
    st.plotly_chart(fig_acc, use_container_width=True)

with col2:
    fig_sr = px.bar(
        group_metrics_df,
        x="Group",
        y="Selection Rate",
        color="Selection Rate",
        color_continuous_scale="Blues",
        title=f"Selection Rate per {selected_attr} Group",
        text=group_metrics_df["Selection Rate"].round(3).astype(str),
    )
    fig_sr.add_hline(
        y=group_metrics_df["Selection Rate"].mean(),
        line_dash="dash",
        annotation_text="Mean",
        line_color="gray",
    )
    fig_sr.update_layout(height=380, showlegend=False, yaxis=dict(range=[0, 1]))
    st.plotly_chart(fig_sr, use_container_width=True)

# Selection Rate vs Accuracy scatter
fig_scatter = px.scatter(
    group_metrics_df,
    x="Selection Rate",
    y="Accuracy",
    text="Group",
    size="Count",
    color="Group",
    title=f"Selection Rate vs Accuracy by {selected_attr} Group",
    labels={"Selection Rate": "Selection Rate", "Accuracy": "Accuracy"},
)
fig_scatter.update_traces(textposition="top center")
fig_scatter.update_layout(height=420)
st.plotly_chart(fig_scatter, use_container_width=True)

# Multi-metric grouped bar
metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1 Score", "Selection Rate"]
fig_multi = go.Figure()
for metric in metrics_to_plot:
    fig_multi.add_trace(go.Bar(
        name=metric,
        x=group_metrics_df["Group"],
        y=group_metrics_df[metric],
    ))
fig_multi.update_layout(
    barmode="group",
    title=f"All Metrics by {selected_attr} Group",
    yaxis=dict(range=[0, 1]),
    height=420,
    xaxis_title=selected_attr,
    yaxis_title="Score",
)
st.plotly_chart(fig_multi, use_container_width=True)

# ── Fairness Metrics ──────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(f"## ⚖️ Fairness Metrics: `{selected_attr}`")

fairness_metrics = compute_fairness_metrics(y_test, y_pred, sensitive_test)
st.session_state["fairness_results"] = fairness_metrics
st.session_state["sensitive_col_for_fairness"] = selected_attr

METRIC_LABELS = {
    "demographic_parity_difference": "Demographic Parity Difference",
    "demographic_parity_ratio": "Demographic Parity Ratio",
    "equalized_odds_difference": "Equalized Odds Difference",
    "equal_opportunity_difference": "Equal Opportunity Difference",
    "disparate_impact": "Disparate Impact",
    "statistical_parity_difference": "Statistical Parity Difference",
}

METRIC_THRESHOLDS = {
    "demographic_parity_difference": {"good": "< 0.1 (absolute)", "bad": "≥ 0.1"},
    "demographic_parity_ratio": {"good": "≥ 0.8 (80% rule)", "bad": "< 0.8"},
    "equalized_odds_difference": {"good": "< 0.1 (absolute)", "bad": "≥ 0.1"},
    "equal_opportunity_difference": {"good": "< 0.1 (absolute)", "bad": "≥ 0.1"},
    "disparate_impact": {"good": "≥ 0.8 (80% rule)", "bad": "< 0.8"},
    "statistical_parity_difference": {"good": "< 0.1 (absolute)", "bad": "≥ 0.1"},
}

rows = []
for key, label in METRIC_LABELS.items():
    val = fairness_metrics.get(key)
    if val is None:
        continue
    interp, color = get_fairness_interpretation(key, val)
    threshold_info = METRIC_THRESHOLDS.get(key, {})
    rows.append({
        "Metric": label,
        "Value": round(val, 4),
        "Fair Threshold": threshold_info.get("good", ""),
        "Interpretation": interp,
        "Status": "✅ Fair" if color == "green" else "⚠️ Unfair",
    })

if rows:
    fairness_df = pd.DataFrame(rows)

    def color_status(val):
        if "✅" in val:
            return "background-color: #d4edda; color: #155724"
        elif "⚠️" in val:
            return "background-color: #fff3cd; color: #856404"
        return ""

    styled = fairness_df.style.applymap(color_status, subset=["Status"])
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # Fairness metrics radar chart
    radar_metrics = ["demographic_parity_difference", "equalized_odds_difference",
                     "equal_opportunity_difference"]
    radar_vals = []
    radar_labels = []
    for m in radar_metrics:
        v = fairness_metrics.get(m)
        if v is not None:
            radar_vals.append(abs(v))
            radar_labels.append(METRIC_LABELS.get(m, m))

    if len(radar_vals) >= 3:
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=radar_vals + [radar_vals[0]],
            theta=radar_labels + [radar_labels[0]],
            fill="toself",
            name="Unfairness (lower is better)",
            line_color="tomato",
            fillcolor="rgba(255, 99, 71, 0.3)",
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=[0.1] * (len(radar_labels) + 1),
            theta=radar_labels + [radar_labels[0]],
            fill="toself",
            name="Fair threshold (0.1)",
            line_color="green",
            line_dash="dash",
            fillcolor="rgba(0, 128, 0, 0.1)",
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, max(0.5, max(radar_vals) * 1.2)])),
            title="Fairness Radar (lower = fairer)",
            height=450,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

# ── Multi-attribute Comparison ────────────────────────────────────────────────
if len(protected_cols) > 1:
    st.markdown("---")
    st.markdown("## 🔄 Cross-Attribute Fairness Comparison")
    st.markdown("Compare fairness metrics across all protected attributes.")

    all_fairness = {}
    for attr in protected_cols:
        s_test = None
        if X_test_raw is not None and attr in X_test_raw.columns:
            s_test = X_test_raw[attr].reset_index(drop=True).values
        elif df is not None and attr in df.columns:
            n_test = len(y_test)
            s_test = df[attr].iloc[-n_test:].reset_index(drop=True).values

        if s_test is not None and len(s_test) == len(y_test):
            try:
                fm = compute_fairness_metrics(y_test, y_pred, s_test)
                all_fairness[attr] = fm
            except Exception:
                pass

    if all_fairness:
        compare_data = []
        for attr, fm in all_fairness.items():
            for key, label in METRIC_LABELS.items():
                v = fm.get(key)
                if v is not None:
                    compare_data.append({
                        "Attribute": attr,
                        "Metric": label,
                        "Value": v,
                        "|Value|": abs(v),
                    })

        if compare_data:
            compare_df = pd.DataFrame(compare_data)
            fig_compare = px.bar(
                compare_df,
                x="Attribute",
                y="|Value|",
                color="Metric",
                barmode="group",
                title="Fairness Metric Magnitudes Across Protected Attributes",
                labels={"|Value|": "Metric |Value| (lower = fairer)"},
            )
            fig_compare.update_layout(height=450)
            st.plotly_chart(fig_compare, use_container_width=True)

st.info("➡️ Next step: Navigate to **🧪 Model Testing** for full test set evaluation.")
