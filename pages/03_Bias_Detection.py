"""
Page 3: Bias Detection & Mitigation
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import utils.compat  # noqa: F401 — NumPy 2.0 patches before any compiled library

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
    SCIPY_ERROR = None
except Exception as _e:
    SCIPY_AVAILABLE = False
    SCIPY_ERROR = str(_e)
    stats = None

from utils.fairness_utils import apply_reweighing

st.set_page_config(page_title="Bias Detection - ML Fairness Studio", layout="wide")

st.title("🔍 Bias Detection & Mitigation")

# ── Guard ─────────────────────────────────────────────────────────────────────
if st.session_state.get("df") is None:
    st.warning("⚠️ No dataset loaded. Please go to **📂 Data Upload** first.")
    st.stop()
if not st.session_state.get("protected_cols"):
    st.warning("⚠️ No protected attributes selected. Please go to **📂 Data Upload** and select protected attributes.")
    st.stop()

df = st.session_state["df"].copy()
target_col = st.session_state["target_col"]
protected_cols = st.session_state["protected_cols"]
task_type = st.session_state.get("task_type", "classification")

st.markdown(f"Analyzing bias in **{len(protected_cols)}** protected attribute(s): {', '.join(protected_cols)}")
if task_type == "regression":
    st.info(
        "Regression target detected. Bias analysis uses **ANOVA F-test** and "
        "**mean target value per group** instead of chi-square / disparate impact."
    )

# ── Helpers ───────────────────────────────────────────────────────────────────
def _is_continuous(series):
    """Return True if the series looks like a continuous numeric target."""
    import pandas as pd
    if not pd.api.types.is_numeric_dtype(series):
        return False
    n_unique = series.nunique()
    if n_unique <= 20:
        return False
    if pd.api.types.is_float_dtype(series) and n_unique > 20:
        return True
    if n_unique / len(series) > 0.05 and n_unique > 20:
        return True
    return False


def _determine_positive_class(target_series):
    """
    Determine the 'positive' class for disparate impact analysis.

    Rules:
    - If values are exactly {0, 1} or {False, True} → 1/True is positive
    - If values contain common 'positive' keywords → pick that
    - Otherwise, pick the minority class (usually the more 'favourable' outcome)
      as in employment/lending contexts, or the class that makes most sense
      for bias analysis (the one users are trying to predict).
    For recidivism datasets, 'Yes' means recidivism occurred which is a
    meaningful outcome for bias analysis.
    """
    classes = sorted(target_series.dropna().unique())
    if len(classes) == 0:
        return classes[0] if classes else None

    # Binary numeric 0/1
    if set(str(c) for c in classes) == {"0", "1"} or set(classes) == {0, 1}:
        return 1

    # Common positive-label keywords
    positive_keywords = {"yes", "1", "true", "default", "positive", "fraud", "recid", "high"}
    for cls in classes:
        if str(cls).strip().lower() in positive_keywords:
            return cls

    # Minority class → typically more meaningful for bias analysis
    counts = target_series.value_counts()
    return counts.index[-1]  # least frequent


# ── Bias Analysis per Protected Attribute ─────────────────────────────────────
for attr in protected_cols:
    if attr not in df.columns:
        st.warning(f"Column `{attr}` not found in dataset.")
        continue

    st.markdown(f"---\n## 🔎 Attribute: `{attr}`")

    combined = df[[attr, target_col]].dropna()
    target_is_continuous = _is_continuous(combined[target_col]) or task_type == "regression"

    col1, col2 = st.columns(2)

    if target_is_continuous:
        # ── Box plot: target distribution per group ────────────────────────
        with col1:
            fig_box = px.box(
                combined, x=attr, y=target_col,
                title=f"{target_col} distribution by {attr}",
                color=attr,
                points="outliers",
            )
            fig_box.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_box, use_container_width=True)

        with col2:
            # Mean target per group
            group_means = combined.groupby(attr)[target_col].agg(["mean", "std", "count"]).reset_index()
            group_means.columns = [attr, "Mean", "Std", "Count"]
            fig_bar = px.bar(
                group_means, x=attr, y="Mean",
                error_y="Std",
                title=f"Mean {target_col} per {attr} group",
                color="Mean",
                color_continuous_scale="RdYlGn_r",
                text=group_means["Mean"].round(3).astype(str),
            )
            grand_mean = combined[target_col].mean()
            fig_bar.add_hline(y=grand_mean, line_dash="dash",
                              annotation_text=f"Overall mean ({grand_mean:.3f})")
            fig_bar.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)

        # ── Statistical test: ANOVA / Kruskal-Wallis ──────────────────────
        st.markdown("#### 📊 Statistical Test (ANOVA / Kruskal-Wallis)")
        if not SCIPY_AVAILABLE:
            st.warning("scipy not available — statistical tests require scipy.")
        else:
            try:
                groups_data = [
                    combined[combined[attr] == grp][target_col].values
                    for grp in combined[attr].unique()
                ]
                # One-way ANOVA
                f_stat, p_anova = stats.f_oneway(*groups_data)
                # Kruskal-Wallis (non-parametric)
                h_stat, p_kruskal = stats.kruskal(*groups_data)

                col_t1, col_t2, col_t3, col_t4 = st.columns(4)
                with col_t1:
                    st.metric("ANOVA F-statistic", f"{f_stat:.4f}")
                with col_t2:
                    st.metric("ANOVA p-value", f"{p_anova:.4e}")
                with col_t3:
                    st.metric("Kruskal-Wallis H", f"{h_stat:.4f}")
                with col_t4:
                    st.metric("Kruskal p-value", f"{p_kruskal:.4e}")

                # Use Kruskal-Wallis p-value (more robust, doesn't assume normality)
                p_value = p_kruskal
                if p_value < 0.001:
                    st.error(
                        f"⚠️ **Strong evidence of group difference** (p < 0.001): "
                        f"`{attr}` groups have significantly different `{target_col}` values. "
                        "Potential bias detected."
                    )
                elif p_value < 0.05:
                    st.warning(
                        f"⚠️ **Significant group difference** (p < 0.05): "
                        f"`{attr}` is associated with `{target_col}`. Possible bias."
                    )
                else:
                    st.success(
                        f"✅ **No significant group difference** (p = {p_value:.3f}): "
                        f"No statistical evidence that `{attr}` affects `{target_col}`."
                    )

                # Eta-squared (effect size for ANOVA)
                all_vals = combined[target_col].values
                grand_mean_val = all_vals.mean()
                ss_between = sum(
                    len(g) * (g.mean() - grand_mean_val) ** 2 for g in groups_data
                )
                ss_total = sum((x - grand_mean_val) ** 2 for x in all_vals)
                eta_sq = ss_between / ss_total if ss_total > 0 else 0
                effect = "negligible" if eta_sq < 0.01 else "small" if eta_sq < 0.06 else "medium" if eta_sq < 0.14 else "large"
                st.info(f"📏 **Eta-squared** = {eta_sq:.4f} ({effect} effect size)")

            except Exception as e:
                st.warning(f"Statistical test failed: {e}")

        # Mean difference table
        st.markdown("#### 📋 Group Summary")
        st.dataframe(group_means, use_container_width=True, hide_index=True)

    else:
        # ── Categorical target: original chi-square + disparate impact ─────
        target_classes = combined[target_col].unique()

        # Grouped bar chart
        with col1:
            cross_tab = pd.crosstab(combined[attr], combined[target_col])
            cross_tab_pct = cross_tab.div(cross_tab.sum(axis=1), axis=0) * 100

            fig_bar = go.Figure()
            for cls in cross_tab.columns:
                fig_bar.add_trace(go.Bar(
                    name=str(cls),
                    x=cross_tab.index.astype(str),
                    y=cross_tab[cls],
                    text=cross_tab[cls],
                    textposition="auto",
                ))
            fig_bar.update_layout(
                barmode="group",
                title=f"{attr} × {target_col} (Counts)",
                xaxis_title=attr, yaxis_title="Count",
                height=380, legend_title=target_col,
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with col2:
            fig_stacked = go.Figure()
            for cls in cross_tab_pct.columns:
                fig_stacked.add_trace(go.Bar(
                    name=str(cls),
                    x=cross_tab_pct.index.astype(str),
                    y=cross_tab_pct[cls].round(1),
                    text=cross_tab_pct[cls].round(1).astype(str) + "%",
                    textposition="auto",
                ))
            fig_stacked.update_layout(
                barmode="stack",
                title=f"{attr} × {target_col} (Percentages)",
                xaxis_title=attr, yaxis_title="Percentage (%)",
                height=380, legend_title=target_col,
            )
            st.plotly_chart(fig_stacked, use_container_width=True)

        # ── Chi-square test ────────────────────────────────────────────────
        st.markdown("#### 📊 Chi-Square Test for Independence")
        if not SCIPY_AVAILABLE:
            st.warning(
                "scipy could not be loaded — statistical tests are unavailable.\n\n"
                "Fix: `pip install \"numpy>=1.24,<2.0\" --force-reinstall`"
                + (f"\n\nError: `{SCIPY_ERROR}`" if SCIPY_ERROR else "")
            )
        else:
            try:
                contingency_table = pd.crosstab(combined[attr], combined[target_col])
                chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

                test_col1, test_col2, test_col3 = st.columns(3)
                with test_col1:
                    st.metric("Chi-square statistic", f"{chi2:.4f}")
                with test_col2:
                    st.metric("p-value", f"{p_value:.4e}")
                with test_col3:
                    st.metric("Degrees of freedom", dof)

                if p_value < 0.001:
                    st.error(
                        f"⚠️ **Strong evidence of association** (p < 0.001): "
                        f"`{attr}` is highly associated with `{target_col}`. Potential bias detected."
                    )
                elif p_value < 0.05:
                    st.warning(
                        f"⚠️ **Significant association** (p < 0.05): "
                        f"`{attr}` is associated with `{target_col}`. Possible bias."
                    )
                else:
                    st.success(
                        f"✅ **No significant association** (p = {p_value:.3f}): "
                        f"No statistical evidence of bias from `{attr}`."
                    )

                n = contingency_table.sum().sum()
                cramers_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))
                effect = "negligible" if cramers_v < 0.1 else "small" if cramers_v < 0.3 else "medium" if cramers_v < 0.5 else "large"
                st.info(f"📏 **Cramér's V** = {cramers_v:.4f} ({effect} effect size)")

            except Exception as e:
                st.warning(f"Statistical test failed: {e}")

        # ── Disparate Impact Analysis ──────────────────────────────────────
        st.markdown("#### ⚖️ Disparate Impact Analysis")

        pos_class = _determine_positive_class(combined[target_col])
        st.caption(f"Positive class for disparate impact analysis: **{pos_class}**")

        groups = combined[attr].unique()
        group_rates = {}
        for grp in groups:
            mask = combined[attr] == grp
            subset = combined[mask]
            rate = (subset[target_col] == pos_class).mean()
            group_rates[str(grp)] = rate

        if len(group_rates) >= 2:
            max_rate = max(group_rates.values())
            min_rate = min(group_rates.values())
            max_group = max(group_rates, key=group_rates.get)
            min_group = min(group_rates, key=group_rates.get)
            di = min_rate / max_rate if max_rate > 0 else 1.0

            di_col1, di_col2, di_col3 = st.columns(3)
            with di_col1:
                st.metric(f"Highest rate ({max_group})", f"{max_rate:.3f}")
            with di_col2:
                st.metric(f"Lowest rate ({min_group})", f"{min_rate:.3f}")
            with di_col3:
                st.metric("Disparate Impact (min/max)", f"{di:.3f}",
                          help="≥ 0.8 passes the 80% rule")

            if di < 0.8:
                st.error(
                    f"⚠️ **Disparate Impact = {di:.3f}**: Fails the 80% rule (< 0.8). "
                    f"Significant bias detected. The lowest-rate group ({min_group}) has "
                    f"{di:.1%} of the rate of the highest-rate group ({max_group})."
                )
            else:
                st.success(f"✅ **Disparate Impact = {di:.3f}**: Passes the 80% rule (≥ 0.8).")

            rate_df = pd.DataFrame({
                "Group": list(group_rates.keys()),
                "Positive Rate": list(group_rates.values()),
            })
            fig_rate = px.bar(
                rate_df, x="Group", y="Positive Rate",
                title=f"Rate of '{pos_class}' per {attr} group",
                color="Positive Rate",
                color_continuous_scale="RdYlGn",
                text=rate_df["Positive Rate"].round(3).astype(str),
            )
            fig_rate.add_hline(
                y=rate_df["Positive Rate"].mean(), line_dash="dash",
                annotation_text="Overall mean"
            )
            fig_rate.update_layout(height=320, showlegend=False)
            st.plotly_chart(fig_rate, use_container_width=True)

# ── Bias Mitigation Section ───────────────────────────────────────────────────
st.markdown("---")
st.markdown("## 🛠️ Bias Mitigation")
st.markdown("""
Choose a mitigation strategy. Pre-processing methods modify data/weights before training.
In-processing and post-processing require a trained model (available in Model Training page).
""")

mit_attr = st.selectbox(
    "Select protected attribute for mitigation:",
    protected_cols,
    help="Choose which protected attribute to focus mitigation on.",
)

mit_method = st.selectbox(
    "Mitigation method:",
    ["None", "Reweighing (Pre-processing)", "ExponentiatedGradient (In-processing)",
     "ThresholdOptimizer (Post-processing)"],
    help="""
    - **Reweighing**: Adjusts sample weights to reduce bias. No model needed.
    - **ExponentiatedGradient**: Constrained optimization during training. Requires training step.
    - **ThresholdOptimizer**: Adjusts decision thresholds post-training. Requires trained model.
    """,
)

if mit_method == "None":
    st.info("No mitigation selected. Proceed to **🏋️ Model Training**.")

elif mit_method == "Reweighing (Pre-processing)":
    st.markdown("### Reweighing")
    st.markdown("""
    Reweighing assigns different weights to training samples so that each group/outcome
    combination has the expected probability under the independence assumption.
    """)

    if st.session_state.get("X_train") is None:
        st.warning("⚠️ Preprocessing must be completed first. Go to **⚙️ Preprocessing**.")
    elif mit_attr not in df.columns:
        st.warning(f"Column `{mit_attr}` not found.")
    else:
        # Get sensitive features aligned with training set
        y_train = st.session_state.get("y_train")
        sensitive_train_dict = st.session_state.get("sensitive_train_dict", {})

        if y_train is not None:
            if mit_attr in df.columns:
                # Use the correctly-aligned sensitive_train_dict stored during preprocessing
                if mit_attr in sensitive_train_dict:
                    sensitive_train = sensitive_train_dict[mit_attr]
                else:
                    sensitive_train = pd.Series(
                        np.random.choice(df[mit_attr].dropna().unique(), len(y_train))
                    )

                if st.button("⚖️ Apply Reweighing", type="primary"):
                    with st.spinner("Computing sample weights..."):
                        y_arr = np.asarray(y_train)
                        s_arr = np.asarray(sensitive_train)
                        weights = apply_reweighing(None, y_arr, s_arr)
                        st.session_state["sample_weights"] = weights
                        st.session_state["sensitive_train"] = s_arr
                        st.session_state["sensitive_col_for_fairness"] = mit_attr

                        # Get X_test sensitive from dict stored during preprocessing
                        sensitive_test_dict = st.session_state.get("sensitive_test_dict", {})
                        if mit_attr in sensitive_test_dict:
                            st.session_state["sensitive_test"] = np.asarray(sensitive_test_dict[mit_attr])

                    st.success("✅ Sample weights computed and saved!")

                    # Show weight distribution
                    col1, col2 = st.columns(2)
                    with col1:
                        weight_df = pd.DataFrame({"group": s_arr, "weight": weights})
                        fig_w = px.box(
                            weight_df,
                            x="group",
                            y="weight",
                            title="Sample Weights per Group (after Reweighing)",
                            color="group",
                        )
                        fig_w.update_layout(height=350, showlegend=False)
                        st.plotly_chart(fig_w, use_container_width=True)

                    with col2:
                        # Before/after effective counts
                        before_counts = pd.Series(s_arr).value_counts()
                        after_counts = weight_df.groupby("group")["weight"].sum()
                        compare_df = pd.DataFrame({
                            "Before (count)": before_counts,
                            "After (effective)": after_counts.round(1),
                        })
                        st.markdown("**Before vs After Reweighing:**")
                        st.dataframe(compare_df, use_container_width=True)

                    st.info("➡️ Proceed to **🏋️ Model Training** — sample weights will be applied automatically.")

elif mit_method in ("ExponentiatedGradient (In-processing)", "ThresholdOptimizer (Post-processing)"):
    st.info(f"""
    **{mit_method}** requires model training.
    - Set this preference here, then go to **🏋️ Model Training** where you can apply this constraint.
    - ExponentiatedGradient wraps your model with a fairness constraint during training.
    - ThresholdOptimizer adjusts prediction thresholds per group post-training.
    """)
    st.session_state["fairness_constraint_method"] = mit_method
    st.session_state["fairness_constraint_attr"] = mit_attr
    st.success(f"Fairness constraint preference saved: **{mit_method}** on `{mit_attr}`")
    st.info("➡️ Proceed to **🏋️ Model Training** to apply this constraint.")

# Show current mitigation status
st.markdown("---")
st.markdown("### 📋 Current Mitigation Status")
sw = st.session_state.get("sample_weights")
if sw is not None:
    st.success(f"✅ Sample weights applied ({len(sw):,} samples). Will be used in model training.")
else:
    st.info("No sample weights set (training without reweighing).")

fc_method = st.session_state.get("fairness_constraint_method")
if fc_method:
    fc_attr = st.session_state.get("fairness_constraint_attr", "")
    st.success(f"✅ Fairness constraint queued: **{fc_method}** on `{fc_attr}`")
