"""
Page 9: Causal Inference
Estimates the causal effect of a treatment variable on the outcome using
propensity-score-based and regression-adjustment methods, with bootstrap
confidence intervals and permutation p-values.
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
from scipy import stats as sp_stats

st.set_page_config(page_title="Causal Inference - ML Fairness Studio", layout="wide")

st.title("🔗 Causal Inference")
st.markdown("""
Estimate whether a feature has a **causal** effect on the outcome — not just a
correlation.  This page uses the original (pre-processed) dataset so that variable
values remain interpretable.

**Example (Adult dataset):** Does obtaining higher education *cause* a person to earn
more than 50 K, after controlling for age, occupation, and other confounders?
""")

# ── Guard ──────────────────────────────────────────────────────────────────────
df_raw = st.session_state.get("df")
target_col = st.session_state.get("target_col")

if df_raw is None or target_col is None:
    st.warning("⚠️ No dataset found. Please complete **📂 Data Upload** first.")
    st.stop()

# ── Optional DoWhy availability ────────────────────────────────────────────────
try:
    import dowhy
    DOWHY_AVAILABLE = True
except ImportError:
    DOWHY_AVAILABLE = False

# ── 1. Variable selection ──────────────────────────────────────────────────────
st.markdown("## 1️⃣ Select Variables")

all_cols = [c for c in df_raw.columns if c != target_col]

col_t, col_o = st.columns(2)
with col_t:
    treatment_col = st.selectbox(
        "Treatment variable (the cause you want to test):",
        all_cols,
        index=0,
        help="E.g. 'education-num' to test whether education causes higher income.",
    )
with col_o:
    st.markdown(f"**Outcome:** `{target_col}`")
    st.caption("The outcome is fixed as the target column configured in Data Upload.")

# Binarize treatment if needed
treat_series = df_raw[treatment_col].dropna()
is_binary = treat_series.nunique() <= 2

st.markdown("### Treatment Definition")
if is_binary:
    unique_vals = sorted(treat_series.unique())
    treat_pos = st.selectbox(
        "Which value of the treatment is the 'treated' (=1) group?",
        unique_vals,
        index=len(unique_vals) - 1,
    )
    T = (df_raw[treatment_col] == treat_pos).astype(int)
    st.info(f"Treatment: `{treatment_col}` == `{treat_pos}` → 1, else → 0")
else:
    if pd.api.types.is_numeric_dtype(treat_series):
        thresh_default = float(treat_series.median())
        thresh = st.slider(
            f"Binarize `{treatment_col}` — threshold (≥ threshold → treated=1):",
            float(treat_series.min()),
            float(treat_series.max()),
            thresh_default,
        )
        T = (df_raw[treatment_col] >= thresh).astype(int)
        st.info(f"Treatment: `{treatment_col}` ≥ {thresh:.2f} → 1  |  n_treated={T.sum()}, n_control={(T==0).sum()}")
    else:
        cats = sorted(treat_series.unique())
        treat_cat = st.selectbox("Select the 'treated' category:", cats, index=len(cats) - 1)
        T = (df_raw[treatment_col] == treat_cat).astype(int)
        st.info(f"Treatment: `{treatment_col}` == `{treat_cat}` → 1")

# Binarize outcome
outcome_series = df_raw[target_col].dropna()
if pd.api.types.is_numeric_dtype(outcome_series) and outcome_series.nunique() > 2:
    out_thresh = st.slider(
        f"Binarize outcome `{target_col}` (≥ threshold → 1):",
        float(outcome_series.min()), float(outcome_series.max()),
        float(outcome_series.median()),
    )
    Y = (df_raw[target_col] >= out_thresh).astype(int)
else:
    out_classes = sorted(outcome_series.unique())
    out_pos_cls = st.selectbox(
        f"Which value of `{target_col}` is the positive outcome (=1)?",
        out_classes, index=len(out_classes) - 1,
    )
    Y = (df_raw[target_col] == out_pos_cls).astype(int)

# Confounders
st.markdown("### Confounders")
st.markdown(
    "Select variables that may affect **both** the treatment and the outcome. "
    "Failing to control for confounders leads to biased causal estimates."
)
potential_confounders = [c for c in all_cols if c != treatment_col]
confounders = st.multiselect(
    "Confounder variables:",
    potential_confounders,
    default=potential_confounders[:min(5, len(potential_confounders))],
)

# ── Build analysis DataFrame ───────────────────────────────────────────────────
complete_idx = df_raw[confounders + [treatment_col, target_col]].dropna().index
df_ci = df_raw.loc[complete_idx].copy()
T_ci = T.loc[complete_idx].values
Y_ci = Y.loc[complete_idx].values

st.markdown(f"**Complete cases:** {len(df_ci):,}  |  "
            f"Treated: {T_ci.sum():,}  |  Control: {(T_ci == 0).sum():,}")

# ── 2. Run Causal Analysis ─────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## 2️⃣ Estimation Methods")

methods_help = """
| Method | Description |
|--------|-------------|
| **Naive Difference** | Raw mean-outcome difference between treated and control. **Biased** if confounders exist. |
| **Regression Adjustment** | Fit a logistic regression on confounders + treatment; marginal ATE via g-computation. |
| **IPW (Inverse Probability Weighting)** | Weight each unit by the inverse of its propensity to be treated. |
| **Doubly Robust (AIPW)** | Combines regression + IPW; consistent if either model is correct. |
| **DoWhy** | Causal graph + multiple identification strategies *(requires `pip install dowhy`)*. |
"""
with st.expander("ℹ️ Method descriptions"):
    st.markdown(methods_help)

n_boot = st.slider("Bootstrap iterations for CIs", 50, 500, 100)
selected_methods = st.multiselect(
    "Estimation methods to run:",
    ["Naive Difference", "Regression Adjustment", "IPW", "Doubly Robust (AIPW)"]
    + (["DoWhy"] if DOWHY_AVAILABLE else []),
    default=["Naive Difference", "Regression Adjustment", "IPW", "Doubly Robust (AIPW)"],
)

if st.button("🔗 Run Causal Analysis", type="primary", use_container_width=True):
    if not confounders:
        st.warning("⚠️ No confounders selected — only the Naive Difference will be meaningful.")

    # ── Prepare feature matrix ──────────────────────────────────────────────
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer

    num_conf = [c for c in confounders if pd.api.types.is_numeric_dtype(df_ci[c])]
    cat_conf = [c for c in confounders if not pd.api.types.is_numeric_dtype(df_ci[c])]

    transformers = []
    if num_conf:
        transformers.append(("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
        ]), num_conf))
    if cat_conf:
        transformers.append(("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), cat_conf))

    if transformers:
        ct = ColumnTransformer(transformers)
        X_conf = ct.fit_transform(df_ci[confounders])
    else:
        X_conf = np.zeros((len(df_ci), 1))

    # ── Helper functions ────────────────────────────────────────────────────
    def _naive_diff(T_arr, Y_arr):
        return Y_arr[T_arr == 1].mean() - Y_arr[T_arr == 0].mean()

    def _fit_ps(X_c, T_arr):
        ps_model = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
        ps_model.fit(X_c, T_arr)
        ps = np.clip(ps_model.predict_proba(X_c)[:, 1], 1e-3, 1 - 1e-3)
        return ps

    def _reg_adj(X_c, T_arr, Y_arr):
        """G-computation / regression adjustment ATE."""
        X_with_T = np.column_stack([X_c, T_arr])
        out_model = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
        out_model.fit(X_with_T, Y_arr)
        X_treat1 = np.column_stack([X_c, np.ones(len(T_arr))])
        X_treat0 = np.column_stack([X_c, np.zeros(len(T_arr))])
        ate = (out_model.predict_proba(X_treat1)[:, 1] -
               out_model.predict_proba(X_treat0)[:, 1]).mean()
        return ate

    def _ipw(X_c, T_arr, Y_arr):
        ps = _fit_ps(X_c, T_arr)
        w1 = T_arr / ps
        w0 = (1 - T_arr) / (1 - ps)
        ate = (w1 * Y_arr).sum() / w1.sum() - (w0 * Y_arr).sum() / w0.sum()
        return ate

    def _aipw(X_c, T_arr, Y_arr):
        ps = _fit_ps(X_c, T_arr)
        X_with_T = np.column_stack([X_c, T_arr])
        out_m = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
        out_m.fit(X_with_T, Y_arr)
        mu1 = out_m.predict_proba(np.column_stack([X_c, np.ones(len(T_arr))]))[:, 1]
        mu0 = out_m.predict_proba(np.column_stack([X_c, np.zeros(len(T_arr))]))[:, 1]
        phi1 = mu1 + T_arr * (Y_arr - mu1) / ps
        phi0 = mu0 + (1 - T_arr) * (Y_arr - mu0) / (1 - ps)
        return (phi1 - phi0).mean()

    method_fns = {
        "Naive Difference": _naive_diff,
        "Regression Adjustment": lambda X_c, T_arr, Y_arr: _reg_adj(X_c, T_arr, Y_arr),
        "IPW": lambda X_c, T_arr, Y_arr: _ipw(X_c, T_arr, Y_arr),
        "Doubly Robust (AIPW)": lambda X_c, T_arr, Y_arr: _aipw(X_c, T_arr, Y_arr),
    }

    results = []
    rng = np.random.RandomState(42)
    n = len(T_ci)

    progress = st.progress(0)
    status = st.empty()

    for mi, method in enumerate(selected_methods):
        if method == "DoWhy":
            continue  # handled separately below

        status.info(f"Running {method}...")
        fn = method_fns.get(method)
        if fn is None:
            continue

        # Point estimate
        try:
            if method == "Naive Difference":
                ate_pt = fn(T_ci, Y_ci)
            else:
                ate_pt = fn(X_conf, T_ci, Y_ci)
        except Exception as e:
            st.warning(f"{method} failed: {e}")
            continue

        # Bootstrap CIs
        boot_ates = []
        for _ in range(n_boot):
            idx_b = rng.choice(n, size=n, replace=True)
            try:
                if method == "Naive Difference":
                    b_ate = fn(T_ci[idx_b], Y_ci[idx_b])
                else:
                    b_ate = fn(X_conf[idx_b], T_ci[idx_b], Y_ci[idx_b])
                boot_ates.append(b_ate)
            except Exception:
                pass

        boot_arr = np.array(boot_ates)
        ci_lo = np.percentile(boot_arr, 2.5)
        ci_hi = np.percentile(boot_arr, 97.5)
        se = boot_arr.std()

        # p-value: two-sided z-test (ATE / SE)
        if se > 0:
            z = ate_pt / se
            p_val = 2 * sp_stats.norm.sf(abs(z))
        else:
            p_val = 0.0 if abs(ate_pt) > 0 else 1.0

        results.append({
            "Method": method,
            "ATE": ate_pt,
            "CI Lower (95%)": ci_lo,
            "CI Upper (95%)": ci_hi,
            "SE (bootstrap)": se,
            "p-value": p_val,
            "Significant (α=0.05)": "✅" if p_val < 0.05 else "❌",
        })

        progress.progress(int((mi + 1) / len(selected_methods) * 80))

    # DoWhy
    if "DoWhy" in selected_methods and DOWHY_AVAILABLE:
        status.info("Running DoWhy...")
        try:
            import dowhy
            from dowhy import CausalModel

            df_dw = df_ci[confounders + [treatment_col]].copy()
            df_dw["__treatment__"] = T_ci
            df_dw["__outcome__"] = Y_ci

            common_causes = confounders if confounders else []
            dw_model = CausalModel(
                data=df_dw,
                treatment="__treatment__",
                outcome="__outcome__",
                common_causes=common_causes,
            )
            identified = dw_model.identify_effect(proceed_when_unidentifiable=True)
            estimate = dw_model.estimate_effect(
                identified,
                method_name="backdoor.linear_regression",
            )
            dw_ate = float(estimate.value)
            results.append({
                "Method": "DoWhy (Linear Regression)",
                "ATE": dw_ate,
                "CI Lower (95%)": dw_ate - 1.96 * estimate.get_standard_error() if hasattr(estimate, "get_standard_error") else float("nan"),
                "CI Upper (95%)": dw_ate + 1.96 * estimate.get_standard_error() if hasattr(estimate, "get_standard_error") else float("nan"),
                "SE (bootstrap)": float("nan"),
                "p-value": float("nan"),
                "Significant (α=0.05)": "N/A",
            })
        except Exception as e:
            st.warning(f"DoWhy failed: {e}")

    progress.progress(100)
    status.empty()
    progress.empty()

    if results:
        res_df = pd.DataFrame(results)
        st.session_state["_causal_results"] = res_df
        st.session_state["_causal_ps"] = None

        # Compute propensity scores for plots (use last non-naive method)
        if "Regression Adjustment" in selected_methods or "IPW" in selected_methods or "Doubly Robust (AIPW)" in selected_methods:
            try:
                ps_vals = _fit_ps(X_conf, T_ci)
                st.session_state["_causal_ps"] = ps_vals
            except Exception:
                pass

        st.success("✅ Causal analysis complete!")
    else:
        st.error("All methods failed. Check that at least one method is selected and data is valid.")


# ── Display results ────────────────────────────────────────────────────────────
res_df = st.session_state.get("_causal_results")
ps_vals = st.session_state.get("_causal_ps")

if res_df is not None:
    st.markdown("---")
    st.markdown("## 3️⃣ Results")

    # Summary table
    st.markdown("### Average Treatment Effect (ATE) Summary")
    st.markdown("""
    **ATE** = E[Y(1) − Y(0)] = the average change in outcome probability if everyone were treated
    vs. everyone were untreated.  A positive ATE means the treatment *increases* the probability
    of the positive outcome.
    """)

    fmt = {
        "ATE": "{:.4f}",
        "CI Lower (95%)": "{:.4f}",
        "CI Upper (95%)": "{:.4f}",
        "SE (bootstrap)": "{:.4f}",
        "p-value": "{:.4f}",
    }
    st.dataframe(
        res_df.style.format(fmt, na_rep="N/A")
        .background_gradient(subset=["ATE"], cmap="RdYlGn", vmin=-0.3, vmax=0.3),
        use_container_width=True,
        hide_index=True,
    )

    # Forest plot
    st.markdown("### Forest Plot (ATE with 95% CI)")
    numeric_res = res_df.dropna(subset=["CI Lower (95%)", "CI Upper (95%)"])
    if len(numeric_res) > 0:
        fig_forest = go.Figure()
        colors = ["#e74c3c" if s == "✅" else "#7f8c8d" for s in numeric_res["Significant (α=0.05)"]]

        for i, row in numeric_res.iterrows():
            color = "#e74c3c" if row["Significant (α=0.05)"] == "✅" else "#7f8c8d"
            fig_forest.add_trace(go.Scatter(
                x=[row["CI Lower (95%)"], row["ATE"], row["CI Upper (95%)"]],
                y=[row["Method"]] * 3,
                mode="lines+markers",
                marker=dict(
                    symbol=["line-ew", "diamond", "line-ew"],
                    size=[8, 12, 8],
                    color=color,
                ),
                line=dict(color=color, width=2),
                name=row["Method"],
                showlegend=False,
            ))

        fig_forest.add_vline(x=0, line_dash="dash", line_color="black",
                             annotation_text="No effect (ATE=0)")
        fig_forest.update_layout(
            title="Forest Plot: Average Treatment Effect",
            xaxis_title="ATE (change in outcome probability)",
            height=max(300, len(numeric_res) * 60 + 100),
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_forest, use_container_width=True)

    # Interpretation
    st.markdown("### Interpretation")
    sig_methods = res_df[res_df["Significant (α=0.05)"] == "✅"]
    if len(sig_methods) > 0:
        ate_vals = sig_methods["ATE"].dropna()
        mean_ate = ate_vals.mean()
        direction = "increases" if mean_ate > 0 else "decreases"
        st.success(
            f"**{len(sig_methods)} method(s)** find a statistically significant causal effect. "
            f"On average, the treatment **{direction}** the probability of the positive outcome "
            f"by **{abs(mean_ate):.1%}** (ATE ≈ {mean_ate:+.4f})."
        )
    else:
        st.info(
            "No method found a statistically significant causal effect at α = 0.05. "
            "This may indicate no causal relationship, insufficient data, or unmeasured confounders."
        )

    # Propensity score plots
    if ps_vals is not None:
        st.markdown("---")
        st.markdown("### Propensity Score Diagnostics")
        st.markdown("""
        Propensity scores estimate the probability of being treated given the confounders.
        Good overlap (similar distributions for treated and control) is necessary for valid causal estimates.
        """)

        T_arr_diag = st.session_state.get("_causal_T_for_diag")
        # Use session-stored T if available, else fall back
        try:
            complete_idx2 = df_raw[confounders + [treatment_col, target_col]].dropna().index
            T_diag = T.loc[complete_idx2].values
        except Exception:
            T_diag = None

        if T_diag is not None and len(ps_vals) == len(T_diag):
            ps_df = pd.DataFrame({
                "Propensity Score": ps_vals,
                "Group": ["Treated" if t == 1 else "Control" for t in T_diag],
            })

            fig_ps = px.histogram(
                ps_df,
                x="Propensity Score",
                color="Group",
                nbins=30,
                barmode="overlay",
                opacity=0.65,
                title="Propensity Score Distribution",
                color_discrete_map={"Treated": "#e74c3c", "Control": "#3498db"},
            )
            fig_ps.update_layout(height=350)
            st.plotly_chart(fig_ps, use_container_width=True)

            # Love plot: Standardized Mean Differences
            if confounders:
                st.markdown("#### Covariate Balance (Standardized Mean Differences)")
                from sklearn.linear_model import LogisticRegression
                from sklearn.preprocessing import StandardScaler

                complete_idx3 = df_raw[confounders + [treatment_col, target_col]].dropna().index
                df_bal = df_raw.loc[complete_idx3].copy()
                T_bal = T.loc[complete_idx3].values

                num_conf_bal = [c for c in confounders if pd.api.types.is_numeric_dtype(df_bal[c])]
                if num_conf_bal:
                    smd_rows = []
                    for col in num_conf_bal:
                        vals = df_bal[col].values.astype(float)
                        t_vals = vals[T_bal == 1]
                        c_vals = vals[T_bal == 0]
                        pooled_sd = np.sqrt((t_vals.var() + c_vals.var()) / 2)
                        smd = (t_vals.mean() - c_vals.mean()) / (pooled_sd + 1e-8)

                        # IPW-weighted SMD
                        w = np.where(T_bal == 1, 1 / ps_vals, 1 / (1 - ps_vals))
                        w1 = w[T_bal == 1] / w[T_bal == 1].sum()
                        w0 = w[T_bal == 0] / w[T_bal == 0].sum()
                        wt_mean1 = (t_vals * w1).sum()
                        wt_mean0 = (c_vals * w0).sum()
                        smd_adj = (wt_mean1 - wt_mean0) / (pooled_sd + 1e-8)

                        smd_rows.append({"Feature": col, "Before IPW": smd, "After IPW": smd_adj})

                    smd_df = pd.DataFrame(smd_rows)
                    fig_love = go.Figure()
                    fig_love.add_trace(go.Scatter(
                        x=smd_df["Before IPW"], y=smd_df["Feature"],
                        mode="markers", name="Before IPW",
                        marker=dict(color="#e74c3c", size=10, symbol="circle"),
                    ))
                    fig_love.add_trace(go.Scatter(
                        x=smd_df["After IPW"], y=smd_df["Feature"],
                        mode="markers", name="After IPW",
                        marker=dict(color="#2ecc71", size=10, symbol="diamond"),
                    ))
                    fig_love.add_vline(x=0, line_color="black")
                    fig_love.add_vline(x=0.1, line_dash="dash", line_color="orange",
                                       annotation_text="±0.1 threshold")
                    fig_love.add_vline(x=-0.1, line_dash="dash", line_color="orange")
                    fig_love.update_layout(
                        title="Love Plot — Standardized Mean Differences",
                        xaxis_title="Standardized Mean Difference",
                        height=max(300, len(num_conf_bal) * 35 + 100),
                    )
                    st.plotly_chart(fig_love, use_container_width=True)
                    st.caption(
                        "Covariates with |SMD| > 0.1 after IPW weighting may indicate poor overlap "
                        "and should be investigated."
                    )

st.info("➡️ Next step: Navigate to **📓 Download Notebook** to export the full analysis.")
