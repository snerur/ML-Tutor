"""
Page 7: Feature Importance
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

st.set_page_config(page_title="Feature Importance - ML Fairness Studio", layout="wide")

st.title("📊 Feature Importance")

# ── Guard ─────────────────────────────────────────────────────────────────────
if st.session_state.get("model") is None:
    st.warning("⚠️ No trained model found. Please complete **🏋️ Model Training** first.")
    st.stop()
if st.session_state.get("X_train") is None:
    st.warning("⚠️ No training data found. Please complete **⚙️ Preprocessing** first.")
    st.stop()

model = st.session_state["model"]
X_train = st.session_state["X_train"]
X_test = st.session_state["X_test"]
y_test = np.asarray(st.session_state["y_test"])
feature_names = st.session_state.get("feature_names", [])
model_name = st.session_state.get("model_name", "Model")

# Ensure feature_names is available
if not feature_names:
    feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]

st.markdown(f"**Model:** {model_name} | **Features:** {len(feature_names)}")

# ── SHAP availability ─────────────────────────────────────────────────────────
try:
    import shap
    SHAP_AVAILABLE = True
    SHAP_ERROR = None
except Exception as _shap_exc:
    SHAP_AVAILABLE = False
    SHAP_ERROR = str(_shap_exc)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏆 Model Importance",
    "🔮 SHAP Values",
    "🔀 Permutation Importance",
    "👤 Individual Prediction",
    "📈 CI & Significance",
])

# ── Tab 1: Model Feature Importance ──────────────────────────────────────────
with tab1:
    st.markdown("### Model-Native Feature Importance")

    importance_values = None
    importance_type = None

    # Get the underlying estimator if wrapped (e.g., ExponentiatedGradient)
    underlying_model = model
    if hasattr(model, "_pmf_predict") or hasattr(model, "predictors_"):
        # ExponentiatedGradient — use best predictor
        if hasattr(model, "predictors_") and len(model.predictors_) > 0:
            underlying_model = model.predictors_[0]

    if hasattr(underlying_model, "feature_importances_"):
        importance_values = underlying_model.feature_importances_
        importance_type = "Feature Importances (Gini/Gain)"
    elif hasattr(underlying_model, "coef_"):
        coef = underlying_model.coef_
        if coef.ndim > 1:
            importance_values = np.abs(coef).mean(axis=0)
        else:
            importance_values = np.abs(coef)
        importance_type = "Feature Coefficients (|value|)"
    else:
        st.warning(f"⚠️ {model_name} does not expose native feature importances. "
                   "Try SHAP or Permutation Importance instead.")

    if importance_values is not None:
        top_n = st.slider("Show top N features", 5, min(50, len(feature_names)), min(20, len(feature_names)))
        imp_df = pd.DataFrame({
            "Feature": feature_names[:len(importance_values)],
            "Importance": importance_values,
        }).sort_values("Importance", ascending=False).head(top_n)

        fig_imp = px.bar(
            imp_df,
            x="Importance",
            y="Feature",
            orientation="h",
            title=f"{importance_type} (Top {top_n})",
            color="Importance",
            color_continuous_scale="Blues",
            text=imp_df["Importance"].round(4).astype(str),
        )
        fig_imp.update_layout(
            height=max(400, top_n * 22),
            yaxis=dict(autorange="reversed"),
            showlegend=False,
        )
        st.plotly_chart(fig_imp, use_container_width=True)

        # Cumulative importance
        imp_df_all = pd.DataFrame({
            "Feature": feature_names[:len(importance_values)],
            "Importance": importance_values,
        }).sort_values("Importance", ascending=False)
        imp_df_all["Cumulative"] = imp_df_all["Importance"].cumsum() / imp_df_all["Importance"].sum()
        fig_cum = px.line(
            imp_df_all.head(min(50, len(imp_df_all))),
            x=range(1, min(51, len(imp_df_all) + 1)),
            y="Cumulative",
            title="Cumulative Feature Importance",
            labels={"x": "Number of Features", "y": "Cumulative Importance"},
        )
        fig_cum.add_hline(y=0.8, line_dash="dash", line_color="orange",
                          annotation_text="80% threshold")
        fig_cum.add_hline(y=0.95, line_dash="dash", line_color="red",
                          annotation_text="95% threshold")
        fig_cum.update_layout(height=350)
        st.plotly_chart(fig_cum, use_container_width=True)

# ── Tab 2: SHAP Values ────────────────────────────────────────────────────────
with tab2:
    st.markdown("### SHAP (SHapley Additive exPlanations) Values")

    if not SHAP_AVAILABLE:
        st.error("❌ SHAP could not be loaded.")
        if SHAP_ERROR:
            st.code(SHAP_ERROR, language="text")
        st.markdown("""
**Most likely cause:** `shap` was compiled against NumPy 1.x and is incompatible with NumPy 2.x.

**Fix — run one of the following:**
```bash
# Option A (recommended): downgrade NumPy to the 1.x series
pip install "numpy>=1.24,<2.0" --force-reinstall

# Option B: upgrade shap to a NumPy-2.0-compatible build
pip install "shap>=0.46.0" --upgrade
```
Then **restart the Streamlit server**. Permutation Importance (tab 3) works without SHAP.
""")
    else:
        sample_size = st.slider(
            "Sample size for SHAP (larger = more accurate but slower)",
            50, min(500, X_train.shape[0]), min(200, X_train.shape[0]),
        )

        if st.button("🔮 Compute SHAP Values", key="compute_shap"):
            with st.spinner("Computing SHAP values... This may take a moment."):
                try:
                    # Sample data
                    rng = np.random.RandomState(42)
                    idx = rng.choice(X_train.shape[0], size=min(sample_size, X_train.shape[0]), replace=False)
                    X_sample = X_train[idx]

                    # Choose appropriate explainer
                    inner = underlying_model
                    model_type = type(inner).__name__

                    if model_type in ("RandomForestClassifier", "GradientBoostingClassifier",
                                     "XGBClassifier", "LGBMClassifier", "DecisionTreeClassifier",
                                     "AdaBoostClassifier", "ExtraTreesClassifier"):
                        explainer = shap.TreeExplainer(inner)
                    elif model_type in ("LogisticRegression", "LinearSVC"):
                        explainer = shap.LinearExplainer(inner, X_sample)
                    else:
                        explainer = shap.KernelExplainer(
                            inner.predict_proba if hasattr(inner, "predict_proba") else inner.predict,
                            shap.sample(X_sample, min(50, len(X_sample))),
                        )

                    shap_values = explainer.shap_values(X_sample)

                    # Handle multi-class (list) vs binary (array)
                    if isinstance(shap_values, list):
                        shap_vals_2d = shap_values[1] if len(shap_values) > 1 else shap_values[0]
                    elif shap_values.ndim == 3:
                        shap_vals_2d = shap_values[:, :, 1]
                    else:
                        shap_vals_2d = shap_values

                    st.session_state["_shap_values"] = shap_vals_2d
                    st.session_state["_shap_X"] = X_sample
                    st.session_state["_shap_explainer"] = explainer
                    st.success("✅ SHAP values computed!")

                except Exception as e:
                    st.error(f"SHAP computation failed: {e}")
                    import traceback
                    st.code(traceback.format_exc())

        # Show SHAP plots if computed
        shap_vals = st.session_state.get("_shap_values")
        shap_X = st.session_state.get("_shap_X")

        if shap_vals is not None and shap_X is not None:
            fn_arr = np.array(feature_names[:shap_vals.shape[1]])

            col_s1, col_s2 = st.columns(2)

            with col_s1:
                st.markdown("#### Summary Bar Plot")
                mean_abs_shap = np.abs(shap_vals).mean(axis=0)
                top_n_shap = st.slider("Top features", 5, min(30, len(fn_arr)), min(15, len(fn_arr)), key="top_shap")
                top_idx = np.argsort(mean_abs_shap)[::-1][:top_n_shap]
                shap_summary = pd.DataFrame({
                    "Feature": fn_arr[top_idx],
                    "Mean |SHAP|": mean_abs_shap[top_idx],
                }).sort_values("Mean |SHAP|", ascending=True)
                fig_shap_bar = px.bar(
                    shap_summary,
                    x="Mean |SHAP|",
                    y="Feature",
                    orientation="h",
                    title="Mean |SHAP| Values",
                    color="Mean |SHAP|",
                    color_continuous_scale="Reds",
                )
                fig_shap_bar.update_layout(height=max(400, top_n_shap * 25), showlegend=False)
                st.plotly_chart(fig_shap_bar, use_container_width=True)

            with col_s2:
                st.markdown("#### SHAP Bee Swarm Plot (matplotlib)")
                try:
                    fig_sw, ax_sw = plt.subplots(figsize=(8, max(4, top_n_shap * 0.35)))
                    shap.summary_plot(
                        shap_vals[:, top_idx],
                        shap_X[:, top_idx],
                        feature_names=fn_arr[top_idx].tolist(),
                        show=False,
                        max_display=top_n_shap,
                        plot_type="dot",
                    )
                    st.pyplot(fig_sw, use_container_width=True)
                    plt.close(fig_sw)
                except Exception as e:
                    st.warning(f"Bee swarm plot failed: {e}")

            # SHAP heatmap (scatter of top 2 features)
            if shap_vals.shape[1] >= 2:
                st.markdown("#### SHAP Value Distribution (Top Features)")
                f1_idx = top_idx[0]
                f2_idx = top_idx[1] if len(top_idx) > 1 else top_idx[0]
                scatter_df = pd.DataFrame({
                    fn_arr[f1_idx]: shap_X[:, f1_idx],
                    f"SHAP({fn_arr[f1_idx]})": shap_vals[:, f1_idx],
                })
                fig_scatter = px.scatter(
                    scatter_df,
                    x=fn_arr[f1_idx],
                    y=f"SHAP({fn_arr[f1_idx]})",
                    color=f"SHAP({fn_arr[f1_idx]})",
                    color_continuous_scale="RdBu_r",
                    title=f"SHAP Dependence: {fn_arr[f1_idx]}",
                )
                fig_scatter.update_layout(height=380)
                st.plotly_chart(fig_scatter, use_container_width=True)

# ── Tab 3: Permutation Importance ─────────────────────────────────────────────
with tab3:
    st.markdown("### Permutation Feature Importance")
    st.markdown("""
    Permutation importance measures the decrease in model performance when a feature's
    values are randomly shuffled, breaking the relationship between the feature and the target.
    """)

    n_repeats = st.slider("Number of permutation repeats", 3, 20, 5)
    perm_sample = st.slider(
        "Test sample size for permutation",
        50,
        min(1000, X_test.shape[0]),
        min(500, X_test.shape[0]),
    )

    if st.button("🔀 Compute Permutation Importance", key="compute_perm"):
        with st.spinner("Computing permutation importance..."):
            try:
                from sklearn.inspection import permutation_importance
                rng = np.random.RandomState(42)
                idx = rng.choice(X_test.shape[0], size=perm_sample, replace=False)
                X_perm = X_test[idx]
                y_perm = y_test[idx]

                perm_result = permutation_importance(
                    model, X_perm, y_perm,
                    n_repeats=n_repeats,
                    random_state=42,
                    n_jobs=-1,
                )

                perm_df = pd.DataFrame({
                    "Feature": feature_names[:len(perm_result.importances_mean)],
                    "Mean Importance": perm_result.importances_mean,
                    "Std": perm_result.importances_std,
                }).sort_values("Mean Importance", ascending=False)

                st.session_state["_perm_importance"] = perm_df
                st.session_state["_perm_n_repeats"] = n_repeats
                st.session_state["_perm_importances_raw"] = perm_result.importances
                st.success("✅ Permutation importance computed!")

            except Exception as e:
                st.error(f"Permutation importance failed: {e}")
                import traceback
                st.code(traceback.format_exc())

    perm_df = st.session_state.get("_perm_importance")
    if perm_df is not None:
        top_n_perm = st.slider("Top N features", 5, min(30, len(perm_df)), min(20, len(perm_df)), key="top_perm")
        perm_top = perm_df.head(top_n_perm).sort_values("Mean Importance", ascending=True)

        fig_perm = go.Figure()
        fig_perm.add_trace(go.Bar(
            x=perm_top["Mean Importance"],
            y=perm_top["Feature"],
            orientation="h",
            error_x=dict(type="data", array=perm_top["Std"], visible=True),
            marker_color="steelblue",
            name="Permutation Importance",
        ))
        fig_perm.add_vline(x=0, line_color="red", line_dash="dash", annotation_text="No effect")
        fig_perm.update_layout(
            title=f"Permutation Importance (Top {top_n_perm})",
            xaxis_title="Mean Decrease in Accuracy",
            height=max(400, top_n_perm * 24),
        )
        st.plotly_chart(fig_perm, use_container_width=True)

        # Table
        st.markdown("#### Permutation Importance Values")
        st.dataframe(
            perm_df.head(top_n_perm).style.format({
                "Mean Importance": "{:.4f}",
                "Std": "{:.4f}",
            }).background_gradient(subset=["Mean Importance"], cmap="Blues"),
            use_container_width=True,
            hide_index=True,
        )

# ── Tab 4: Individual Prediction Explanation ──────────────────────────────────
with tab4:
    st.markdown("### Individual Prediction Explanation (SHAP)")

    if not SHAP_AVAILABLE:
        st.error("❌ SHAP unavailable — see the **SHAP Values** tab for fix instructions.")
    else:
        shap_explainer = st.session_state.get("_shap_explainer")
        if shap_explainer is None:
            st.info("Please compute SHAP values in the **SHAP Values** tab first.")
        else:
            max_row = X_test.shape[0] - 1
            row_idx = st.slider("Select test sample index", 0, max_row, 0)

            x_row = X_test[row_idx:row_idx + 1]
            pred_class = model.predict(x_row)[0]
            pred_label = f"Predicted: {pred_class}"

            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(x_row)[0]
                classes = model.classes_
                proba_str = " | ".join([f"{c}: {p:.3f}" for c, p in zip(classes, prob)])
                st.markdown(f"**{pred_label}** | Probabilities: {proba_str}")
            else:
                st.markdown(f"**{pred_label}**")

            try:
                row_shap = shap_explainer.shap_values(x_row)

                if isinstance(row_shap, list):
                    row_shap_1d = row_shap[1][0] if len(row_shap) > 1 else row_shap[0][0]
                elif row_shap.ndim == 3:
                    row_shap_1d = row_shap[0, :, 1]
                else:
                    row_shap_1d = row_shap[0]

                fn_arr = np.array(feature_names[:len(row_shap_1d)])
                feature_values = x_row[0, :len(row_shap_1d)]

                # SHAP waterfall (plotly)
                sorted_idx = np.argsort(np.abs(row_shap_1d))[::-1][:15]
                waterfall_df = pd.DataFrame({
                    "Feature": [f"{fn_arr[i]} = {feature_values[i]:.3f}" for i in sorted_idx],
                    "SHAP Value": row_shap_1d[sorted_idx],
                })
                waterfall_df = waterfall_df.sort_values("SHAP Value", ascending=True)

                colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in waterfall_df["SHAP Value"]]

                fig_wf = go.Figure(go.Bar(
                    x=waterfall_df["SHAP Value"],
                    y=waterfall_df["Feature"],
                    orientation="h",
                    marker_color=colors,
                    text=[f"{v:+.4f}" for v in waterfall_df["SHAP Value"]],
                    textposition="auto",
                ))
                fig_wf.update_layout(
                    title=f"SHAP Feature Contributions for Sample #{row_idx}",
                    xaxis_title="SHAP Value (impact on model output)",
                    height=max(400, len(sorted_idx) * 30),
                )
                st.plotly_chart(fig_wf, use_container_width=True)
                st.caption("Red = pushes toward positive class, Green = pushes toward negative class")

                # Try matplotlib force plot
                try:
                    st.markdown("#### SHAP Force Plot")
                    if hasattr(shap_explainer, "expected_value"):
                        exp_val = shap_explainer.expected_value
                        if isinstance(exp_val, list):
                            exp_val = exp_val[1]
                        fig_force, ax_force = plt.subplots(figsize=(14, 3))
                        shap.force_plot(
                            exp_val,
                            row_shap_1d,
                            feature_names=fn_arr.tolist(),
                            matplotlib=True,
                            show=False,
                        )
                        st.pyplot(fig_force, use_container_width=True)
                        plt.close(fig_force)
                except Exception as fp_err:
                    st.info(f"Force plot not available: {fp_err}")

            except Exception as e:
                st.error(f"Individual explanation failed: {e}")

# ── Tab 5: Confidence Intervals & Significance ────────────────────────────────
with tab5:
    st.markdown("### Feature Importance: Confidence Intervals & Statistical Significance")
    st.markdown("""
    **Confidence Intervals** are estimated via bootstrap resampling of the training data.
    **P-values** are derived from a permutation test: the target labels are randomly shuffled
    to build a null distribution, then the p-value is the fraction of null importances that
    exceed the actual importance (one-sided test).

    > A small p-value (< 0.05) means the feature's importance is unlikely to arise by chance.
    """)

    ci_method = st.radio(
        "Importance source",
        ["Model-native (feature_importances_)", "Permutation (from Tab 3)"],
        horizontal=True,
    )

    use_perm = ci_method.startswith("Permutation")

    if use_perm:
        perm_df_ci = st.session_state.get("_perm_importance")
        if perm_df_ci is None:
            st.info("Please compute Permutation Importance in Tab 3 first, then return here.")
        else:
            st.markdown("#### Results from Permutation Importance")
            st.markdown("""
            P-values are computed as a one-sample t-test (H₀: mean importance = 0).
            95% CIs use the t-distribution: mean ± t₀.₀₂₅ × SE.
            """)

            from scipy import stats as _stats

            n_rep = st.session_state.get("_perm_n_repeats", 5)
            # Recompute p-values from importances_raw if stored, else use mean/std
            perm_raw = st.session_state.get("_perm_importances_raw")

            rows_ci = []
            for _, row in perm_df_ci.iterrows():
                mean_imp = row["Mean Importance"]
                std_imp = row["Std"]
                n = n_rep
                se = std_imp / np.sqrt(n) if n > 1 else std_imp
                if se > 0:
                    t_stat = mean_imp / se
                    p_val = _stats.t.sf(t_stat, df=n - 1)  # one-sided: H1: mean > 0
                    ci_half = _stats.t.ppf(0.975, df=n - 1) * se
                else:
                    p_val = 0.0 if mean_imp > 0 else 1.0
                    ci_half = 0.0

                rows_ci.append({
                    "Feature": row["Feature"],
                    "Importance": mean_imp,
                    "CI Lower (95%)": mean_imp - ci_half,
                    "CI Upper (95%)": mean_imp + ci_half,
                    "p-value": p_val,
                    "Significant": "✅" if p_val < 0.05 else "❌",
                })

            ci_df = pd.DataFrame(rows_ci).sort_values("Importance", ascending=False)

            top_n_ci = st.slider("Show top N features", 5, min(50, len(ci_df)), min(20, len(ci_df)), key="top_ci_perm")
            ci_top = ci_df.head(top_n_ci).sort_values("Importance", ascending=True)

            fig_ci = go.Figure()
            sig_colors = ["#e74c3c" if s == "✅" else "#95a5a6" for s in ci_top["Significant"]]
            fig_ci.add_trace(go.Bar(
                x=ci_top["Importance"],
                y=ci_top["Feature"],
                orientation="h",
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=(ci_top["CI Upper (95%)"] - ci_top["Importance"]).clip(lower=0),
                    arrayminus=(ci_top["Importance"] - ci_top["CI Lower (95%)"]).clip(lower=0),
                    visible=True,
                ),
                marker_color=sig_colors,
                name="Permutation Importance",
            ))
            fig_ci.add_vline(x=0, line_color="black", line_dash="dash")
            fig_ci.update_layout(
                title="Feature Importance with 95% Confidence Intervals (red = significant)",
                xaxis_title="Mean Decrease in Score",
                height=max(400, top_n_ci * 26),
            )
            st.plotly_chart(fig_ci, use_container_width=True)

            st.dataframe(
                ci_df.head(top_n_ci).style.format({
                    "Importance": "{:.4f}",
                    "CI Lower (95%)": "{:.4f}",
                    "CI Upper (95%)": "{:.4f}",
                    "p-value": "{:.4f}",
                }).background_gradient(subset=["Importance"], cmap="Blues"),
                use_container_width=True,
                hide_index=True,
            )

    else:
        # Bootstrap CI for model-native importances
        if not hasattr(underlying_model, "feature_importances_"):
            st.warning("This model does not expose `feature_importances_`. Switch to Permutation source or use a tree-based model.")
        else:
            n_boot = st.slider("Bootstrap iterations", 20, 200, 50, key="n_boot_ci")
            alpha = st.slider("Significance level α", 0.01, 0.20, 0.05, 0.01, key="alpha_ci")
            top_n_boot = st.slider("Top N features", 5, min(50, len(feature_names)), min(20, len(feature_names)), key="top_boot_ci")

            if st.button("📈 Run Bootstrap CI Analysis", key="run_boot_ci"):
                with st.spinner(f"Bootstrapping {n_boot} iterations..."):
                    try:
                        from sklearn.base import clone as _clone
                        y_train_arr = np.asarray(st.session_state["y_train"])
                        rng_b = np.random.RandomState(42)
                        n_samples = X_train.shape[0]
                        boot_imps = []

                        for _ in range(n_boot):
                            idx_b = rng_b.choice(n_samples, size=n_samples, replace=True)
                            X_b = X_train[idx_b]
                            y_b = y_train_arr[idx_b]
                            m_b = _clone(underlying_model)
                            m_b.fit(X_b, y_b)
                            boot_imps.append(m_b.feature_importances_)

                        boot_arr = np.array(boot_imps)  # shape (n_boot, n_features)
                        actual_imp = underlying_model.feature_importances_

                        # Permutation test: shuffle y, train, record importances
                        null_imps = []
                        n_null = min(50, n_boot)
                        for _ in range(n_null):
                            idx_p = rng_b.permutation(n_samples)
                            m_p = _clone(underlying_model)
                            m_p.fit(X_train, y_train_arr[idx_p])
                            null_imps.append(m_p.feature_importances_)
                        null_arr = np.array(null_imps)  # shape (n_null, n_features)

                        lower_q = alpha / 2 * 100
                        upper_q = (1 - alpha / 2) * 100

                        rows_boot = []
                        fn_arr = np.array(feature_names[:len(actual_imp)])
                        for i, feat in enumerate(fn_arr):
                            imp_i = actual_imp[i]
                            ci_lo = np.percentile(boot_arr[:, i], lower_q)
                            ci_hi = np.percentile(boot_arr[:, i], upper_q)
                            # p-value: fraction of null dist >= actual importance
                            p_val_b = (null_arr[:, i] >= imp_i).mean()
                            # Adjust for finite null distribution
                            p_val_b = max(p_val_b, 1.0 / (n_null + 1))
                            rows_boot.append({
                                "Feature": feat,
                                "Importance": imp_i,
                                f"CI Lower ({int((1-alpha)*100)}%)": ci_lo,
                                f"CI Upper ({int((1-alpha)*100)}%)": ci_hi,
                                "p-value": p_val_b,
                                "Significant": "✅" if p_val_b < alpha else "❌",
                            })

                        boot_df = pd.DataFrame(rows_boot).sort_values("Importance", ascending=False)
                        st.session_state["_boot_ci_df"] = boot_df
                        st.success("✅ Bootstrap analysis complete!")

                    except Exception as e:
                        st.error(f"Bootstrap failed: {e}")
                        import traceback
                        st.code(traceback.format_exc())

            boot_df = st.session_state.get("_boot_ci_df")
            if boot_df is not None:
                ci_col_lo = [c for c in boot_df.columns if "CI Lower" in c][0]
                ci_col_hi = [c for c in boot_df.columns if "CI Upper" in c][0]

                boot_top = boot_df.head(top_n_boot).sort_values("Importance", ascending=True)
                sig_colors_b = ["#e74c3c" if s == "✅" else "#95a5a6" for s in boot_top["Significant"]]

                fig_boot = go.Figure()
                fig_boot.add_trace(go.Bar(
                    x=boot_top["Importance"],
                    y=boot_top["Feature"],
                    orientation="h",
                    error_x=dict(
                        type="data",
                        symmetric=False,
                        array=(boot_top[ci_col_hi] - boot_top["Importance"]).clip(lower=0),
                        arrayminus=(boot_top["Importance"] - boot_top[ci_col_lo]).clip(lower=0),
                        visible=True,
                    ),
                    marker_color=sig_colors_b,
                    name="Feature Importance",
                ))
                fig_boot.update_layout(
                    title=f"Bootstrap Feature Importance with {ci_col_lo.split('(')[1].rstrip(')')} CI (red = significant)",
                    xaxis_title="Importance",
                    height=max(400, top_n_boot * 26),
                )
                st.plotly_chart(fig_boot, use_container_width=True)

                fmt_cols = {c: "{:.4f}" for c in ["Importance", ci_col_lo, ci_col_hi, "p-value"]}
                st.dataframe(
                    boot_df.head(top_n_boot).style.format(fmt_cols)
                    .background_gradient(subset=["Importance"], cmap="Blues"),
                    use_container_width=True,
                    hide_index=True,
                )

                n_sig = (boot_df.head(top_n_boot)["Significant"] == "✅").sum()
                st.info(f"**{n_sig} / {top_n_boot}** features are statistically significant at α = {st.session_state.get('alpha_ci', 0.05)}")

st.info("➡️ Next step: Navigate to **🤖 LLM Analysis** for AI-powered insights.")
