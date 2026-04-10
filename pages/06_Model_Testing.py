"""
Page 6: Model Testing & Evaluation
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

from utils.ml_utils import evaluate_model, evaluate_model_regression

st.set_page_config(page_title="Model Testing - ML Fairness Studio", layout="wide")

st.title("🧪 Model Testing & Evaluation")

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
model_name = st.session_state.get("model_name", "Model")
task_type = st.session_state.get("task_type", "classification")
is_regression = task_type == "regression"

st.markdown(f"**Model:** {model_name} | **Test samples:** {len(y_test):,} | **Task:** {task_type.capitalize()}")

# ── Run Evaluation ────────────────────────────────────────────────────────────
run_eval = st.button("▶️ Run Test Set Evaluation", type="primary", use_container_width=True)

existing_results = st.session_state.get("test_results", {})
if run_eval or existing_results:
    if run_eval:
        with st.spinner("Evaluating model on test set..."):
            try:
                if is_regression:
                    results = evaluate_model_regression(model, X_test, y_test)
                else:
                    results = evaluate_model(model, X_test, y_test)
                st.session_state["test_results"] = results
                st.session_state["y_pred"] = results["y_pred"]
                st.session_state["y_proba"] = results.get("y_proba")
                existing_results = results
                st.success("✅ Evaluation complete!")
            except Exception as e:
                st.error(f"❌ Evaluation failed: {e}")
                import traceback
                st.code(traceback.format_exc())
                st.stop()

    results = existing_results

    # ═════════════════════════════════════════════════════════════════════════
    # REGRESSION RESULTS
    # ═════════════════════════════════════════════════════════════════════════
    if is_regression:
        mae = results.get("mae", 0)
        mse = results.get("mse", 0)
        rmse = results.get("rmse", 0)
        r2 = results.get("r2", 0)
        mape = results.get("mape")
        y_pred = np.asarray(results.get("y_pred", []))
        residuals = np.asarray(results.get("residuals", []))

        # KPI Cards
        kc1, kc2, kc3, kc4, kc5 = st.columns(5)
        with kc1:
            st.metric("R²", f"{r2:.4f}")
        with kc2:
            st.metric("RMSE", f"{rmse:.4f}")
        with kc3:
            st.metric("MAE", f"{mae:.4f}")
        with kc4:
            st.metric("MSE", f"{mse:.4f}")
        with kc5:
            st.metric("MAPE (%)", f"{mape:.2f}" if mape is not None else "N/A")

        st.markdown("---")

        tab_scatter, tab_residuals, tab_stats = st.tabs([
            "📈 Actual vs Predicted",
            "📉 Residuals",
            "📋 Statistics",
        ])

        with tab_scatter:
            y_actual = np.asarray(results.get("scatter", {}).get("actual", y_test))
            y_predicted = np.asarray(results.get("scatter", {}).get("predicted", y_pred))

            fig_scatter = go.Figure()
            fig_scatter.add_trace(go.Scatter(
                x=y_actual, y=y_predicted,
                mode="markers",
                marker=dict(color="royalblue", opacity=0.5, size=5),
                name="Predictions",
            ))
            _min_val = min(y_actual.min(), y_predicted.min())
            _max_val = max(y_actual.max(), y_predicted.max())
            fig_scatter.add_trace(go.Scatter(
                x=[_min_val, _max_val], y=[_min_val, _max_val],
                mode="lines",
                name="Perfect fit",
                line=dict(color="red", dash="dash"),
            ))
            fig_scatter.update_layout(
                title=f"Actual vs Predicted (R² = {r2:.4f})",
                xaxis_title="Actual",
                yaxis_title="Predicted",
                height=480,
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        with tab_residuals:
            col_r1, col_r2 = st.columns(2)
            with col_r1:
                fig_res_hist = px.histogram(
                    residuals, nbins=40,
                    title="Residuals Distribution",
                    labels={"value": "Residual", "count": "Frequency"},
                )
                fig_res_hist.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Zero")
                fig_res_hist.update_layout(height=380, showlegend=False)
                st.plotly_chart(fig_res_hist, use_container_width=True)

            with col_r2:
                fig_res_pred = go.Figure()
                fig_res_pred.add_trace(go.Scatter(
                    x=y_pred, y=residuals,
                    mode="markers",
                    marker=dict(color="darkorange", opacity=0.5, size=5),
                    name="Residuals",
                ))
                fig_res_pred.add_hline(y=0, line_dash="dash", line_color="red")
                fig_res_pred.update_layout(
                    title="Residuals vs Predicted",
                    xaxis_title="Predicted",
                    yaxis_title="Residual",
                    height=380,
                )
                st.plotly_chart(fig_res_pred, use_container_width=True)

            # Residual stats
            res_stats = {
                "Mean residual": f"{residuals.mean():.4f}",
                "Std residual": f"{residuals.std():.4f}",
                "Max residual": f"{residuals.max():.4f}",
                "Min residual": f"{residuals.min():.4f}",
                "% within ±RMSE": f"{(np.abs(residuals) <= rmse).mean() * 100:.1f}%",
            }
            st.dataframe(
                pd.DataFrame(list(res_stats.items()), columns=["Metric", "Value"]),
                use_container_width=True, hide_index=True
            )

        with tab_stats:
            stats_df = pd.DataFrame([
                {"Metric": "R²", "Value": f"{r2:.6f}",
                 "Interpretation": "1.0 = perfect, 0.0 = no better than mean, <0 = worse than mean"},
                {"Metric": "RMSE", "Value": f"{rmse:.6f}",
                 "Interpretation": "Lower is better. In the same units as the target."},
                {"Metric": "MAE", "Value": f"{mae:.6f}",
                 "Interpretation": "Lower is better. Average absolute error."},
                {"Metric": "MSE", "Value": f"{mse:.6f}",
                 "Interpretation": "Lower is better. Penalizes large errors more than MAE."},
                {"Metric": "MAPE (%)", "Value": f"{mape:.2f}" if mape is not None else "N/A",
                 "Interpretation": "Lower is better. Mean absolute percentage error."},
            ])
            st.dataframe(stats_df, use_container_width=True, hide_index=True)

            transformer_info = st.session_state.get("target_transformer")
            if transformer_info is not None:
                transform_name = transformer_info[0]
                st.info(
                    f"Note: The target was transformed using **{transform_name}** during preprocessing. "
                    "The metrics above are computed in the transformed space. "
                    "To get metrics in the original scale, inverse-transform predictions manually."
                )

    # ═════════════════════════════════════════════════════════════════════════
    # CLASSIFICATION RESULTS
    # ═════════════════════════════════════════════════════════════════════════
    else:
        report = results.get("report", {})
        cm = results.get("confusion_matrix")
        classes = results.get("classes", [])
        roc_auc = results.get("roc_auc")
        roc_data = results.get("roc_curve")
        pr_data = results.get("pr_curve")
        y_pred = results.get("y_pred")
        y_proba = results.get("y_proba")

        acc = report.get("accuracy", 0)
        macro = report.get("macro avg", {})
        macro_f1 = macro.get("f1-score", 0)

        kc1, kc2, kc3 = st.columns(3)
        with kc1:
            st.metric("Accuracy", f"{acc:.4f}")
        with kc2:
            st.metric("AUC-ROC", f"{roc_auc:.4f}" if roc_auc else "N/A")
        with kc3:
            st.metric("Macro F1", f"{macro_f1:.4f}")

        st.markdown("---")

        tab1, tab2, tab3, tab4 = st.tabs([
            "🔢 Confusion Matrix",
            "📋 Classification Report",
            "📈 ROC Curves",
            "📉 Additional Metrics",
        ])

        with tab1:
            if cm is not None and len(classes) > 0:
                col_a, col_b = st.columns(2)

                with col_a:
                    cm_df = pd.DataFrame(
                        cm,
                        index=[f"Actual: {c}" for c in classes],
                        columns=[f"Pred: {c}" for c in classes],
                    )
                    fig_cm = px.imshow(
                        cm_df, text_auto=True,
                        color_continuous_scale="Blues",
                        title="Confusion Matrix (Counts)", aspect="auto",
                    )
                    fig_cm.update_layout(height=420)
                    st.plotly_chart(fig_cm, use_container_width=True)

                with col_b:
                    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
                    cm_norm_df = pd.DataFrame(
                        np.round(cm_norm, 3),
                        index=[f"Actual: {c}" for c in classes],
                        columns=[f"Pred: {c}" for c in classes],
                    )
                    fig_cmn = px.imshow(
                        cm_norm_df, text_auto=True,
                        color_continuous_scale="RdYlGn",
                        title="Confusion Matrix (Normalized)",
                        zmin=0, zmax=1, aspect="auto",
                    )
                    fig_cmn.update_layout(height=420)
                    st.plotly_chart(fig_cmn, use_container_width=True)

                st.markdown("#### Per-Class Stats from Confusion Matrix")
                tp_list, fp_list, fn_list, tn_list = [], [], [], []
                for i in range(len(classes)):
                    tp = cm[i, i]
                    fp = cm[:, i].sum() - tp
                    fn = cm[i, :].sum() - tp
                    tn = cm.sum() - tp - fp - fn
                    tp_list.append(tp); fp_list.append(fp)
                    fn_list.append(fn); tn_list.append(tn)
                cm_stats = pd.DataFrame({
                    "Class": classes, "TP": tp_list, "FP": fp_list,
                    "FN": fn_list, "TN": tn_list,
                })
                st.dataframe(cm_stats, use_container_width=True, hide_index=True)
            else:
                st.warning("Confusion matrix not available.")

        with tab2:
            st.markdown("#### Classification Report")
            class_rows = []
            for cls, vals in report.items():
                if isinstance(vals, dict) and cls not in ("accuracy",):
                    class_rows.append({
                        "Class": str(cls),
                        "Precision": vals.get("precision", 0),
                        "Recall": vals.get("recall", 0),
                        "F1-Score": vals.get("f1-score", 0),
                        "Support": int(vals.get("support", 0)),
                    })
            if class_rows:
                report_df = pd.DataFrame(class_rows)
                st.dataframe(report_df, use_container_width=True, hide_index=True)

                class_only = report_df[~report_df["Class"].isin(["macro avg", "weighted avg"])]
                if len(class_only) > 0:
                    fig_rep = go.Figure()
                    for metric in ["Precision", "Recall", "F1-Score"]:
                        fig_rep.add_trace(go.Bar(name=metric, x=class_only["Class"], y=class_only[metric]))
                    fig_rep.update_layout(
                        barmode="group", title="Per-Class Precision / Recall / F1",
                        yaxis=dict(range=[0, 1.05]), height=400,
                    )
                    st.plotly_chart(fig_rep, use_container_width=True)
            st.markdown("#### Full Text Report")
            st.code(results.get("report_text", "Not available"))

        with tab3:
            if roc_data is not None and roc_auc is not None:
                fpr = roc_data["fpr"]
                tpr = roc_data["tpr"]
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(
                    x=fpr, y=tpr, mode="lines",
                    name=f"ROC Curve (AUC = {roc_auc:.4f})",
                    line=dict(color="royalblue", width=2),
                ))
                fig_roc.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1], mode="lines",
                    name="Random Classifier", line=dict(color="gray", dash="dash"),
                ))
                fig_roc.update_layout(
                    title=f"ROC Curve — AUC = {roc_auc:.4f}",
                    xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
                    height=480, xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1.02]),
                )
                st.plotly_chart(fig_roc, use_container_width=True)
            else:
                st.info("ROC curve not available (requires probability estimates or binary classification).")

        with tab4:
            col_pr, col_thresh = st.columns(2)
            with col_pr:
                if pr_data is not None:
                    precision_arr = pr_data["precision"]
                    recall_arr = pr_data["recall"]
                    avg_prec = pr_data.get("average_precision", 0)
                    fig_pr = go.Figure()
                    fig_pr.add_trace(go.Scatter(
                        x=recall_arr, y=precision_arr, mode="lines",
                        name=f"PR Curve (AP = {avg_prec:.4f})",
                        line=dict(color="darkorange", width=2),
                        fill="tozeroy", fillcolor="rgba(255,165,0,0.1)",
                    ))
                    fig_pr.update_layout(
                        title=f"Precision-Recall Curve (AP = {avg_prec:.4f})",
                        xaxis_title="Recall", yaxis_title="Precision",
                        height=450, xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1.05]),
                    )
                    st.plotly_chart(fig_pr, use_container_width=True)
                else:
                    st.info("Precision-Recall curve not available.")

            with col_thresh:
                if y_proba is not None and len(classes) == 2:
                    st.markdown("#### Threshold Analysis")
                    threshold = st.slider("Decision threshold", 0.01, 0.99, 0.50, 0.01)
                    y_scores = y_proba[:, 1]
                    y_pred_thresh = (y_scores >= threshold).astype(int)
                    pos_class = classes[1]
                    y_test_binary = (y_test == pos_class).astype(int)
                    tp = np.sum((y_test_binary == 1) & (y_pred_thresh == 1))
                    fp = np.sum((y_test_binary == 0) & (y_pred_thresh == 1))
                    fn = np.sum((y_test_binary == 1) & (y_pred_thresh == 0))
                    tn = np.sum((y_test_binary == 0) & (y_pred_thresh == 0))
                    prec_t = tp / (tp + fp) if (tp + fp) > 0 else 0
                    rec_t = tp / (tp + fn) if (tp + fn) > 0 else 0
                    f1_t = 2 * prec_t * rec_t / (prec_t + rec_t) if (prec_t + rec_t) > 0 else 0
                    acc_t = (tp + tn) / len(y_test)
                    mc1, mc2, mc3, mc4 = st.columns(4)
                    mc1.metric("Precision", f"{prec_t:.3f}")
                    mc2.metric("Recall", f"{rec_t:.3f}")
                    mc3.metric("F1-Score", f"{f1_t:.3f}")
                    mc4.metric("Accuracy", f"{acc_t:.3f}")

st.info("➡️ Next step: Navigate to **📊 Feature Importance** to understand model decisions.")
