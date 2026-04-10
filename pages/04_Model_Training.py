"""
Page 4: Model Training
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
import time
from sklearn.model_selection import cross_val_score, StratifiedKFold

from utils.ml_utils import (
    MODELS,
    MODEL_PARAMS,
    REGRESSION_MODELS,
    REGRESSION_MODEL_PARAMS,
    OPTIMIZATION_PARAM_GRIDS,
    REGRESSION_OPTIMIZATION_PARAM_GRIDS,
    get_model_instance,
    get_regression_model_instance,
    evaluate_model,
    run_model_optimization,
)
from utils.llm_utils import ask_ai_recommendation

st.set_page_config(page_title="Model Training - ML Fairness Studio", layout="wide")

st.title("🏋️ Model Training")

# ── Guard ─────────────────────────────────────────────────────────────────────
if st.session_state.get("X_train") is None:
    st.warning("⚠️ No preprocessed data found. Please complete **⚙️ Preprocessing** first.")
    st.stop()

X_train = st.session_state["X_train"]
X_test = st.session_state["X_test"]
y_train = np.asarray(st.session_state["y_train"])
y_test = np.asarray(st.session_state["y_test"])
feature_names = st.session_state.get("feature_names", [])
sample_weights = st.session_state.get("sample_weights")
protected_cols = st.session_state.get("protected_cols", [])
task_type = st.session_state.get("task_type", "classification")
is_regression = task_type == "regression"

# Select correct model and param registries
_model_registry = REGRESSION_MODELS if is_regression else MODELS
_param_registry = REGRESSION_MODEL_PARAMS if is_regression else MODEL_PARAMS
_get_model_fn = get_regression_model_instance if is_regression else get_model_instance

st.markdown(
    f"**Training:** {X_train.shape[0]:,} samples × {X_train.shape[1]} features | "
    f"**Test:** {X_test.shape[0]:,} samples | **Task:** {task_type.capitalize()}"
)

if sample_weights is not None:
    st.success("✅ Sample weights from Reweighing are available and will be applied during training.")

tab_manual, tab_optimize = st.tabs(["🏋️ Manual Training", "🔍 Optimize Model"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Manual Training
# ══════════════════════════════════════════════════════════════════════════════
with tab_manual:
    # ── Model Selection ───────────────────────────────────────────────────────
    st.markdown("## 1️⃣ Model Selection")

    model_names = list(_model_registry.keys())
    default_model = st.session_state.get("model_name", model_names[0])
    if default_model not in model_names:
        default_model = model_names[0]

    selected_model = st.selectbox(
        "Choose ML algorithm:",
        model_names,
        index=model_names.index(default_model),
        key="manual_model_select",
    )

    # Ask AI for model selection
    llm_cfg = st.session_state.get("llm_config", {})
    if llm_cfg.get("provider") and llm_cfg.get("model"):
        with st.expander("🤖 Ask AI: Which algorithm should I use?"):
            if st.button("Get AI recommendation", key="ai_model_btn"):
                with st.spinner("Asking AI..."):
                    rec = ask_ai_recommendation(
                        question=f"Which {task_type} algorithm is best suited for this dataset and task?",
                        options=model_names,
                        session_state=dict(st.session_state),
                        llm_config=llm_cfg,
                    )
                if rec.startswith("Error:"):
                    st.error(rec)
                else:
                    st.markdown(rec)
    else:
        st.caption("💡 Configure an LLM in the sidebar to get AI algorithm recommendations.")

    # ── Hyperparameter Configuration ──────────────────────────────────────────
    st.markdown("## 2️⃣ Hyperparameters")
    params = _param_registry.get(selected_model, {})
    user_params = {}

    if params:
        n_cols = min(3, len(params))
        cols = st.columns(n_cols)
        col_idx = 0
        for param_name, spec in params.items():
            with cols[col_idx % n_cols]:
                ptype = spec["type"]
                default = spec["default"]

                if ptype == "int":
                    user_params[param_name] = st.slider(
                        param_name,
                        min_value=spec["min"],
                        max_value=spec["max"],
                        value=default,
                        step=1,
                        key=f"manual_param_{param_name}",
                    )
                elif ptype == "int_none":
                    use_none = st.checkbox(f"Use None for {param_name}", value=False, key=f"manual_none_{param_name}")
                    if use_none:
                        user_params[param_name] = None
                    else:
                        user_params[param_name] = st.slider(
                            param_name,
                            min_value=spec["min"],
                            max_value=spec["max"],
                            value=default if default != -1 else spec["max"] // 2,
                            step=1,
                            key=f"manual_param_{param_name}",
                        )
                elif ptype == "float":
                    user_params[param_name] = st.slider(
                        param_name,
                        min_value=float(spec["min"]),
                        max_value=float(spec["max"]),
                        value=float(default),
                        step=float((spec["max"] - spec["min"]) / 100),
                        format="%.4f",
                        key=f"manual_param_{param_name}",
                    )
                elif ptype == "select":
                    options = spec["options"]
                    default_idx = options.index(default) if default in options else 0
                    user_params[param_name] = st.selectbox(
                        param_name,
                        options,
                        index=default_idx,
                        key=f"manual_param_{param_name}",
                    )
            col_idx += 1

        # Ask AI for hyperparameter advice
        if llm_cfg.get("provider") and llm_cfg.get("model"):
            with st.expander("🤖 Ask AI: How should I tune these hyperparameters?"):
                if st.button("Get AI advice on hyperparameters", key="ai_hparam_btn"):
                    with st.spinner("Asking AI..."):
                        param_options = [
                            f"{k} (current: {user_params.get(k, spec['default'])})"
                            for k, spec in params.items()
                        ]
                        rec = ask_ai_recommendation(
                            question=f"How should I tune the hyperparameters for {selected_model} on this dataset?",
                            options=param_options,
                            session_state=dict(st.session_state),
                            llm_config=llm_cfg,
                        )
                    if rec.startswith("Error:"):
                        st.error(rec)
                    else:
                        st.markdown(rec)
    else:
        st.info("No hyperparameters to configure for this model.")

    # ── Cross-Validation Settings ──────────────────────────────────────────────
    st.markdown("## 3️⃣ Cross-Validation")
    cv_col1, cv_col2 = st.columns(2)
    with cv_col1:
        cv_folds = st.slider("CV folds", 2, 10, 5, key="manual_cv_folds")
    with cv_col2:
        if is_regression:
            cv_scoring = st.selectbox(
                "Scoring metric",
                ["neg_mean_squared_error", "neg_mean_absolute_error", "r2",
                 "neg_root_mean_squared_error"],
                format_func=lambda x: {
                    "neg_mean_squared_error": "Negative MSE",
                    "neg_mean_absolute_error": "Negative MAE",
                    "r2": "R²",
                    "neg_root_mean_squared_error": "Negative RMSE",
                }.get(x, x),
                key="manual_cv_scoring",
            )
        else:
            cv_scoring = st.selectbox(
                "Scoring metric",
                ["accuracy", "f1_macro", "f1_weighted", "roc_auc", "precision_macro", "recall_macro"],
                key="manual_cv_scoring",
            )

    # ── Fairness Constraint (classification only) ─────────────────────────────
    fc_method = st.session_state.get("fairness_constraint_method")
    if not is_regression and fc_method and protected_cols:
        st.markdown("## 4️⃣ Fairness Constraint")
        st.info(f"Queued constraint: **{fc_method}** on `{st.session_state.get('fairness_constraint_attr', '')}`")
        apply_fc = st.checkbox("Apply fairness constraint during training", value=False)
        fc_eps = st.slider("Constraint tolerance (ε)", 0.01, 0.2, 0.05, 0.01) if apply_fc else 0.05
    else:
        apply_fc = False
        fc_eps = 0.05
        if is_regression and fc_method:
            st.info("Fairness constraints (ExponentiatedGradient/ThresholdOptimizer) are not applicable to regression tasks.")

    # ── Train Button ──────────────────────────────────────────────────────────
    st.markdown("---")
    if st.button("🚀 Train Model", type="primary", use_container_width=True, key="manual_train_btn"):
        with st.spinner(f"Training {selected_model}..."):
            progress = st.progress(0)
            log_placeholder = st.empty()

            try:
                log_placeholder.info("Instantiating model...")
                model = _get_model_fn(selected_model, user_params)
                progress.progress(15)

                log_placeholder.info(f"Running {cv_folds}-fold cross-validation...")
                t0 = time.time()

                cv_score_metric = cv_scoring
                if not is_regression and cv_scoring == "roc_auc" and len(np.unique(y_train)) > 2:
                    cv_score_metric = "roc_auc_ovr"

                cv_n_jobs = 1 if selected_model == "LightGBM" else -1

                from sklearn.model_selection import KFold
                _cv_splitter = (
                    KFold(n_splits=cv_folds, shuffle=True, random_state=42)
                    if is_regression
                    else StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
                )
                _fallback_scoring = "r2" if is_regression else "accuracy"

                try:
                    cv_scores = cross_val_score(
                        model, X_train, y_train,
                        cv=_cv_splitter,
                        scoring=cv_score_metric,
                        n_jobs=cv_n_jobs,
                    )
                except Exception as cv_err:
                    st.warning(f"CV with {cv_score_metric} failed, falling back to {_fallback_scoring}: {cv_err}")
                    cv_scores = cross_val_score(
                        model, X_train, y_train,
                        cv=_cv_splitter,
                        scoring=_fallback_scoring,
                        n_jobs=cv_n_jobs,
                    )
                    cv_score_metric = _fallback_scoring

                cv_time = time.time() - t0
                progress.progress(55)

                log_placeholder.info("Training on full training set...")
                t1 = time.time()

                if sample_weights is not None:
                    try:
                        model.fit(X_train, y_train, sample_weight=sample_weights)
                    except TypeError:
                        st.warning("⚠️ This model doesn't support sample_weight. Training without weights.")
                        model.fit(X_train, y_train)
                else:
                    model.fit(X_train, y_train)

                train_time = time.time() - t1
                progress.progress(75)

                if apply_fc and fc_method:
                    log_placeholder.info(f"Applying {fc_method}...")
                    sensitive_train = st.session_state.get("sensitive_train")
                    if sensitive_train is not None:
                        try:
                            from utils.fairness_utils import apply_fairness_constraint
                            fc_constraint = (
                                "demographic_parity"
                                if "Demographic" in fc_method or "ExponentiatedGradient" in fc_method
                                else "equalized_odds"
                            )
                            model = apply_fairness_constraint(
                                model, X_train, y_train, sensitive_train,
                                constraint=fc_constraint, eps=fc_eps
                            )
                            st.success(f"✅ {fc_method} applied successfully.")
                        except Exception as fc_err:
                            st.warning(f"⚠️ Fairness constraint failed: {fc_err}")

                progress.progress(85)

                train_pred = model.predict(X_train)
                train_acc = (train_pred == y_train).mean()
                progress.progress(95)

                st.session_state["model"] = model
                st.session_state["model_name"] = selected_model
                st.session_state["cv_results"] = {
                    "scores": cv_scores.tolist(),
                    "mean": float(cv_scores.mean()),
                    "std": float(cv_scores.std()),
                    "metric": cv_score_metric,
                    "folds": cv_folds,
                }
                st.session_state["test_results"] = {}
                st.session_state["fairness_results"] = {}
                st.session_state["y_pred"] = None
                st.session_state["y_proba"] = None

                progress.progress(100)
                log_placeholder.empty()

                st.success(f"✅ **{selected_model}** trained successfully!")

                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                with metric_col1:
                    st.metric("Training Accuracy", f"{train_acc:.4f}")
                with metric_col2:
                    st.metric(f"CV {cv_score_metric} (mean)", f"{cv_scores.mean():.4f}")
                with metric_col3:
                    st.metric("CV Std Dev", f"{cv_scores.std():.4f}")
                with metric_col4:
                    st.metric("Training Time", f"{train_time:.2f}s")

                st.markdown("#### Cross-Validation Scores")
                fig_cv = go.Figure()
                fig_cv.add_trace(go.Box(
                    y=cv_scores,
                    name=cv_score_metric,
                    boxpoints="all",
                    jitter=0.3,
                    pointpos=-1.8,
                    marker_color="royalblue",
                ))
                fig_cv.add_hline(y=cv_scores.mean(), line_dash="dash", line_color="red",
                                 annotation_text=f"Mean: {cv_scores.mean():.4f}")
                fig_cv.update_layout(
                    title=f"{cv_folds}-Fold CV Scores ({cv_score_metric})",
                    yaxis_title=cv_score_metric,
                    height=380,
                )
                st.plotly_chart(fig_cv, use_container_width=True)

                cv_df = pd.DataFrame({
                    "Fold": [f"Fold {i+1}" for i in range(len(cv_scores))],
                    cv_score_metric: cv_scores,
                })
                fig_cv_bar = px.bar(
                    cv_df,
                    x="Fold",
                    y=cv_score_metric,
                    title="Score per Fold",
                    color=cv_score_metric,
                    color_continuous_scale="Blues",
                    text=cv_df[cv_score_metric].round(4).astype(str),
                )
                fig_cv_bar.update_layout(height=320, showlegend=False)
                st.plotly_chart(fig_cv_bar, use_container_width=True)

                st.markdown("#### Model Summary")
                model_info = {
                    "Algorithm": selected_model,
                    "Training Samples": X_train.shape[0],
                    "Features": X_train.shape[1],
                    "Training Accuracy": f"{train_acc:.4f}",
                    f"CV {cv_score_metric} (mean ± std)": f"{cv_scores.mean():.4f} ± {cv_scores.std():.4f}",
                    "Training Time": f"{train_time:.3f}s",
                    "CV Time": f"{cv_time:.3f}s",
                }
                for k, v in user_params.items():
                    model_info[f"Param: {k}"] = str(v)

                info_df = pd.DataFrame(list(model_info.items()), columns=["Property", "Value"])
                st.dataframe(info_df, use_container_width=True, hide_index=True)

                st.info("➡️ Next: Navigate to **⚖️ Fairness Evaluation** or **🧪 Model Testing**.")

            except Exception as e:
                st.error(f"❌ Training failed: {e}")
                import traceback
                st.code(traceback.format_exc())
            finally:
                progress.empty()

    # Show existing model info if no button pressed
    if st.session_state.get("model") is not None and not st.session_state.get("_just_trained"):
        model_name = st.session_state.get("model_name", "Unknown")
        cv_res = st.session_state.get("cv_results", {})
        st.success(f"✅ Model already trained: **{model_name}**")
        if cv_res:
            st.metric(
                f"CV {cv_res.get('metric', 'score')} (mean ± std)",
                f"{cv_res.get('mean', 0):.4f} ± {cv_res.get('std', 0):.4f}",
            )
        st.info("You can retrain above with different settings.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Optimize Model
# ══════════════════════════════════════════════════════════════════════════════
with tab_optimize:
    st.markdown("## 🔍 Automated Model Optimization")
    st.markdown(
        "Automatically search for the best model and hyperparameters using "
        "**GridSearchCV** or **RandomizedSearchCV** across multiple algorithms."
    )

    opt_col1, opt_col2 = st.columns(2)

    with opt_col1:
        search_type = st.radio(
            "Search strategy",
            ["Random Search (faster)", "Grid Search (exhaustive)"],
            help="Random Search samples hyperparameter combinations randomly. "
                 "Grid Search tries every combination (slower but guaranteed to find the grid optimum).",
        )
        search_type_key = "random" if "Random" in search_type else "grid"

        if is_regression:
            opt_scoring = st.selectbox(
                "Optimization metric",
                ["r2", "neg_mean_squared_error", "neg_mean_absolute_error", "neg_root_mean_squared_error"],
                key="opt_scoring",
            )
        else:
            opt_scoring = st.selectbox(
                "Optimization metric",
                ["accuracy", "f1_macro", "f1_weighted", "roc_auc", "precision_macro", "recall_macro"],
                key="opt_scoring",
            )
        opt_cv = st.slider("CV folds", 2, 10, 5, key="opt_cv")

    with opt_col2:
        if search_type_key == "random":
            opt_n_iter = st.slider(
                "Iterations per model (RandomSearch)",
                5, 100, 20,
                help="Number of hyperparameter combinations to sample per model.",
            )
        else:
            opt_n_iter = 100  # unused for grid search

        _opt_grids = REGRESSION_OPTIMIZATION_PARAM_GRIDS if is_regression else OPTIMIZATION_PARAM_GRIDS
        available_models = list(_opt_grids.keys())
        models_to_optimize = st.multiselect(
            "Models to include",
            available_models,
            default=available_models,
            help="Choose which algorithms to include in the search.",
        )

    # Ask AI which models to try
    llm_cfg = st.session_state.get("llm_config", {})
    if llm_cfg.get("provider") and llm_cfg.get("model"):
        with st.expander("🤖 Ask AI: Which models should I optimize?"):
            if st.button("Get AI recommendation for optimization", key="ai_opt_models_btn"):
                with st.spinner("Asking AI..."):
                    rec = ask_ai_recommendation(
                        question="Which machine learning algorithms should I prioritize for hyperparameter optimization on this dataset?",
                        options=available_models,
                        session_state=dict(st.session_state),
                        llm_config=llm_cfg,
                    )
                if rec.startswith("Error:"):
                    st.error(rec)
                else:
                    st.markdown(rec)

    st.markdown("---")

    if not models_to_optimize:
        st.warning("Please select at least one model to optimize.")
    else:
        if st.button("🚀 Run Optimization", type="primary", use_container_width=True, key="opt_run_btn"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            total_models = len(models_to_optimize)

            def _progress_cb(current, total, name):
                if total > 0:
                    progress_bar.progress(min(current / total, 1.0))
                status_text.info(f"Searching {name}... ({current}/{total})")

            with st.spinner("Running optimization — this may take a few minutes..."):
                try:
                    opt_results = run_model_optimization(
                        X_train=X_train,
                        y_train=y_train,
                        models_to_try=models_to_optimize,
                        search_type=search_type_key,
                        scoring=opt_scoring,
                        cv=opt_cv,
                        n_iter=opt_n_iter,
                        sample_weights=sample_weights,
                        random_state=42,
                        progress_callback=_progress_cb,
                        task_type=task_type,
                    )
                    st.session_state["opt_results"] = opt_results
                except Exception as opt_err:
                    st.error(f"Optimization failed: {opt_err}")
                    import traceback
                    st.code(traceback.format_exc())
                    opt_results = []

            progress_bar.empty()
            status_text.empty()

            if opt_results:
                st.success("✅ Optimization complete!")

                # Summary table
                summary_rows = []
                for r in opt_results:
                    row = {
                        "Model": r["model_name"],
                        f"Best CV {opt_scoring}": f"{r['best_score']:.4f}" if not np.isnan(r["best_score"]) else "Error",
                    }
                    if "error" in r:
                        row["Notes"] = f"Error: {r['error'][:60]}"
                    else:
                        row["Notes"] = ""
                    for k, v in r["best_params"].items():
                        row[f"best_{k}"] = str(v)
                    summary_rows.append(row)

                summary_df = pd.DataFrame(summary_rows)
                st.markdown("### Results Summary")
                st.dataframe(summary_df, use_container_width=True, hide_index=True)

                # Bar chart of best scores
                valid_results = [r for r in opt_results if not np.isnan(r["best_score"])]
                if valid_results:
                    fig_opt = px.bar(
                        x=[r["model_name"] for r in valid_results],
                        y=[r["best_score"] for r in valid_results],
                        color=[r["best_score"] for r in valid_results],
                        color_continuous_scale="Viridis",
                        labels={"x": "Model", "y": f"Best CV {opt_scoring}"},
                        title=f"Best CV {opt_scoring} per Model",
                        text=[f"{r['best_score']:.4f}" for r in valid_results],
                    )
                    fig_opt.update_layout(height=400, showlegend=False)
                    st.plotly_chart(fig_opt, use_container_width=True)

                # Best model callout + one-click adopt
                best = opt_results[0]
                if not np.isnan(best["best_score"]):
                    st.markdown(f"### 🏆 Best Model: **{best['model_name']}**")
                    st.markdown(f"CV {opt_scoring}: **{best['best_score']:.4f}**")
                    st.markdown("**Best hyperparameters:**")
                    st.json(best["best_params"])

                    if st.button(
                        f"✅ Use {best['model_name']} with best params as current model",
                        type="primary",
                        key="adopt_best_model_btn",
                    ):
                        # Retrain best estimator on full training set
                        best_model = best["best_estimator"]
                        if best_model is not None:
                            if sample_weights is not None:
                                try:
                                    best_model.fit(X_train, y_train, sample_weight=sample_weights)
                                except TypeError:
                                    best_model.fit(X_train, y_train)
                            else:
                                best_model.fit(X_train, y_train)

                            st.session_state["model"] = best_model
                            st.session_state["model_name"] = best["model_name"]
                            st.session_state["cv_results"] = {
                                "scores": [best["best_score"]],
                                "mean": best["best_score"],
                                "std": 0.0,
                                "metric": opt_scoring,
                                "folds": opt_cv,
                            }
                            st.session_state["test_results"] = {}
                            st.session_state["fairness_results"] = {}
                            st.session_state["y_pred"] = None
                            st.session_state["y_proba"] = None
                            st.success(
                                f"✅ **{best['model_name']}** is now the active model. "
                                "Proceed to Fairness Evaluation or Model Testing."
                            )

        # Show previous optimization results if available
        elif "opt_results" in st.session_state and st.session_state["opt_results"]:
            opt_results = st.session_state["opt_results"]
            st.info("Showing results from the last optimization run. Press **Run Optimization** to redo.")

            summary_rows = []
            for r in opt_results:
                row = {
                    "Model": r["model_name"],
                    f"Best CV Score": f"{r['best_score']:.4f}" if not np.isnan(r["best_score"]) else "Error",
                }
                if "error" in r:
                    row["Notes"] = f"Error: {r['error'][:60]}"
                summary_rows.append(row)

            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

            best = opt_results[0]
            if not np.isnan(best["best_score"]):
                st.markdown(f"**Best model:** {best['model_name']} (CV score: {best['best_score']:.4f})")
                st.json(best["best_params"])
