"""
ML Fairness Studio - Main Dashboard
"""

# ── NumPy compatibility — must be the very first import ───────────────────────
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import utils.compat  # noqa: F401 — applies NumPy 2.0 patches before anything else

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.llm_utils import PROVIDER_MODELS, test_connection

st.set_page_config(
    page_title="ML Fairness Studio",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── NumPy version warning ─────────────────────────────────────────────────────
_np_major = int(np.__version__.split(".")[0])
if _np_major >= 2:
    st.warning(
        f"**NumPy {np.__version__} detected.** Several ML libraries (shap, scipy < 1.13, "
        "and others) ship compiled extensions built against NumPy 1.x and will fail with "
        "NumPy 2.x. If you see `numpy.core.multiarray failed to import` errors, run:\n\n"
        "```\npip install \"numpy>=1.24,<2.0\" --force-reinstall\n```\n\n"
        "then restart the Streamlit server. "
        "The app will continue in degraded mode for affected pages.",
        icon="⚠️",
    )

# ── Session state initialization ─────────────────────────────────────────────
_defaults = {
    "df": None,
    "target_col": None,
    "protected_cols": [],
    "numeric_cols": [],
    "categorical_cols": [],
    "X_train": None,
    "X_test": None,
    "y_train": None,
    "y_test": None,
    "X_train_raw": None,
    "X_test_raw": None,
    "preprocessor": None,
    "feature_names": [],
    "preprocessing_config": {},
    "sensitive_train": None,
    "sensitive_test": None,
    "sensitive_col_for_fairness": None,
    "model": None,
    "model_name": None,
    "cv_results": None,
    "y_pred": None,
    "y_proba": None,
    "test_results": {},
    "fairness_results": {},
    "sample_weights": None,
    "task_type": "classification",          # 'classification' or 'regression'
    "target_transformer": None,             # fitted scaler/transform for regression targets
    "regression_results": {},               # regression evaluation results
    "llm_config": {
        "provider": "",
        "model": "",
        "api_key": "",
        "ollama_host": "http://localhost:11434",
        "custom_base_url": "http://localhost:11434/v1",
        "custom_model": "",
    },
    "chat_history": [],
}

for key, default in _defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    # ── LLM Configuration ─────────────────────────────────────────────────────
    st.markdown("## 🤖 AI Assistant")

    _cfg = st.session_state["llm_config"]
    _provider_list = list(PROVIDER_MODELS.keys())
    _provider_opts = ["(not configured)"] + _provider_list
    _cur_provider = _cfg.get("provider", "")
    _cur_idx = _provider_opts.index(_cur_provider) if _cur_provider in _provider_opts else 0

    _sel_provider = st.selectbox(
        "Provider",
        _provider_opts,
        index=_cur_idx,
        key="sidebar_llm_provider",
    )

    if _sel_provider == "(not configured)":
        st.caption("Select a provider to enable AI recommendations throughout the pipeline.")
        _sel_model = ""
        _api_key = ""
        _ollama_host = _cfg.get("ollama_host", "http://localhost:11434")
        _custom_base_url = _cfg.get("custom_base_url", "http://localhost:11434/v1")
        _custom_model = ""
    else:
        _models_for_provider = PROVIDER_MODELS.get(_sel_provider, [])
        _is_custom = _sel_provider == "Custom (OpenAI-Compatible)"
        _saved_model = _cfg.get("model", "")
        _saved_custom = _cfg.get("custom_model", "")

        if not _is_custom and _models_for_provider:
            _dd_default = _saved_model if _saved_model in _models_for_provider else _models_for_provider[0]
            _model_dropdown = st.selectbox(
                "Model",
                _models_for_provider,
                index=_models_for_provider.index(_dd_default),
                key="sidebar_llm_model",
            )
            _custom_model = st.text_input(
                "Custom model name (optional)",
                value=_saved_custom,
                key="sidebar_llm_custom_model",
                placeholder="Leave blank to use dropdown",
            )
            _sel_model = _custom_model.strip() if _custom_model.strip() else _model_dropdown
        else:
            _custom_model = st.text_input(
                "Model name",
                value=_saved_custom or _saved_model,
                key="sidebar_llm_custom_model",
                placeholder="e.g. llama3.2",
            )
            _sel_model = _custom_model.strip()

        if _sel_provider != "Ollama (Local)":
            _api_key = st.text_input(
                "API Key",
                value=_cfg.get("api_key", ""),
                type="password",
                key="sidebar_llm_api_key",
            )
        else:
            _api_key = ""

        if _sel_provider == "Ollama (Local)":
            _ollama_host = st.text_input(
                "Ollama Host",
                value=_cfg.get("ollama_host", "http://localhost:11434"),
                key="sidebar_ollama_host",
            )
        else:
            _ollama_host = _cfg.get("ollama_host", "http://localhost:11434")

        if _is_custom:
            _custom_base_url = st.text_input(
                "Base URL",
                value=_cfg.get("custom_base_url", "http://localhost:11434/v1"),
                key="sidebar_custom_base_url",
            )
        else:
            _custom_base_url = _cfg.get("custom_base_url", "http://localhost:11434/v1")

        _sb_col1, _sb_col2 = st.columns(2)
        with _sb_col1:
            if st.button("Save", use_container_width=True, key="sidebar_save_llm"):
                st.session_state["llm_config"] = {
                    "provider": _sel_provider,
                    "model": _sel_model,
                    "api_key": _api_key,
                    "ollama_host": _ollama_host,
                    "custom_base_url": _custom_base_url,
                    "custom_model": _custom_model.strip() if _custom_model else "",
                }
                st.success("Saved!")
        with _sb_col2:
            if st.button("Test", use_container_width=True, key="sidebar_test_llm"):
                with st.spinner("Testing..."):
                    _ok, _msg = test_connection(
                        provider=_sel_provider,
                        model=_sel_model,
                        api_key=_api_key,
                        ollama_host=_ollama_host,
                        custom_base_url=_custom_base_url,
                    )
                if _ok:
                    st.success("Connected!")
                else:
                    st.error(_msg[:120])

        # Persist any config changes made by the widgets above
        st.session_state["llm_config"] = {
            "provider": _sel_provider,
            "model": _sel_model,
            "api_key": _api_key,
            "ollama_host": _ollama_host,
            "custom_base_url": _custom_base_url,
            "custom_model": _custom_model.strip() if _custom_model else "",
        }

    # ── Pipeline Progress ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 🗺️ Pipeline Progress")

    _task = st.session_state.get("task_type", "classification")
    _test_res = st.session_state.get("test_results", {})

    steps = [
        ("📂 Data Upload", st.session_state["df"] is not None),
        ("⚙️ Preprocessing", st.session_state["X_train"] is not None),
        ("🔍 Bias Detection", bool(st.session_state["protected_cols"])),
        ("🏋️ Model Training", st.session_state["model"] is not None),
        ("⚖️ Fairness Evaluation", bool(st.session_state["fairness_results"])),
        ("🧪 Model Testing", bool(_test_res)),
        ("📊 Feature Importance", st.session_state["model"] is not None),
        ("🤖 LLM Analysis", bool(st.session_state["chat_history"])),
        ("🔗 Causal Inference", st.session_state["df"] is not None),
        ("📓 Download Notebook", True),
    ]

    for step_name, completed in steps:
        icon = "✅" if completed else "⬜"
        st.markdown(f"{icon} {step_name}")

    st.markdown("---")
    st.markdown("### Quick Stats")
    if st.session_state["df"] is not None:
        _df = st.session_state["df"]
        st.metric("Rows", f"{_df.shape[0]:,}")
        st.metric("Columns", _df.shape[1])
    st.metric("Task", _task.capitalize())
    if st.session_state["model_name"]:
        st.metric("Model", st.session_state["model_name"])
    if _test_res:
        if _task == "regression":
            _r2 = _test_res.get("r2")
            if _r2 is not None:
                st.metric("Test R²", f"{_r2:.3f}")
        else:
            _acc = _test_res.get("report", {}).get("accuracy")
            if _acc is not None:
                st.metric("Test Accuracy", f"{_acc:.3f}")

    st.markdown("---")
    st.caption("ML Fairness Studio v1.0")


# ── Main Content ──────────────────────────────────────────────────────────────
st.title("🤖 ML Fairness Studio")
st.markdown("*A comprehensive platform for fair and responsible machine learning.*")

# ── Disclaimer ────────────────────────────────────────────────────────────────
with st.expander("📌 About this Application", expanded=False):
    st.info(
        """
        **ML Fairness Studio** was developed by **Sridhar Nerur** as part of a graduate-level
        course on responsible machine learning, with substantial assistance from
        [Claude](https://www.anthropic.com/claude) (Anthropic's AI assistant) — a collaboration
        that itself illustrates how humans and AI can work together productively.

        **Please keep the following in mind:**

        - This application is intended **for educational purposes only**. It is designed to help
          students and practitioners explore concepts such as algorithmic fairness, bias detection,
          causal inference, and model interpretability in an interactive, hands-on setting.
        - As with any AI-assisted tool, **outputs should be interpreted critically**. Always
          validate results against domain knowledge, established benchmarks, and sound statistical
          practice.
        - **Reproducibility matters.** Different preprocessing choices, random seeds, train/test
          splits, and hyperparameter settings can yield meaningfully different results. Treat every
          finding as a starting point for deeper investigation, not a definitive conclusion.
        - The inclusion of an LLM-powered analysis assistant is meant to augment — not replace —
          rigorous human judgment. AI-generated insights can be illuminating, but they can also
          be confidently wrong. Verify before you act.

        *Use this tool thoughtfully, question its outputs, and let it spark curiosity rather than
        shortcut understanding.*
        """
    )

# ── CASE 1: Model is trained → full dashboard ─────────────────────────────────
if st.session_state["model"] is not None and st.session_state["test_results"]:
    df = st.session_state["df"]
    test_results = st.session_state["test_results"]
    fairness_results = st.session_state.get("fairness_results", {})
    report = test_results.get("report", {})

    # KPI cards
    st.markdown("### 📈 Pipeline Summary")
    _task_type = st.session_state.get("task_type", "classification")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Dataset Size", f"{df.shape[0]:,}" if df is not None else "N/A")
    with c2:
        st.metric("Features", len(st.session_state.get("feature_names", [])))
    with c3:
        missing_pct = 0.0
        if df is not None:
            missing_pct = (df.isnull().sum().sum() / df.size * 100)
        st.metric("Missing Data", f"{missing_pct:.1f}%")
    with c4:
        if _task_type == "regression":
            _r2 = test_results.get("r2")
            st.metric("Test R²", f"{_r2:.3f}" if _r2 is not None else "N/A")
        else:
            acc = report.get("accuracy", None)
            st.metric("Test Accuracy", f"{acc:.3f}" if acc else "N/A")
    with c5:
        dp_diff = fairness_results.get("demographic_parity_difference", None)
        st.metric(
            "Dem. Parity Diff",
            f"{dp_diff:.3f}" if dp_diff is not None else "N/A",
            delta=None,
        )

    st.markdown("---")

    # Results overview
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### Confusion Matrix")
        cm = test_results.get("confusion_matrix")
        classes = test_results.get("classes", [])
        if cm is not None and len(classes) > 0:
            cm_df = pd.DataFrame(
                cm,
                index=[f"Actual: {c}" for c in classes],
                columns=[f"Pred: {c}" for c in classes],
            )
            fig_cm = px.imshow(
                cm_df,
                text_auto=True,
                color_continuous_scale="Blues",
                title="Confusion Matrix",
            )
            fig_cm.update_layout(height=350, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_cm, use_container_width=True)

    with col_right:
        st.markdown("#### Per-Class Metrics")
        class_metrics = []
        for cls, vals in report.items():
            if isinstance(vals, dict) and cls not in ("accuracy", "macro avg", "weighted avg"):
                class_metrics.append({
                    "Class": str(cls),
                    "Precision": vals.get("precision", 0),
                    "Recall": vals.get("recall", 0),
                    "F1-Score": vals.get("f1-score", 0),
                })
        if class_metrics:
            cm_df2 = pd.DataFrame(class_metrics)
            fig_cls = go.Figure()
            for metric in ["Precision", "Recall", "F1-Score"]:
                fig_cls.add_trace(go.Bar(
                    name=metric,
                    x=cm_df2["Class"],
                    y=cm_df2[metric],
                ))
            fig_cls.update_layout(
                barmode="group",
                title="Per-Class Metrics",
                height=350,
                margin=dict(l=10, r=10, t=40, b=10),
                yaxis=dict(range=[0, 1]),
            )
            st.plotly_chart(fig_cls, use_container_width=True)

    # Fairness overview
    if fairness_results:
        st.markdown("---")
        st.markdown("#### ⚖️ Fairness Metrics Overview")
        metric_names = {
            "demographic_parity_difference": "Dem. Parity Diff",
            "demographic_parity_ratio": "Dem. Parity Ratio",
            "equalized_odds_difference": "Eq. Odds Diff",
            "equal_opportunity_difference": "Eq. Opp Diff",
            "disparate_impact": "Disparate Impact",
        }
        fair_cols = st.columns(len(metric_names))
        for i, (key, label) in enumerate(metric_names.items()):
            val = fairness_results.get(key)
            with fair_cols[i]:
                if val is not None:
                    # Determine good/bad
                    if key in ("demographic_parity_ratio", "disparate_impact"):
                        delta_color = "normal" if val >= 0.8 else "inverse"
                    else:
                        delta_color = "normal" if abs(val) < 0.1 else "inverse"
                    st.metric(label, f"{val:.3f}")
                else:
                    st.metric(label, "N/A")


# ── CASE 2: Data loaded but no model ─────────────────────────────────────────
elif st.session_state["df"] is not None:
    df = st.session_state["df"]
    st.info("📊 Data loaded! Continue through the pipeline to train a model.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Dataset Overview")
        st.markdown(f"- **Rows:** {df.shape[0]:,}")
        st.markdown(f"- **Columns:** {df.shape[1]}")
        target = st.session_state.get("target_col")
        st.markdown(f"- **Target:** {target if target else 'Not set'}")
        protected = st.session_state.get("protected_cols", [])
        st.markdown(f"- **Protected attrs:** {', '.join(protected) if protected else 'Not set'}")

        # Missing values bar chart
        missing = df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        if len(missing) > 0:
            fig_miss = px.bar(
                x=missing.index,
                y=missing.values,
                labels={"x": "Column", "y": "Missing Count"},
                title="Missing Values per Column",
                color=missing.values,
                color_continuous_scale="Reds",
            )
            fig_miss.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig_miss, use_container_width=True)
        else:
            st.success("No missing values in dataset!")

    with col2:
        st.markdown("#### Data Types Distribution")
        dtypes = df.dtypes.astype(str).value_counts()
        fig_types = px.pie(
            values=dtypes.values,
            names=dtypes.index,
            title="Column Data Types",
            hole=0.4,
        )
        fig_types.update_layout(height=250)
        st.plotly_chart(fig_types, use_container_width=True)

        # Sample data
        st.markdown("#### Data Preview (first 5 rows)")
        st.dataframe(df.head(), use_container_width=True)


# ── CASE 3: Nothing loaded → welcome screen ──────────────────────────────────
else:
    st.markdown("""
    ## Welcome to ML Fairness Studio! 👋

    This platform helps you build, evaluate, and audit machine learning models
    for fairness and bias across protected demographic groups.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### 🚀 Key Features

        - **Data Upload**: Load CSV/Excel files or use built-in sample datasets
        - **Preprocessing**: Handle missing values, encode features, scale data
        - **Bias Detection**: Identify bias in protected attributes before training
        - **Model Training**: 9 ML algorithms with tunable hyperparameters
        - **Fairness Evaluation**: Demographic parity, equalized odds, and more
        - **Model Testing**: Confusion matrix, ROC curves, classification reports
        - **Feature Importance**: SHAP values, permutation importance, CIs & p-values
        - **LLM Analysis**: AI-powered insights via OpenAI, Claude, Ollama, and more
        - **Causal Inference**: Estimate causal effects with IPW, AIPW & propensity scores
        - **Download Notebook**: Export the full pipeline as a runnable Jupyter Notebook
        """)

    with col2:
        st.markdown("""
        ### 📋 Quick Start Guide

        1. **📂 Data Upload** → Load your dataset and configure target/protected columns
        2. **⚙️ Preprocessing** → Handle missing values, encode and scale features
        3. **🔍 Bias Detection** → Analyze bias in protected attributes
        4. **🏋️ Model Training** → Select and train your ML model
        5. **⚖️ Fairness Evaluation** → Measure fairness across demographic groups
        6. **🧪 Model Testing** → Evaluate on held-out test set
        7. **📊 Feature Importance** → Understand which features drive predictions
        8. **🤖 LLM Analysis** → Get AI-powered insights and recommendations
        """)

    st.markdown("---")
    st.markdown("### 🎯 Get Started")
    st.markdown("Navigate to **📂 Data Upload** in the sidebar to begin, or explore the pages above.")

    # Navigation cards
    pages = [
        ("📂 Data Upload", "Load and configure your dataset", "pages/01_Data_Upload.py"),
        ("⚙️ Preprocessing", "Handle missing values and encode features", "pages/02_Preprocessing.py"),
        ("🔍 Bias Detection", "Analyze bias in protected attributes", "pages/03_Bias_Detection.py"),
        ("🏋️ Model Training", "Train ML models with hyperparameter tuning", "pages/04_Model_Training.py"),
        ("⚖️ Fairness Evaluation", "Measure demographic parity and fairness metrics", "pages/05_Fairness_Evaluation.py"),
        ("🧪 Model Testing", "Evaluate on test set with detailed metrics", "pages/06_Model_Testing.py"),
        ("📊 Feature Importance", "SHAP values, permutation importance, CIs & p-values", "pages/07_Feature_Importance.py"),
        ("🤖 LLM Analysis", "AI-powered insights and recommendations", "pages/08_LLM_Analysis.py"),
        ("🔗 Causal Inference", "Estimate causal effects with IPW, AIPW & propensity scores", "pages/09_Causal_Inference.py"),
        ("📓 Download Notebook", "Export the full pipeline as a Jupyter Notebook", "pages/10_Download_Notebook.py"),
    ]

    col_count = 4
    rows = [pages[i:i+col_count] for i in range(0, len(pages), col_count)]
    for row in rows:
        cols = st.columns(col_count)
        for i, (title, desc, _) in enumerate(row):
            with cols[i]:
                st.info(f"**{title}**\n\n{desc}")
