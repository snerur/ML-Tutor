"""
Page 1: Data Upload and Configuration
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

from utils.data_utils import (
    load_data,
    get_column_types,
    analyze_missing,
    compute_class_distribution,
    suggest_protected_attributes,
    generate_adult_income_dataset,
    generate_credit_risk_dataset,
    generate_compas_dataset,
    detect_task_type,
)
from utils.llm_utils import ask_ai_recommendation

st.set_page_config(page_title="Data Upload - ML Fairness Studio", layout="wide")

st.title("📂 Data Upload & Configuration")
st.markdown("Load your dataset and configure the target variable and protected attributes.")

# ── Source selection ──────────────────────────────────────────────────────────
source = st.radio(
    "Data source:",
    [
        "Upload file",
        "Sample: Adult Income Dataset",
        "Sample: Credit Risk Dataset",
        "Sample: COMPAS Recidivism Dataset",
    ],
    horizontal=True,
)

df = None

if source == "Upload file":
    uploaded = st.file_uploader(
        "Upload a CSV or Excel file",
        type=["csv", "xlsx", "xls"],
        help="Supported formats: CSV, Excel (.xlsx, .xls)",
    )
    if uploaded is not None:
        with st.spinner("Loading data..."):
            try:
                df = load_data(uploaded)
                st.success(f"✅ Loaded **{uploaded.name}** — {df.shape[0]:,} rows × {df.shape[1]} columns")
            except Exception as e:
                st.error(f"Failed to load file: {e}")

elif source == "Sample: Adult Income Dataset":
    n = st.slider("Number of samples", 500, 5000, 1000, 100)
    with st.spinner("Generating synthetic Adult Income dataset..."):
        df = generate_adult_income_dataset(n_samples=n)
    st.success(f"✅ Loaded synthetic Adult Income dataset — {df.shape[0]:,} rows × {df.shape[1]} columns")

elif source == "Sample: Credit Risk Dataset":
    n = st.slider("Number of samples", 500, 5000, 1000, 100)
    with st.spinner("Generating synthetic Credit Risk dataset..."):
        df = generate_credit_risk_dataset(n_samples=n)
    st.success(f"✅ Loaded synthetic Credit Risk dataset — {df.shape[0]:,} rows × {df.shape[1]} columns")

elif source == "Sample: COMPAS Recidivism Dataset":
    n = st.slider("Number of samples", 500, 7000, 2000, 100)
    with st.spinner("Generating synthetic COMPAS dataset..."):
        df = generate_compas_dataset(n_samples=n)
    st.success(f"✅ Loaded synthetic COMPAS dataset — {df.shape[0]:,} rows × {df.shape[1]} columns")
    st.info(
        "ℹ️ **About COMPAS**: This dataset mimics the ProPublica COMPAS recidivism dataset "
        "(Broward County, FL). The `decile_score` column represents the algorithmic risk score. "
        "The `two_year_recid` column is the actual two-year recidivism outcome (the prediction target). "
        "Protected attributes: **race**, **sex**, **age**. "
        "This dataset is widely used to study algorithmic bias in criminal justice."
    )

# ── Dataset exploration (only when data is available) ─────────────────────────
if df is not None:
    col_types = get_column_types(df)
    numeric_cols = col_types["numeric"]
    categorical_cols = col_types["categorical"]
    datetime_cols = col_types["datetime"]

    tab1, tab2, tab3, tab4 = st.tabs(["📋 Preview", "📊 Statistics", "❓ Missing Values", "📈 Distributions"])

    # ── Tab 1: Preview ────────────────────────────────────────────────────────
    with tab1:
        st.markdown(f"**Shape:** {df.shape[0]:,} rows × {df.shape[1]} columns")
        col_info = pd.DataFrame({
            "Column": df.columns,
            "Type": df.dtypes.astype(str).values,
            "Unique": df.nunique().values,
            "Missing": df.isnull().sum().values,
            "Missing %": (df.isnull().sum() / len(df) * 100).round(2).values,
        })
        st.dataframe(col_info, use_container_width=True, height=250)
        st.markdown("**First 10 rows:**")
        st.dataframe(df.head(10), use_container_width=True)

    # ── Tab 2: Statistics ─────────────────────────────────────────────────────
    with tab2:
        st.markdown("#### Descriptive Statistics (Numeric)")
        if numeric_cols:
            st.dataframe(df[numeric_cols].describe().round(4), use_container_width=True)
        else:
            st.info("No numeric columns found.")

        if len(numeric_cols) >= 2:
            st.markdown("#### Correlation Heatmap")
            corr = df[numeric_cols].corr().round(3)
            fig_corr = px.imshow(
                corr,
                text_auto=True,
                color_continuous_scale="RdBu_r",
                zmin=-1, zmax=1,
                title="Correlation Matrix",
                aspect="auto",
            )
            fig_corr.update_layout(height=500)
            st.plotly_chart(fig_corr, use_container_width=True)

    # ── Tab 3: Missing Values ─────────────────────────────────────────────────
    with tab3:
        missing_df = analyze_missing(df)
        if len(missing_df) == 0:
            st.success("✅ No missing values found in the dataset!")
        else:
            st.markdown(f"**{len(missing_df)} columns** have missing values.")
            st.dataframe(missing_df, use_container_width=True)

            fig_miss = px.bar(
                missing_df,
                x=missing_df.index,
                y="Percentage",
                color="Percentage",
                color_continuous_scale="Reds",
                title="Missing Value Percentage per Column",
                labels={"x": "Column", "Percentage": "Missing (%)"},
            )
            fig_miss.add_hline(y=5, line_dash="dash", line_color="orange",
                               annotation_text="5% threshold")
            fig_miss.add_hline(y=20, line_dash="dash", line_color="red",
                               annotation_text="20% threshold")
            fig_miss.update_layout(height=400)
            st.plotly_chart(fig_miss, use_container_width=True)

            # Missing values heatmap for small datasets
            if df.shape[0] <= 500:
                st.markdown("#### Missing Value Pattern Heatmap")
                miss_matrix = df.isnull().astype(int)
                fig_heat = px.imshow(
                    miss_matrix.T,
                    color_continuous_scale=["white", "red"],
                    title="Missing Value Pattern (red = missing)",
                    aspect="auto",
                )
                fig_heat.update_layout(height=400)
                st.plotly_chart(fig_heat, use_container_width=True)

    # ── Tab 4: Distributions ──────────────────────────────────────────────────
    with tab4:
        if numeric_cols:
            st.markdown("#### Numeric Distributions")
            cols_to_show = st.multiselect(
                "Select numeric columns to visualize:",
                numeric_cols,
                default=numeric_cols[:min(4, len(numeric_cols))],
            )
            if cols_to_show:
                for i in range(0, len(cols_to_show), 2):
                    c1, c2 = st.columns(2)
                    for j, col in enumerate(cols_to_show[i:i+2]):
                        target_col_for_color = (
                            st.session_state.get("target_col")
                            if st.session_state.get("target_col") in df.columns
                            else None
                        )
                        with [c1, c2][j]:
                            fig = make_subplots(rows=1, cols=2,
                                                subplot_titles=("Histogram", "Box Plot"))
                            if target_col_for_color and df[target_col_for_color].nunique() <= 10:
                                for grp in df[target_col_for_color].dropna().unique():
                                    vals = df[df[target_col_for_color] == grp][col].dropna()
                                    fig.add_trace(go.Histogram(x=vals, name=str(grp),
                                                               opacity=0.7), row=1, col=1)
                                    fig.add_trace(go.Box(y=vals, name=str(grp)), row=1, col=2)
                            else:
                                vals = df[col].dropna()
                                fig.add_trace(go.Histogram(x=vals, name=col,
                                                           marker_color="#636EFA"), row=1, col=1)
                                fig.add_trace(go.Box(y=vals, name=col,
                                                     marker_color="#636EFA"), row=1, col=2)
                            fig.update_layout(title=col, height=300, showlegend=False,
                                              margin=dict(l=5, r=5, t=30, b=5))
                            st.plotly_chart(fig, use_container_width=True)

        if categorical_cols:
            st.markdown("#### Categorical Distributions")
            cat_to_show = st.multiselect(
                "Select categorical columns to visualize:",
                categorical_cols,
                default=categorical_cols[:min(4, len(categorical_cols))],
            )
            for i in range(0, len(cat_to_show), 2):
                c1, c2 = st.columns(2)
                for j, col in enumerate(cat_to_show[i:i+2]):
                    with [c1, c2][j]:
                        vc = df[col].value_counts().head(20)
                        fig = px.bar(
                            x=vc.index.astype(str),
                            y=vc.values,
                            title=f"{col} (top 20)",
                            labels={"x": col, "y": "Count"},
                            color=vc.values,
                            color_continuous_scale="Blues",
                        )
                        fig.update_layout(height=300, showlegend=False,
                                          margin=dict(l=5, r=5, t=30, b=5))
                        st.plotly_chart(fig, use_container_width=True)

    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 🎯 Target Variable & Protected Attributes")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### Target Variable")
        target_default = (
            st.session_state.get("target_col")
            if st.session_state.get("target_col") in df.columns
            else df.columns[-1]
        )
        target_col = st.selectbox(
            "Select the target (outcome) variable:",
            df.columns.tolist(),
            index=df.columns.tolist().index(target_default) if target_default in df.columns else len(df.columns) - 1,
        )

        # Auto-detect task type
        detected_type = detect_task_type(df, target_col)
        _type_color = "green" if detected_type == "classification" else "blue"
        st.markdown(
            f"**Auto-detected task:** "
            f"<span style='color:{_type_color};font-weight:bold'>{detected_type.upper()}</span>",
            unsafe_allow_html=True,
        )

        task_type_override = st.radio(
            "Task type (override if needed):",
            ["Auto-detect", "Classification", "Regression"],
            index=0,
            horizontal=True,
            help="Auto-detect is correct in most cases. Override only if the detection is wrong.",
        )
        if task_type_override == "Classification":
            final_task_type = "classification"
        elif task_type_override == "Regression":
            final_task_type = "regression"
        else:
            final_task_type = detected_type

        if final_task_type == "regression":
            st.info(
                "Regression task detected. The preprocessing page will offer target transformation options. "
                "Regression metrics (RMSE, MAE, R²) will be used for evaluation."
            )
            # Show target distribution for regression
            target_series = df[target_col].dropna()
            fig_target = px.histogram(
                target_series,
                nbins=40,
                title=f"Target Distribution: {target_col}",
                labels={"value": target_col, "count": "Frequency"},
            )
            fig_target.update_layout(height=280, showlegend=False)
            st.plotly_chart(fig_target, use_container_width=True)
            st.markdown(
                f"Mean: **{target_series.mean():.4f}** | "
                f"Std: **{target_series.std():.4f}** | "
                f"Min: **{target_series.min():.4f}** | "
                f"Max: **{target_series.max():.4f}**"
            )
        else:
            # Class distribution for classification
            cls_dist = compute_class_distribution(df, target_col)
            st.dataframe(cls_dist, use_container_width=True)
            fig_pie = px.pie(
                values=cls_dist["Count"],
                names=cls_dist.index.astype(str),
                title=f"Class Distribution: {target_col}",
                hole=0.35,
            )
            fig_pie.update_layout(height=300)
            st.plotly_chart(fig_pie, use_container_width=True)

    with col_right:
        st.markdown("### Protected Attributes")
        suggestions = suggest_protected_attributes(df)
        # Remove target from options
        all_cols = [c for c in df.columns if c != target_col]

        default_protected = [
            c for c in (st.session_state.get("protected_cols") or suggestions)
            if c in all_cols
        ]

        protected_cols = st.multiselect(
            "Select protected demographic attributes:",
            all_cols,
            default=default_protected,
            help="These are columns like gender, race, age, etc. that should not be the basis for discrimination.",
        )
        if suggestions:
            st.info(f"💡 Suggested protected attributes: {', '.join(suggestions)}")
        else:
            st.info("💡 No protected attribute suggestions found. Look for columns related to gender, race, age, etc.")

        # Show distribution of protected attributes if selected
        for pc in protected_cols[:2]:  # Show first 2
            vc = df[pc].value_counts()
            fig_pc = px.bar(
                x=vc.index.astype(str),
                y=vc.values,
                title=f"Distribution: {pc}",
                labels={"x": pc, "y": "Count"},
                color=vc.values,
                color_continuous_scale="Viridis",
            )
            fig_pc.update_layout(height=200, showlegend=False, margin=dict(l=5, r=5, t=30, b=5))
            st.plotly_chart(fig_pc, use_container_width=True)

    # ── Ask AI about protected attributes ─────────────────────────────────────
    llm_cfg = st.session_state.get("llm_config", {})
    if llm_cfg.get("provider") and llm_cfg.get("model"):
        with st.expander("🤖 Ask AI: Which columns should be protected attributes?"):
            if st.button("Get AI recommendation", key="ai_protected_btn"):
                with st.spinner("Asking AI..."):
                    all_options = [c for c in df.columns if c != target_col]
                    rec = ask_ai_recommendation(
                        question="Which columns in this dataset should be treated as protected demographic attributes for fairness analysis?",
                        options=all_options,
                        session_state=dict(st.session_state),
                        llm_config=llm_cfg,
                    )
                if rec.startswith("Error:"):
                    st.error(rec)
                else:
                    st.markdown(rec)
    else:
        st.caption("💡 Configure an LLM on the home page to get AI recommendations for protected attributes.")

    st.markdown("---")

    # Feature type override
    with st.expander("🔧 Advanced: Review & Adjust Column Types"):
        st.markdown("Adjust which columns are treated as numeric vs. categorical.")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Numeric columns:**")
            selected_numeric = st.multiselect(
                "Numeric", all_cols,
                default=[c for c in numeric_cols if c != target_col],
                key="numeric_override",
            )
        with col_b:
            st.markdown("**Categorical columns:**")
            selected_cat = st.multiselect(
                "Categorical", all_cols,
                default=[c for c in categorical_cols if c != target_col],
                key="cat_override",
            )
        numeric_cols = selected_numeric
        categorical_cols = selected_cat

    # Save configuration
    st.markdown("### 💾 Save Configuration")
    if st.button("✅ Save Configuration & Proceed", type="primary", use_container_width=True):
        st.session_state["df"] = df
        st.session_state["target_col"] = target_col
        st.session_state["protected_cols"] = protected_cols
        st.session_state["numeric_cols"] = [c for c in numeric_cols if c != target_col]
        st.session_state["categorical_cols"] = [c for c in categorical_cols if c != target_col]
        st.session_state["task_type"] = final_task_type
        # Reset downstream state when task type changes
        st.session_state["model"] = None
        st.session_state["test_results"] = {}
        st.session_state["regression_results"] = {}
        st.success(f"""
        ✅ Configuration saved!
        - **Target:** {target_col}
        - **Task type:** {final_task_type}
        - **Protected attributes:** {', '.join(protected_cols) if protected_cols else 'None'}
        - **Numeric features:** {len([c for c in numeric_cols if c != target_col])}
        - **Categorical features:** {len([c for c in categorical_cols if c != target_col])}
        """)
        st.info("➡️ Next step: Navigate to **⚙️ Preprocessing** to prepare your data for training.")

elif st.session_state.get("df") is not None:
    # Data already loaded in session, show summary
    df_ss = st.session_state["df"]
    st.info(f"📊 Dataset already loaded: {df_ss.shape[0]:,} rows × {df_ss.shape[1]} columns. Upload a new file to replace it.")
else:
    st.info("👆 Please upload a file or select a sample dataset to begin.")
