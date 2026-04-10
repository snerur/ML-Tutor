"""
Page 2: Data Preprocessing
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler, RobustScaler

from utils.data_utils import detect_outliers_iqr, compute_class_distribution
from utils.ml_utils import (
    build_preprocessor,
    get_feature_names_after_preprocessing,
    encode_high_cardinality,
)

st.set_page_config(page_title="Preprocessing - ML Fairness Studio", layout="wide")

st.title("⚙️ Data Preprocessing")

# ── Guard ─────────────────────────────────────────────────────────────────────
if st.session_state.get("df") is None or st.session_state.get("target_col") is None:
    st.warning("⚠️ No dataset configured. Please go to **📂 Data Upload** first.")
    st.stop()

df_orig = st.session_state["df"].copy()
target_col = st.session_state["target_col"]
numeric_cols = st.session_state.get("numeric_cols", [])
categorical_cols = st.session_state.get("categorical_cols", [])
protected_cols = st.session_state.get("protected_cols", [])
task_type = st.session_state.get("task_type", "classification")

st.markdown(
    f"**Dataset:** {df_orig.shape[0]:,} rows × {df_orig.shape[1]} columns | "
    f"**Target:** `{target_col}` | **Task:** {task_type.capitalize()}"
)

# ── Feature selection ─────────────────────────────────────────────────────────
with st.expander("🔧 Feature Selection", expanded=False):
    all_feature_cols = [c for c in df_orig.columns if c != target_col]
    selected_features = st.multiselect(
        "Select features to include in training:",
        all_feature_cols,
        default=all_feature_cols,
        help="Unselected features will be excluded from training.",
    )
    numeric_cols = [c for c in numeric_cols if c in selected_features]
    categorical_cols = [c for c in categorical_cols if c in selected_features]
    st.markdown(f"**Selected:** {len(selected_features)} features ({len(numeric_cols)} numeric, {len(categorical_cols)} categorical)")

    st.markdown("---")
    st.markdown(
        "**Algorithmic Feature Selection** "
        "*(applied automatically after encoding & scaling — uses training data only to avoid data leakage)*"
    )

    algo_fs_method = st.selectbox(
        "Method",
        [
            "None",
            "SelectKBest",
            "Lasso (L1 Regularization)",
            "Recursive Feature Elimination (RFE)",
            "Dimensionality Reduction (PCA)",
        ],
        help=(
            "SelectKBest — keeps the k highest-scoring features via a univariate statistical test.\n"
            "Lasso — fits an L1-penalized model and drops near-zero-coefficient features.\n"
            "RFE — iteratively prunes the least important features using a linear estimator.\n"
            "PCA — projects features into a compact set of principal components."
        ),
        key="algo_fs_method",
    )

    algo_fs_params: dict = {}

    if algo_fs_method == "SelectKBest":
        st.caption(
            "Scores every feature with an F-test (F-classification or F-regression) "
            "and retains the top k. Fast and effective for linear relationships."
        )
        _max_k = max(1, len(selected_features))
        skb_k = st.slider(
            "Number of features to select (k)",
            1, _max_k, min(10, _max_k),
            key="skb_k",
        )
        algo_fs_params["k"] = skb_k

    elif algo_fs_method == "Lasso (L1 Regularization)":
        st.caption(
            "Fits a Lasso (regression) or L1-penalized logistic regression (classification) "
            "on the preprocessed training data. Features whose coefficients shrink to zero are removed."
        )
        lasso_C = st.number_input(
            "Regularization strength C — or α for regression (smaller = stronger sparsity)",
            value=0.1, min_value=1e-4, max_value=100.0, step=0.01, format="%.4f",
            key="lasso_C",
        )
        lasso_threshold = st.selectbox(
            "Coefficient magnitude threshold for retention",
            ["mean", "median", "0.5*mean", "1.25*mean"],
            help="Features with |coefficient| below this threshold are dropped.",
            key="lasso_thresh",
        )
        algo_fs_params["C"] = lasso_C
        algo_fs_params["threshold"] = lasso_threshold

    elif algo_fs_method == "Recursive Feature Elimination (RFE)":
        st.caption(
            "Trains a linear model, removes the feature with the smallest weight, and repeats "
            "until the desired number of features remains. Captures feature interactions that "
            "univariate tests miss."
        )
        _max_rfe = max(1, len(selected_features))
        rfe_n = st.slider(
            "Number of features to keep",
            1, _max_rfe, min(10, _max_rfe),
            key="rfe_n",
        )
        algo_fs_params["n_features"] = rfe_n

    elif algo_fs_method == "Dimensionality Reduction (PCA)":
        st.caption(
            "Computes principal components — linear combinations of the original features "
            "ordered by the variance they explain. Useful when features are highly correlated. "
            "Note: components are no longer interpretable as individual features."
        )
        _max_pca = max(1, min(len(selected_features), 50))
        pca_n = st.slider(
            "Number of principal components",
            1, _max_pca, min(10, _max_pca),
            key="pca_n",
        )
        algo_fs_params["n_components"] = pca_n

# ── Section 1: Missing Value Handling ────────────────────────────────────────
with st.expander("1️⃣ Missing Value Handling", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Numeric columns strategy:**")
        num_missing = st.selectbox(
            "Numeric imputation",
            ["median", "mean", "most_frequent", "KNN", "constant", "drop"],
            index=0,
        )
        if num_missing == "constant":
            num_fill_value = st.number_input("Fill value (numeric)", value=0.0)
        elif num_missing == "KNN":
            knn_neighbors = st.slider("KNN neighbors", 1, 20, 5)
        else:
            num_fill_value = None
            knn_neighbors = 5

    with col2:
        st.markdown("**Categorical columns strategy:**")
        cat_missing = st.selectbox(
            "Categorical imputation",
            ["most_frequent", "constant", "drop"],
            index=0,
        )
        if cat_missing == "constant":
            cat_fill_value = st.text_input("Fill value (categorical)", value="Unknown")
        else:
            cat_fill_value = None

    # Preview missing values
    missing_cols = df_orig[selected_features + [target_col]].isnull().sum()
    missing_cols = missing_cols[missing_cols > 0]
    if len(missing_cols) > 0:
        st.markdown(f"**Columns with missing values:** {list(missing_cols.index)}")
        fig_m = px.bar(x=missing_cols.index, y=missing_cols.values,
                       title="Missing values to handle",
                       labels={"x": "Column", "y": "Count"},
                       color=missing_cols.values, color_continuous_scale="Reds")
        fig_m.update_layout(height=250, showlegend=False)
        st.plotly_chart(fig_m, use_container_width=True)
    else:
        st.success("✅ No missing values in selected features.")

# ── Section 2: Outlier Handling ───────────────────────────────────────────────
with st.expander("2️⃣ Outlier Handling", expanded=False):
    handle_outliers = st.checkbox("Enable outlier handling", value=False)
    if handle_outliers:
        outlier_method = st.radio("Method", ["IQR", "Z-score"], horizontal=True)
        outlier_action = st.radio("Action", ["clip", "remove"], horizontal=True)
        if outlier_method == "Z-score":
            zscore_threshold = st.slider("Z-score threshold", 2.0, 4.0, 3.0, 0.1)
        else:
            zscore_threshold = 3.0

        if numeric_cols:
            selected_outlier_cols = st.multiselect(
                "Apply to columns:",
                numeric_cols,
                default=numeric_cols[:min(3, len(numeric_cols))],
            )
            if selected_outlier_cols:
                st.markdown("**Outlier counts per column (IQR method):**")
                for col in selected_outlier_cols:
                    outliers_df, lb, ub = detect_outliers_iqr(df_orig, col)
                    pct = len(outliers_df) / len(df_orig) * 100
                    st.markdown(f"- `{col}`: {len(outliers_df)} outliers ({pct:.1f}%) | bounds [{lb:.2f}, {ub:.2f}]")
        else:
            selected_outlier_cols = []
    else:
        outlier_method = "IQR"
        outlier_action = "clip"
        selected_outlier_cols = []
        zscore_threshold = 3.0

# ── Section 3: Encoding & Scaling ────────────────────────────────────────────
with st.expander("3️⃣ Feature Encoding & Scaling", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Encoding strategy:**")
        max_ohe_cardinality = st.slider(
            "Max unique values for OHE",
            5, 100, 20,
            help="Categorical columns with more unique values than this are considered high-cardinality.",
        )
        high_card_strategy = st.selectbox(
            "High-cardinality encoding",
            ["drop", "target", "binary"],
            format_func=lambda x: {
                "drop": "Drop (remove high-cardinality columns)",
                "target": "Target Encoding (replace with mean target value per category)",
                "binary": "Binary Encoding (ordinal → binary bit columns)",
            }[x],
            help="How to handle columns with more unique values than the OHE threshold.",
        )
    with col2:
        st.markdown("**Scaling strategy:**")
        scaler_type = st.selectbox(
            "Scaler",
            ["standard", "minmax", "robust", "none"],
            format_func=lambda x: {
                "standard": "Standard Scaler (zero mean, unit variance)",
                "minmax": "Min-Max Scaler [0, 1]",
                "robust": "Robust Scaler (median/IQR based)",
                "none": "No Scaling",
            }[x],
        )

    # Separate low- and high-cardinality categorical columns
    valid_cat_cols = [c for c in categorical_cols if df_orig[c].nunique() <= max_ohe_cardinality]
    high_card_cat_cols = [c for c in categorical_cols if c not in valid_cat_cols]
    if high_card_cat_cols:
        if high_card_strategy == "drop":
            st.warning(f"⚠️ High-cardinality columns will be dropped (>{max_ohe_cardinality} unique): {high_card_cat_cols}")
        else:
            label = "Target Encoding" if high_card_strategy == "target" else "Binary Encoding"
            st.info(f"ℹ️ High-cardinality columns → **{label}**: {high_card_cat_cols}")
    categorical_cols = valid_cat_cols

# ── Section 4: Feature Engineering ───────────────────────────────────────────
with st.expander("4️⃣ Feature Engineering (Optional)", expanded=False):
    use_poly = st.checkbox("Add polynomial features", value=False)
    poly_degree = 2
    if use_poly:
        poly_degree = st.slider("Polynomial degree", 2, 3, 2)
        st.warning(f"⚠️ Degree {poly_degree} will add many features. Use with caution on large datasets.")

    use_interactions = st.checkbox("Add interaction terms between numeric features", value=False)
    if use_interactions:
        interaction_cols = st.multiselect(
            "Select columns for interactions (2-way):",
            numeric_cols,
            default=numeric_cols[:min(3, len(numeric_cols))],
        )
    else:
        interaction_cols = []

# ── Section 5: Target Transformation (regression only) ───────────────────────
if task_type == "regression":
    with st.expander("5️⃣ Target Variable Transformation", expanded=True):
        st.markdown(
            "For regression targets, you can optionally transform the target to improve model fit "
            "(e.g. log-transform right-skewed targets, standardize for regularized models)."
        )
        target_series_preview = df_orig[target_col].dropna()

        col_ta, col_tb = st.columns(2)
        with col_ta:
            target_transform = st.selectbox(
                "Target transformation",
                ["none", "log1p", "sqrt", "standard", "minmax", "robust"],
                format_func=lambda x: {
                    "none": "None (raw values)",
                    "log1p": "Log(y + 1)  — right-skewed, non-negative",
                    "sqrt": "√y  — moderate right skew, non-negative",
                    "standard": "Standardize (zero mean, unit variance)",
                    "minmax": "Min-Max scale [0, 1]",
                    "robust": "Robust scale (median/IQR)",
                }[x],
            )
        with col_tb:
            # Preview before/after
            if target_transform == "log1p":
                transformed_preview = np.log1p(target_series_preview)
                transform_label = "log(y+1)"
            elif target_transform == "sqrt":
                _pos = target_series_preview.clip(lower=0)
                transformed_preview = np.sqrt(_pos)
                transform_label = "√y"
            elif target_transform == "standard":
                _sc = StandardScaler()
                transformed_preview = pd.Series(
                    _sc.fit_transform(target_series_preview.values.reshape(-1, 1)).ravel()
                )
                transform_label = "standardized"
            elif target_transform == "minmax":
                _sc = MinMaxScaler()
                transformed_preview = pd.Series(
                    _sc.fit_transform(target_series_preview.values.reshape(-1, 1)).ravel()
                )
                transform_label = "min-max scaled"
            elif target_transform == "robust":
                _sc = RobustScaler()
                transformed_preview = pd.Series(
                    _sc.fit_transform(target_series_preview.values.reshape(-1, 1)).ravel()
                )
                transform_label = "robust scaled"
            else:
                transformed_preview = target_series_preview
                transform_label = "raw"

            if target_transform != "none":
                fig_before_after = go.Figure()
                fig_before_after.add_trace(go.Histogram(
                    x=target_series_preview, name="Before", opacity=0.6, nbinsx=30
                ))
                fig_before_after.add_trace(go.Histogram(
                    x=transformed_preview, name=f"After ({transform_label})", opacity=0.6, nbinsx=30
                ))
                fig_before_after.update_layout(
                    barmode="overlay", height=240, title="Target: Before vs After Transform",
                    showlegend=True, margin=dict(l=5, r=5, t=30, b=5)
                )
                st.plotly_chart(fig_before_after, use_container_width=True)
else:
    target_transform = "none"

# ── Section 6: Train/Test Split & Class Balance ───────────────────────────────
with st.expander("6️⃣ Train/Test Split & Class Balance", expanded=True):
    col1, col2, col3 = st.columns(3)
    with col1:
        test_size = st.slider("Test set size (%)", 10, 40, 20, 5) / 100
    with col2:
        random_seed = st.number_input("Random seed", value=42, min_value=0)
    with col3:
        if task_type == "regression":
            stratify = False
            st.info("Stratification is not applicable for regression.")
        else:
            stratify = st.checkbox("Stratified split", value=True,
                                   help="Maintain class distribution in train/test sets")

    if task_type == "classification":
        st.markdown("**Class imbalance handling:**")
        imbalance_method = st.selectbox(
            "Method",
            ["none", "SMOTE", "class_weight"],
            format_func=lambda x: {
                "none": "None (use as-is)",
                "SMOTE": "SMOTE (oversample minority class)",
                "class_weight": "Class weights (adjust loss function)",
            }[x],
        )

        if imbalance_method == "SMOTE":
            try:
                from imblearn.over_sampling import SMOTE
                st.success("✅ imbalanced-learn available for SMOTE")
            except ImportError:
                st.error("❌ imbalanced-learn not installed. Run: pip install imbalanced-learn")
                imbalance_method = "none"
    else:
        imbalance_method = "none"

# ── Run Preprocessing ─────────────────────────────────────────────────────────
st.markdown("---")
if st.button("🚀 Run Preprocessing", type="primary", use_container_width=True):
    with st.spinner("Running preprocessing pipeline..."):
        progress_bar = st.progress(0)

        try:
            df_work = df_orig.copy()

            # Step 1: Handle outliers
            progress_bar.progress(10)
            if handle_outliers and selected_outlier_cols:
                for col in selected_outlier_cols:
                    if outlier_method == "IQR":
                        _, lb, ub = detect_outliers_iqr(df_work, col)
                    else:
                        mean = df_work[col].mean()
                        std = df_work[col].std()
                        lb = mean - zscore_threshold * std
                        ub = mean + zscore_threshold * std

                    if outlier_action == "clip":
                        df_work[col] = df_work[col].clip(lower=lb, upper=ub)
                    else:  # remove
                        mask = (df_work[col] >= lb) & (df_work[col] <= ub)
                        df_work = df_work[mask]

            # Step 2: Prepare X and y
            progress_bar.progress(20)
            # Include high-cardinality cols in X so the preprocessor can encode them
            feature_cols = numeric_cols + categorical_cols + high_card_cat_cols
            X = df_work[feature_cols].copy()
            y = df_work[target_col].copy()

            # Save raw sensitive features before encoding
            sensitive_raw = {}
            for pc in protected_cols:
                if pc in df_work.columns and pc != target_col:
                    sensitive_raw[pc] = df_work[pc].copy()

            # Handle "drop" strategy: remove rows with missing values before splitting
            if num_missing == "drop" and numeric_cols:
                drop_mask = X[numeric_cols].isna().any(axis=1)
                if drop_mask.any():
                    X = X[~drop_mask]
                    y = y[~drop_mask]
                    for pc in list(sensitive_raw.keys()):
                        sensitive_raw[pc] = sensitive_raw[pc][~drop_mask]
            all_cat_cols_in_X = [c for c in (categorical_cols + high_card_cat_cols) if c in X.columns]
            if cat_missing == "drop" and all_cat_cols_in_X:
                drop_mask_cat = X[all_cat_cols_in_X].isna().any(axis=1)
                if drop_mask_cat.any():
                    X = X[~drop_mask_cat]
                    y = y[~drop_mask_cat]
                    for pc in list(sensitive_raw.keys()):
                        sensitive_raw[pc] = sensitive_raw[pc][~drop_mask_cat]

            # Step 3: Train/test split
            progress_bar.progress(30)
            if task_type == "regression":
                stratify_y = None  # regression can't stratify continuous target
            else:
                stratify_y = y if stratify else None
            X_train_raw, X_test_raw, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_seed, stratify=stratify_y
            )

            # Split sensitive features along same split
            sensitive_train_dict = {}
            sensitive_test_dict = {}
            for pc, s_series in sensitive_raw.items():
                s_train = s_series.loc[X_train_raw.index]
                s_test = s_series.loc[X_test_raw.index]
                sensitive_train_dict[pc] = s_train.reset_index(drop=True)
                sensitive_test_dict[pc] = s_test.reset_index(drop=True)

            # Label-encode string classification targets
            label_encoder_obj = None
            if task_type == "classification" and pd.api.types.is_object_dtype(y_train):
                from sklearn.preprocessing import LabelEncoder as _LE
                label_encoder_obj = _LE()
                y_train = pd.Series(
                    label_encoder_obj.fit_transform(y_train),
                    name=target_col
                )
                y_test = pd.Series(
                    label_encoder_obj.transform(y_test),
                    name=target_col
                )

            # Step 4a: Pre-encode high-cardinality columns (outside ColumnTransformer)
            progress_bar.progress(35)
            hc_enc_cols = []
            hc_encoder_obj = None
            if high_card_cat_cols and high_card_strategy != "drop":
                X_train_raw, X_test_raw, hc_enc_cols, hc_encoder_obj = encode_high_cardinality(
                    X_train_raw, X_test_raw,
                    high_card_cols=high_card_cat_cols,
                    strategy=high_card_strategy,
                    y_train=y_train,
                )
                # Treat the newly created numeric columns as numeric for scaling
                numeric_cols = numeric_cols + hc_enc_cols

            # Step 4b: Build preprocessor
            progress_bar.progress(40)
            use_knn = (num_missing == "KNN")
            if use_knn or num_missing == "drop":
                actual_num_strategy = "median"
            elif num_missing == "constant":
                actual_num_strategy = "constant"
            else:
                actual_num_strategy = num_missing

            preprocessor = build_preprocessor(
                numeric_cols=numeric_cols,
                categorical_cols=categorical_cols,
                num_strategy=actual_num_strategy,
                cat_strategy=cat_missing if cat_missing != "constant" else "most_frequent",
                scaler_type=scaler_type,
                use_knn_imputer=use_knn,
                knn_neighbors=knn_neighbors if use_knn else 5,
            )

            # Step 5: Fit and transform
            progress_bar.progress(55)
            X_train_processed = preprocessor.fit_transform(X_train_raw)
            X_test_processed = preprocessor.transform(X_test_raw)

            # Step 6: Feature engineering
            progress_bar.progress(65)
            feature_names = get_feature_names_after_preprocessing(
                preprocessor, numeric_cols, categorical_cols
            )

            if use_poly and numeric_cols:
                poly = PolynomialFeatures(degree=poly_degree, include_bias=False, interaction_only=False)
                # Apply only to numeric columns (first len(numeric_cols) cols)
                n_num = len(numeric_cols)
                X_train_poly = poly.fit_transform(X_train_processed[:, :n_num])
                X_test_poly = poly.transform(X_test_processed[:, :n_num])
                poly_names = poly.get_feature_names_out(numeric_cols).tolist()
                X_train_processed = np.hstack([X_train_poly, X_train_processed[:, n_num:]])
                X_test_processed = np.hstack([X_test_poly, X_test_processed[:, n_num:]])
                feature_names = poly_names + feature_names[n_num:]

            if use_interactions and len(interaction_cols) >= 2:
                # Find indices of interaction columns in feature_names
                int_idx = [feature_names.index(c) for c in interaction_cols if c in feature_names]
                if len(int_idx) >= 2:
                    for i in range(len(int_idx)):
                        for j in range(i + 1, len(int_idx)):
                            col_a_vals_train = X_train_processed[:, int_idx[i]]
                            col_b_vals_train = X_train_processed[:, int_idx[j]]
                            col_a_vals_test = X_test_processed[:, int_idx[i]]
                            col_b_vals_test = X_test_processed[:, int_idx[j]]
                            interaction_train = (col_a_vals_train * col_b_vals_train).reshape(-1, 1)
                            interaction_test = (col_a_vals_test * col_b_vals_test).reshape(-1, 1)
                            X_train_processed = np.hstack([X_train_processed, interaction_train])
                            X_test_processed = np.hstack([X_test_processed, interaction_test])
                            feature_names.append(f"{interaction_cols[i]}__x__{interaction_cols[j]}")

            # Step 6b: Algorithmic feature selection
            if algo_fs_method != "None":
                progress_bar.progress(68)
                try:
                    y_train_arr_fs = np.asarray(y_train)

                    if algo_fs_method == "SelectKBest":
                        from sklearn.feature_selection import SelectKBest, f_classif, f_regression
                        score_fn = f_regression if task_type == "regression" else f_classif
                        k = min(algo_fs_params["k"], X_train_processed.shape[1])
                        fs_selector = SelectKBest(score_fn, k=k)
                        X_train_processed = fs_selector.fit_transform(X_train_processed, y_train_arr_fs)
                        X_test_processed = fs_selector.transform(X_test_processed)
                        mask = fs_selector.get_support()
                        feature_names = [fn for fn, m in zip(feature_names, mask) if m]
                        st.info(f"SelectKBest: retained {X_train_processed.shape[1]} features.")

                    elif algo_fs_method == "Lasso (L1 Regularization)":
                        from sklearn.feature_selection import SelectFromModel
                        if task_type == "regression":
                            from sklearn.linear_model import Lasso
                            lasso_est = Lasso(alpha=algo_fs_params["C"], max_iter=2000)
                        else:
                            from sklearn.linear_model import LogisticRegression
                            lasso_est = LogisticRegression(
                                C=algo_fs_params["C"], penalty="l1",
                                solver="liblinear", max_iter=1000,
                            )
                        fs_selector = SelectFromModel(lasso_est, threshold=algo_fs_params["threshold"])
                        fs_selector.fit(X_train_processed, y_train_arr_fs)
                        X_train_new = fs_selector.transform(X_train_processed)
                        X_test_new = fs_selector.transform(X_test_processed)
                        if X_train_new.shape[1] == 0:
                            st.warning("⚠️ Lasso removed all features — try a larger C or a looser threshold. Skipping.")
                        else:
                            X_train_processed = X_train_new
                            X_test_processed = X_test_new
                            mask = fs_selector.get_support()
                            feature_names = [fn for fn, m in zip(feature_names, mask) if m]
                            st.info(f"Lasso: retained {X_train_processed.shape[1]} features.")

                    elif algo_fs_method == "Recursive Feature Elimination (RFE)":
                        from sklearn.feature_selection import RFE
                        from sklearn.linear_model import LogisticRegression, LinearRegression
                        est_rfe = (
                            LinearRegression()
                            if task_type == "regression"
                            else LogisticRegression(max_iter=1000, solver="liblinear")
                        )
                        n_feat = min(algo_fs_params["n_features"], X_train_processed.shape[1])
                        fs_selector = RFE(est_rfe, n_features_to_select=n_feat)
                        X_train_processed = fs_selector.fit_transform(X_train_processed, y_train_arr_fs)
                        X_test_processed = fs_selector.transform(X_test_processed)
                        mask = fs_selector.get_support()
                        feature_names = [fn for fn, m in zip(feature_names, mask) if m]
                        st.info(f"RFE: retained {X_train_processed.shape[1]} features.")

                    elif algo_fs_method == "Dimensionality Reduction (PCA)":
                        from sklearn.decomposition import PCA
                        n_comp = min(
                            algo_fs_params["n_components"],
                            X_train_processed.shape[1],
                            X_train_processed.shape[0],
                        )
                        pca_obj = PCA(n_components=n_comp, random_state=42)
                        X_train_processed = pca_obj.fit_transform(X_train_processed)
                        X_test_processed = pca_obj.transform(X_test_processed)
                        feature_names = [f"PC_{i + 1}" for i in range(n_comp)]
                        st.session_state["_pca"] = pca_obj
                        explained = pca_obj.explained_variance_ratio_.cumsum()[-1] * 100
                        st.info(
                            f"PCA: {n_comp} components explain "
                            f"{explained:.1f}% of total variance."
                        )

                except Exception as _fs_err:
                    st.warning(
                        f"⚠️ Algorithmic feature selection ({algo_fs_method}) encountered an error "
                        f"and was skipped: {_fs_err}"
                    )

            # Step 7: Target transformation (regression only)
            progress_bar.progress(70)
            target_transformer_obj = None
            if task_type == "regression" and target_transform != "none":
                y_train_arr = np.asarray(y_train, dtype=float)
                y_test_arr = np.asarray(y_test, dtype=float)
                if target_transform == "log1p":
                    y_train = pd.Series(np.log1p(y_train_arr))
                    y_test = pd.Series(np.log1p(y_test_arr))
                    target_transformer_obj = ("log1p", None)
                elif target_transform == "sqrt":
                    y_train = pd.Series(np.sqrt(np.clip(y_train_arr, 0, None)))
                    y_test = pd.Series(np.sqrt(np.clip(y_test_arr, 0, None)))
                    target_transformer_obj = ("sqrt", None)
                elif target_transform in ("standard", "minmax", "robust"):
                    if target_transform == "standard":
                        tscaler = StandardScaler()
                    elif target_transform == "minmax":
                        tscaler = MinMaxScaler()
                    else:
                        tscaler = RobustScaler()
                    y_train = pd.Series(
                        tscaler.fit_transform(y_train_arr.reshape(-1, 1)).ravel()
                    )
                    y_test = pd.Series(
                        tscaler.transform(y_test_arr.reshape(-1, 1)).ravel()
                    )
                    target_transformer_obj = (target_transform, tscaler)

            # Step 8: Class imbalance (classification only)
            progress_bar.progress(75)
            sample_weights = None
            if task_type == "classification":
                if imbalance_method == "SMOTE":
                    from imblearn.over_sampling import SMOTE
                    smote = SMOTE(random_state=random_seed)
                    X_train_processed, y_train_arr2 = smote.fit_resample(X_train_processed, y_train)
                    y_train = pd.Series(y_train_arr2)
                    for pc in list(sensitive_train_dict.keys()):
                        sensitive_train_dict.pop(pc, None)
                    st.session_state["sensitive_train"] = None
                    st.session_state["sensitive_test"] = None
                elif imbalance_method == "class_weight":
                    from sklearn.utils.class_weight import compute_sample_weight
                    sample_weights = compute_sample_weight("balanced", y_train)
                    st.session_state["sample_weights"] = sample_weights

            # Step 9: Save all to session state
            progress_bar.progress(90)
            st.session_state["X_train"] = X_train_processed
            st.session_state["X_test"] = X_test_processed
            st.session_state["y_train"] = y_train.reset_index(drop=True) if hasattr(y_train, "reset_index") else y_train
            st.session_state["y_test"] = y_test.reset_index(drop=True) if hasattr(y_test, "reset_index") else y_test
            st.session_state["X_train_raw"] = X_train_raw.reset_index(drop=True)
            st.session_state["X_test_raw"] = X_test_raw.reset_index(drop=True)
            st.session_state["preprocessor"] = preprocessor
            st.session_state["feature_names"] = feature_names
            st.session_state["numeric_cols"] = numeric_cols
            st.session_state["categorical_cols"] = categorical_cols
            st.session_state["sensitive_train_dict"] = sensitive_train_dict
            st.session_state["sensitive_test_dict"] = sensitive_test_dict
            st.session_state["label_encoder"] = label_encoder_obj
            st.session_state["hc_encoder"] = hc_encoder_obj
            st.session_state["hc_enc_cols"] = hc_enc_cols

            if sensitive_train_dict:
                # Pick first protected col for default
                first_pc = list(sensitive_train_dict.keys())[0]
                st.session_state["sensitive_train"] = sensitive_train_dict[first_pc].values
                st.session_state["sensitive_test"] = sensitive_test_dict[first_pc].values
                st.session_state["sensitive_col_for_fairness"] = first_pc

            st.session_state["target_transformer"] = target_transformer_obj
            st.session_state["preprocessing_config"] = {
                "num_missing": num_missing,
                "cat_missing": cat_missing,
                "scaler_type": scaler_type,
                "test_size": test_size,
                "random_seed": int(random_seed),
                "stratify": stratify,
                "imbalance_method": imbalance_method,
                "handle_outliers": handle_outliers,
                "outlier_method": outlier_method if handle_outliers else None,
                "outlier_action": outlier_action if handle_outliers else None,
                "use_poly": use_poly,
                "poly_degree": poly_degree if use_poly else None,
                "task_type": task_type,
                "target_transform": target_transform,
                "algo_fs_method": algo_fs_method,
                "algo_fs_params": algo_fs_params,
            }

            progress_bar.progress(100)

            # ── Results summary ────────────────────────────────────────────
            st.success("✅ Preprocessing complete!")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Training samples", f"{X_train_processed.shape[0]:,}")
            with col2:
                st.metric("Test samples", f"{X_test_processed.shape[0]:,}")
            with col3:
                st.metric("Features (after)", X_train_processed.shape[1])
            with col4:
                st.metric("Features (before)", len(feature_cols))

            if task_type == "regression":
                # Show target distributions for regression
                st.markdown("#### Target Distribution (Train vs Test)")
                y_train_series = pd.Series(st.session_state["y_train"], dtype=float)
                y_test_series = pd.Series(st.session_state["y_test"], dtype=float)
                col_a, col_b = st.columns(2)
                with col_a:
                    fig_t = px.histogram(y_train_series, nbins=40, title="Train Target Distribution")
                    fig_t.update_layout(height=280, showlegend=False)
                    st.plotly_chart(fig_t, use_container_width=True)
                    st.markdown(f"Mean: {y_train_series.mean():.4f} | Std: {y_train_series.std():.4f}")
                with col_b:
                    fig_te = px.histogram(y_test_series, nbins=40, title="Test Target Distribution")
                    fig_te.update_layout(height=280, showlegend=False)
                    st.plotly_chart(fig_te, use_container_width=True)
                    st.markdown(f"Mean: {y_test_series.mean():.4f} | Std: {y_test_series.std():.4f}")
                if target_transform != "none":
                    st.info(f"Target has been transformed using: **{target_transform}**. "
                            "Predictions will be in the transformed scale.")
            else:
                # Before/after class distributions
                st.markdown("#### Class Distribution Comparison")
                y_train_series = pd.Series(st.session_state["y_train"])
                y_test_series = pd.Series(st.session_state["y_test"])

                col_a, col_b = st.columns(2)
                with col_a:
                    train_dist = y_train_series.value_counts()
                    fig_t = px.pie(values=train_dist.values, names=train_dist.index.astype(str),
                                   title="Training Set Class Distribution", hole=0.4)
                    fig_t.update_layout(height=280)
                    st.plotly_chart(fig_t, use_container_width=True)
                with col_b:
                    test_dist = y_test_series.value_counts()
                    fig_te = px.pie(values=test_dist.values, names=test_dist.index.astype(str),
                                    title="Test Set Class Distribution", hole=0.4)
                    fig_te.update_layout(height=280)
                    st.plotly_chart(fig_te, use_container_width=True)

            st.info("➡️ Next step: Navigate to **🔍 Bias Detection** or **🏋️ Model Training**.")

        except Exception as e:
            st.error(f"❌ Preprocessing failed: {e}")
            import traceback
            st.code(traceback.format_exc())
        finally:
            progress_bar.empty()

# Show current state if already preprocessed
elif st.session_state.get("X_train") is not None:
    X_tr = st.session_state["X_train"]
    X_te = st.session_state["X_test"]
    st.success(f"✅ Data already preprocessed: {X_tr.shape[0]:,} train samples, {X_te.shape[0]:,} test samples, {X_tr.shape[1]} features.")
    st.info("You can re-run preprocessing above with different settings.")
