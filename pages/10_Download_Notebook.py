"""
Page 10: Download Jupyter Notebook
Generates a self-contained .ipynb notebook from the current session's pipeline
configuration and makes it available for download.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import utils.compat  # noqa: F401

import json
import textwrap
import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(page_title="Download Notebook - ML Fairness Studio", layout="wide")

st.title("📓 Download Jupyter Notebook")
st.markdown("""
Generate a **self-contained Jupyter Notebook** that reproduces the full ML Fairness Studio
pipeline — data loading, preprocessing, model training (with the LightGBM CV fix),
feature importance with confidence intervals & p-values, and causal inference.

The notebook runs standalone: just open it in JupyterLab / VS Code and execute all cells.
""")

# ── Read current session config ────────────────────────────────────────────────
model_name = st.session_state.get("model_name", "LightGBM")
target_col = st.session_state.get("target_col", "income")
protected_cols = st.session_state.get("protected_cols", ["gender", "race"])
numeric_cols = st.session_state.get("numeric_cols", [])
categorical_cols = st.session_state.get("categorical_cols", [])
cv_results = st.session_state.get("cv_results", {})
preprocessing_cfg = st.session_state.get("preprocessing_config", {})

st.markdown("### Notebook Configuration")
col1, col2, col3 = st.columns(3)
with col1:
    nb_model = st.selectbox(
        "Model for notebook",
        ["LightGBM", "Random Forest", "Gradient Boosting", "Logistic Regression", "XGBoost"],
        index=["LightGBM", "Random Forest", "Gradient Boosting", "Logistic Regression", "XGBoost"].index(model_name)
        if model_name in ["LightGBM", "Random Forest", "Gradient Boosting", "Logistic Regression", "XGBoost"]
        else 0,
    )
with col2:
    nb_dataset = st.selectbox(
        "Dataset",
        ["Adult Income (built-in synthetic)", "Upload your own (edit path in notebook)"],
    )
with col3:
    nb_cv_folds = st.slider("CV folds", 3, 10, cv_results.get("folds", 5))

nb_treatment = st.text_input(
    "Causal inference — treatment variable (column name):",
    value="education-num",
    help="Set to any numeric/binary column in the dataset.",
)
nb_treatment_thresh = st.number_input(
    "Treatment threshold (treated = column ≥ threshold):",
    value=13.0,
    step=1.0,
)


# ── Notebook generation ────────────────────────────────────────────────────────
def _cell(source, cell_type="code"):
    """Return a notebook cell dict."""
    return {
        "cell_type": cell_type,
        "metadata": {},
        "source": source if isinstance(source, list) else [source],
        **({"outputs": [], "execution_count": None} if cell_type == "code" else {}),
    }


def _md(text):
    return _cell(textwrap.dedent(text).strip(), cell_type="markdown")


def _code(text):
    lines = textwrap.dedent(text).strip().splitlines(keepends=True)
    return _cell(lines, cell_type="code")


def build_notebook(model: str, cv_folds: int, treatment_col: str, treatment_thresh: float) -> dict:
    model_map = {
        "LightGBM": "lgbm",
        "Random Forest": "rf",
        "Gradient Boosting": "gb",
        "Logistic Regression": "lr",
        "XGBoost": "xgb",
    }
    model_key = model_map.get(model, "lgbm")

    model_init_code = {
        "lgbm": "from lightgbm import LGBMClassifier\nmodel = LGBMClassifier(n_estimators=100, learning_rate=0.1, num_leaves=31, verbose=-1, subsample_freq=1)",
        "rf": "from sklearn.ensemble import RandomForestClassifier\nmodel = RandomForestClassifier(n_estimators=100, random_state=42)",
        "gb": "from sklearn.ensemble import GradientBoostingClassifier\nmodel = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)",
        "lr": "from sklearn.linear_model import LogisticRegression\nmodel = LogisticRegression(max_iter=1000, random_state=42)",
        "xgb": "from xgboost import XGBClassifier\nmodel = XGBClassifier(n_estimators=100, learning_rate=0.1, eval_metric='logloss', random_state=42)",
    }

    cv_jobs_comment = (
        "# LightGBM uses OpenMP internally; n_jobs=1 avoids threading conflicts in CV\ncv_n_jobs = 1"
        if model == "LightGBM"
        else "cv_n_jobs = -1"
    )

    cells = [
        _md("""\
        # ML Fairness Studio — Full Pipeline Notebook

        This notebook reproduces the complete ML Fairness Studio pipeline:
        1. Data loading & exploration
        2. Preprocessing
        3. Model training with cross-validation (LightGBM threading fix included)
        4. Feature importance with confidence intervals & p-values
        5. Causal inference (propensity-score methods + Doubly Robust AIPW)

        **Requirements:** `pip install lightgbm scikit-learn pandas numpy scipy plotly`
        """),

        _md("## 1. Imports & Setup"),
        _code("""\
        import warnings
        warnings.filterwarnings("ignore")

        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import plotly.express as px
        import plotly.graph_objects as go
        from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.metrics import classification_report, roc_auc_score
        from scipy import stats as sp_stats

        RANDOM_STATE = 42
        np.random.seed(RANDOM_STATE)
        print("Imports OK")
        """),

        _md("## 2. Load Dataset"),
        _code(f"""\
        # ── Synthetic Adult Income dataset ─────────────────────────────────────────
        # Replace this block with  df = pd.read_csv("your_file.csv")  if needed.

        np.random.seed(42)
        n = 2000
        age = np.random.randint(18, 65, n)
        education_num = np.random.randint(1, 16, n)
        hours_per_week = np.random.randint(20, 60, n)
        capital_gain = np.random.exponential(500, n) * (np.random.rand(n) < 0.15)
        gender = np.random.choice(["Male", "Female"], n, p=[0.67, 0.33])
        race = np.random.choice(["White", "Black", "Asian", "Other"], n, p=[0.75, 0.10, 0.10, 0.05])
        occupation = np.random.choice(
            ["Tech-support", "Craft-repair", "Other-service", "Sales",
             "Exec-managerial", "Prof-specialty", "Handlers-cleaners",
             "Machine-op-inspct", "Adm-clerical", "Transport-moving"],
            n,
        )
        marital_status = np.random.choice(
            ["Married-civ-spouse", "Never-married", "Divorced", "Separated", "Widowed"], n
        )

        log_odds = (
            -4.0
            + 0.04 * age
            + 0.25 * education_num
            + 0.01 * hours_per_week
            + 0.0001 * capital_gain
            + (gender == "Male") * 0.5
            + (race == "White") * 0.3
            + (race == "Asian") * 0.2
            + (race == "Black") * -0.2
        )
        proba = 1 / (1 + np.exp(-log_odds))
        income_bin = (np.random.rand(n) < proba).astype(int)
        income = np.where(income_bin == 1, ">50K", "<=50K")

        df = pd.DataFrame({{
            "age": age,
            "education-num": education_num,
            "hours-per-week": hours_per_week,
            "capital-gain": capital_gain,
            "gender": gender,
            "race": race,
            "occupation": occupation,
            "marital-status": marital_status,
            "income": income,
        }})

        TARGET = "income"
        PROTECTED = ["gender", "race"]
        print(f"Dataset shape: {{df.shape}}")
        df.head()
        """),

        _md("## 3. Preprocessing"),
        _code("""\
        numeric_cols = ["age", "education-num", "hours-per-week", "capital-gain"]
        categorical_cols = ["gender", "race", "occupation", "marital-status"]

        X = df.drop(columns=[TARGET])
        y = (df[TARGET] == ">50K").astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
        )

        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        preprocessor = ColumnTransformer([
            ("num", num_pipeline, numeric_cols),
            ("cat", cat_pipeline, categorical_cols),
        ])

        X_train_proc = preprocessor.fit_transform(X_train)
        X_test_proc  = preprocessor.transform(X_test)

        # Recover feature names
        cat_feat_names = preprocessor.named_transformers_["cat"]["encoder"].get_feature_names_out(categorical_cols).tolist()
        feature_names = numeric_cols + cat_feat_names

        y_train_arr = y_train.values
        y_test_arr  = y_test.values

        print(f"Train: {{X_train_proc.shape}} | Test: {{X_test_proc.shape}}")
        """),

        _md(f"## 4. Model Training — {model} (with LightGBM CV Fix)"),
        _code(f"""\
        {model_init_code[model_key]}

        # Cross-validation
        # ─────────────────────────────────────────────────────────────────────
        {cv_jobs_comment}

        cv_scores = cross_val_score(
            model,
            X_train_proc,
            y_train_arr,
            cv=StratifiedKFold(n_splits={cv_folds}, shuffle=True, random_state=RANDOM_STATE),
            scoring="roc_auc",
            n_jobs=cv_n_jobs,
        )

        print(f"CV ROC-AUC: {{cv_scores.mean():.4f}} ± {{cv_scores.std():.4f}}")

        # Full training
        model.fit(X_train_proc, y_train_arr)
        y_pred = model.predict(X_test_proc)
        print("\\nClassification Report:")
        print(classification_report(y_test_arr, y_pred))
        if hasattr(model, "predict_proba"):
            roc = roc_auc_score(y_test_arr, model.predict_proba(X_test_proc)[:, 1])
            print(f"Test ROC-AUC: {{roc:.4f}}")
        """),

        _md(f"""\
        ## 5. Feature Importance with Confidence Intervals & P-values

        We use two approaches:

        - **Permutation importance** (model-agnostic): measures how much accuracy drops when a
          feature is randomly shuffled. P-values come from a one-sample t-test (H₀: mean importance = 0).
        - **Bootstrap CI** (tree-models only): resample training data, retrain, record importances,
          then take percentile-based 95% CIs.
        """),

        _code("""\
        from sklearn.inspection import permutation_importance

        # ── Permutation importance with t-test p-values ────────────────────────────
        N_REPEATS = 10
        perm_result = permutation_importance(
            model, X_test_proc, y_test_arr,
            n_repeats=N_REPEATS, random_state=RANDOM_STATE, n_jobs=-1,
        )

        perm_rows = []
        for i, feat in enumerate(feature_names):
            mean_imp = perm_result.importances_mean[i]
            std_imp  = perm_result.importances_std[i]
            se = std_imp / np.sqrt(N_REPEATS)
            if se > 0:
                t_stat = mean_imp / se
                p_val  = sp_stats.t.sf(t_stat, df=N_REPEATS - 1)  # one-sided: H1: mean > 0
                ci_half = sp_stats.t.ppf(0.975, df=N_REPEATS - 1) * se
            else:
                p_val, ci_half = (0.0 if mean_imp > 0 else 1.0), 0.0

            perm_rows.append({
                "Feature": feat,
                "Importance": mean_imp,
                "CI Lower": mean_imp - ci_half,
                "CI Upper": mean_imp + ci_half,
                "p-value": p_val,
                "Significant": p_val < 0.05,
            })

        perm_df = pd.DataFrame(perm_rows).sort_values("Importance", ascending=False)
        perm_df.head(15)
        """),

        _code("""\
        # ── Plot: Feature Importance with 95% CI ──────────────────────────────────
        top20 = perm_df.head(20).sort_values("Importance")
        colors = ["#e74c3c" if s else "#95a5a6" for s in top20["Significant"]]

        fig, ax = plt.subplots(figsize=(9, 7))
        ax.barh(top20["Feature"], top20["Importance"],
                xerr=[top20["Importance"] - top20["CI Lower"],
                      top20["CI Upper"] - top20["Importance"]],
                color=colors, ecolor="black", capsize=3)
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Mean Decrease in ROC-AUC")
        ax.set_title("Permutation Importance (red = p < 0.05)")
        plt.tight_layout()
        plt.show()
        """),

        _code("""\
        # ── Bootstrap CI (tree/boosting models only) ──────────────────────────────
        if hasattr(model, "feature_importances_"):
            from sklearn.base import clone as _clone

            N_BOOT = 100
            rng_b  = np.random.RandomState(RANDOM_STATE)
            n_samp = X_train_proc.shape[0]
            boot_imps = []

            for _ in range(N_BOOT):
                idx_b  = rng_b.choice(n_samp, size=n_samp, replace=True)
                m_b    = _clone(model)
                m_b.fit(X_train_proc[idx_b], y_train_arr[idx_b])
                boot_imps.append(m_b.feature_importances_)

            boot_arr  = np.array(boot_imps)            # (N_BOOT, n_features)
            actual    = model.feature_importances_

            # Permutation-test p-values (null: shuffle labels)
            N_NULL = 50
            null_imps = []
            for _ in range(N_NULL):
                m_p = _clone(model)
                m_p.fit(X_train_proc, rng_b.permutation(y_train_arr))
                null_imps.append(m_p.feature_importances_)
            null_arr = np.array(null_imps)

            boot_rows = []
            for i, feat in enumerate(feature_names):
                p_b = max((null_arr[:, i] >= actual[i]).mean(), 1 / (N_NULL + 1))
                boot_rows.append({
                    "Feature":       feat,
                    "Importance":    actual[i],
                    "CI Lower 95%":  np.percentile(boot_arr[:, i], 2.5),
                    "CI Upper 95%":  np.percentile(boot_arr[:, i], 97.5),
                    "p-value":       p_b,
                    "Significant":   p_b < 0.05,
                })

            boot_df = pd.DataFrame(boot_rows).sort_values("Importance", ascending=False)
            print(f"Significant features: {boot_df['Significant'].sum()} / {len(boot_df)}")
            boot_df.head(15)
        else:
            print("Bootstrap CI skipped (model does not have feature_importances_).")
        """),

        _md(f"""\
        ## 6. Causal Inference

        **Question:** Does `{treatment_col}` ≥ {treatment_thresh:.0f} causally *increase* the
        probability of earning >50 K, after controlling for age, occupation, marital status, and
        capital gain?

        Methods used:
        - **Naive Difference** (unadjusted, biased baseline)
        - **Regression Adjustment** (g-computation)
        - **IPW** — Inverse Probability Weighting
        - **Doubly Robust AIPW** — consistent if either the outcome or propensity model is correct

        Bootstrap 95% CIs + two-sided z-test p-values are reported for each.
        """),

        _code(f"""\
        from sklearn.linear_model import LogisticRegression

        # ── Variable definitions ───────────────────────────────────────────────────
        TREATMENT_COL = "{treatment_col}"
        TREATMENT_THRESH = {treatment_thresh}
        CONFOUNDERS = ["age", "hours-per-week", "capital-gain", "marital-status", "occupation"]

        df_ci = df.dropna(subset=CONFOUNDERS + [TREATMENT_COL, TARGET]).copy()
        T = (df_ci[TREATMENT_COL] >= TREATMENT_THRESH).astype(int).values
        Y = (df_ci[TARGET] == ">50K").astype(int).values

        # Build confounder matrix
        num_c = [c for c in CONFOUNDERS if pd.api.types.is_numeric_dtype(df_ci[c])]
        cat_c = [c for c in CONFOUNDERS if not pd.api.types.is_numeric_dtype(df_ci[c])]

        ct_ci = ColumnTransformer([
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                              ("sc", StandardScaler())]), num_c),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                              ("ohe", OneHotEncoder(handle_unknown="ignore",
                                                    sparse_output=False))]), cat_c),
        ]) if CONFOUNDERS else None

        X_conf = ct_ci.fit_transform(df_ci[CONFOUNDERS]) if ct_ci is not None else np.zeros((len(df_ci), 1))

        print(f"N={{len(df_ci)}} | Treated={{T.sum()}} | Control={{(T==0).sum()}}")
        """),

        _code("""\
        # ── Estimation helpers ─────────────────────────────────────────────────────
        def fit_ps(X_c, T_arr):
            m = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
            m.fit(X_c, T_arr)
            return np.clip(m.predict_proba(X_c)[:, 1], 1e-3, 1 - 1e-3)

        def naive_diff(T_arr, Y_arr):
            return Y_arr[T_arr == 1].mean() - Y_arr[T_arr == 0].mean()

        def reg_adj(X_c, T_arr, Y_arr):
            X_t = np.column_stack([X_c, T_arr])
            m = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs").fit(X_t, Y_arr)
            m1 = m.predict_proba(np.column_stack([X_c, np.ones(len(T_arr))]))[:, 1]
            m0 = m.predict_proba(np.column_stack([X_c, np.zeros(len(T_arr))]))[:, 1]
            return (m1 - m0).mean()

        def ipw(X_c, T_arr, Y_arr):
            ps = fit_ps(X_c, T_arr)
            w1, w0 = T_arr / ps, (1 - T_arr) / (1 - ps)
            return (w1 * Y_arr).sum() / w1.sum() - (w0 * Y_arr).sum() / w0.sum()

        def aipw(X_c, T_arr, Y_arr):
            ps = fit_ps(X_c, T_arr)
            X_t = np.column_stack([X_c, T_arr])
            out_m = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs").fit(X_t, Y_arr)
            mu1 = out_m.predict_proba(np.column_stack([X_c, np.ones(len(T_arr))]))[:, 1]
            mu0 = out_m.predict_proba(np.column_stack([X_c, np.zeros(len(T_arr))]))[:, 1]
            phi1 = mu1 + T_arr * (Y_arr - mu1) / ps
            phi0 = mu0 + (1 - T_arr) * (Y_arr - mu0) / (1 - ps)
            return (phi1 - phi0).mean()

        METHODS = {
            "Naive Difference":       lambda Xc, T, Y: naive_diff(T, Y),
            "Regression Adjustment":  reg_adj,
            "IPW":                    ipw,
            "Doubly Robust (AIPW)":   aipw,
        }
        """),

        _code("""\
        # ── Bootstrap estimation ───────────────────────────────────────────────────
        N_BOOT_CI = 200
        rng_ci = np.random.RandomState(RANDOM_STATE)
        n_ci   = len(T)
        results = []

        for name, fn in METHODS.items():
            ate_pt = fn(X_conf, T, Y)

            boot_ates = []
            for _ in range(N_BOOT_CI):
                idx_b = rng_ci.choice(n_ci, size=n_ci, replace=True)
                try:
                    boot_ates.append(fn(X_conf[idx_b], T[idx_b], Y[idx_b]))
                except Exception:
                    pass

            b = np.array(boot_ates)
            se = b.std()
            ci_lo, ci_hi = np.percentile(b, [2.5, 97.5])
            z = ate_pt / se if se > 0 else 0.0
            p = 2 * sp_stats.norm.sf(abs(z))

            results.append({
                "Method": name,
                "ATE": ate_pt,
                "CI Lower (95%)": ci_lo,
                "CI Upper (95%)": ci_hi,
                "SE": se,
                "p-value": p,
                "Significant": "✅" if p < 0.05 else "❌",
            })
            print(f"{name:30s}  ATE={ate_pt:+.4f}  95% CI=[{ci_lo:+.4f}, {ci_hi:+.4f}]  p={p:.4f}")

        causal_df = pd.DataFrame(results)
        causal_df
        """),

        _code("""\
        # ── Forest plot ────────────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(8, 4))
        y_pos = range(len(causal_df))
        colors_f = ["#e74c3c" if s == "✅" else "#7f8c8d" for s in causal_df["Significant"]]

        ax.scatter(causal_df["ATE"], y_pos, color=colors_f, zorder=3, s=80)
        for i, row in causal_df.iterrows():
            ax.plot([row["CI Lower (95%)"], row["CI Upper (95%)"]], [i, i],
                    color=colors_f[i], linewidth=2)

        ax.axvline(0, color="black", linewidth=1, linestyle="--", label="No effect")
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(causal_df["Method"])
        ax.set_xlabel("ATE (change in P(income > 50K))")
        ax.set_title("Causal Effect Forest Plot (red = p < 0.05)")
        ax.legend()
        plt.tight_layout()
        plt.show()
        """),

        _code("""\
        # ── Propensity score overlap diagnostic ───────────────────────────────────
        ps_vals = fit_ps(X_conf, T)
        ps_df = pd.DataFrame({"PS": ps_vals, "Group": np.where(T == 1, "Treated", "Control")})

        fig, ax = plt.subplots(figsize=(7, 4))
        for grp, color in [("Treated", "#e74c3c"), ("Control", "#3498db")]:
            subset = ps_df[ps_df["Group"] == grp]["PS"]
            ax.hist(subset, bins=25, alpha=0.6, color=color, label=grp)
        ax.set_xlabel("Propensity Score")
        ax.set_ylabel("Count")
        ax.set_title("Propensity Score Overlap (positivity check)")
        ax.legend()
        plt.tight_layout()
        plt.show()
        """),

        _md("""\
        ## 7. Conclusion

        - The **Doubly Robust AIPW** estimator is the most reliable: it is consistent if
          *either* the propensity model or the outcome model is correctly specified.
        - If confidence intervals exclude 0 and p-values < 0.05, we have statistical evidence
          of a causal effect under the **no-unmeasured-confounders assumption**.
        - Inspect the **propensity score overlap** plot: if treated and control groups have
          non-overlapping propensity scores, causal estimates may be unreliable.
        """),
    ]

    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0",
            },
        },
        "cells": cells,
    }
    return nb


# ── Build & Download ───────────────────────────────────────────────────────────
if st.button("📓 Generate Notebook", type="primary", use_container_width=True):
    with st.spinner("Building notebook..."):
        nb = build_notebook(
            model=nb_model,
            cv_folds=nb_cv_folds,
            treatment_col=nb_treatment,
            treatment_thresh=nb_treatment_thresh,
        )
        nb_json = json.dumps(nb, indent=2)

    st.success("✅ Notebook ready for download!")
    st.download_button(
        label="⬇️ Download ml_fairness_studio.ipynb",
        data=nb_json,
        file_name="ml_fairness_studio.ipynb",
        mime="application/json",
    )

    st.markdown("### Notebook Preview")
    st.markdown("The notebook contains the following cells:")
    for cell in nb["cells"]:
        ct = cell["cell_type"]
        src = "".join(cell["source"])
        first_line = src.strip().splitlines()[0] if src.strip() else "(empty)"
        icon = "📝" if ct == "markdown" else "💻"
        st.markdown(f"- {icon} `{first_line[:80]}`")

st.markdown("---")
st.markdown("""
**How to run the notebook:**
```bash
jupyter lab ml_fairness_studio.ipynb
# or
jupyter notebook ml_fairness_studio.ipynb
```
All cells can be executed top-to-bottom with `Kernel → Restart & Run All`.
""")
