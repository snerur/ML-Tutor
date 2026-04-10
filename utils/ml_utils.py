"""
Machine learning utility functions for ML Fairness Studio.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
)
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    OneHotEncoder,
    LabelEncoder,
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
import time

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from sklearn.preprocessing import TargetEncoder as _SKTargetEncoder
    SKLEARN_TARGET_ENCODER = True
except ImportError:
    SKLEARN_TARGET_ENCODER = False


# ── Custom encoders ──────────────────────────────────────────────────────────

class BinaryEncoder(BaseEstimator, TransformerMixin):
    """Encode each category as a binary bit-vector (far fewer columns than OHE)."""

    def __init__(self):
        self._ordinal = None
        self._n_bits = 0
        self._feature_names_in = None

    def fit(self, X, y=None):
        df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        self._feature_names_in = list(df.columns)
        self._ordinal = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        self._ordinal.fit(df)
        n_cats = max((len(c) for c in self._ordinal.categories_), default=2)
        self._n_bits = max(1, int(np.ceil(np.log2(n_cats + 2))))
        return self

    def transform(self, X):
        df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        ordinal = (self._ordinal.transform(df).astype(int) + 1)  # shift -1 (unknown) → 0
        cols = []
        for col_idx in range(ordinal.shape[1]):
            col = ordinal[:, col_idx]
            for bit in range(self._n_bits):
                cols.append((col >> bit) & 1)
        return np.column_stack(cols).astype(float) if cols else np.empty((len(df), 0))

    def get_feature_names_out(self, input_features=None):
        names_in = input_features if input_features is not None else (self._feature_names_in or [])
        return np.array(
            [f"{f}_bin{b}" for f in names_in for b in range(self._n_bits)], dtype=object
        )


class MeanTargetEncoder(BaseEstimator, TransformerMixin):
    """Mean target encoder — fallback when sklearn.preprocessing.TargetEncoder unavailable."""

    def __init__(self):
        self._mapping = {}
        self._global_mean = 0.0
        self._feature_names_in = None

    def fit(self, X, y=None):
        df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        self._feature_names_in = list(df.columns)
        if y is None:
            self._global_mean = 0.0
            return self
        y_arr = np.asarray(y)
        try:
            y_num = y_arr.astype(float)
        except (ValueError, TypeError):
            from sklearn.preprocessing import LabelEncoder as _LE
            y_num = _LE().fit_transform(y_arr).astype(float)
        self._global_mean = float(y_num.mean())
        for col in df.columns:
            means = {}
            for cat in df[col].dropna().unique():
                mask = df[col] == cat
                means[cat] = float(y_num[mask.values].mean()) if mask.any() else self._global_mean
            self._mapping[col] = means
        return self

    def transform(self, X):
        df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        result = np.empty((len(df), len(df.columns)), dtype=float)
        for i, col in enumerate(df.columns):
            mapping = self._mapping.get(col, {})
            result[:, i] = df[col].map(mapping).fillna(self._global_mean).values
        return result

    def get_feature_names_out(self, input_features=None):
        names_in = input_features if input_features is not None else (self._feature_names_in or [])
        return np.array([f"{f}_te" for f in names_in], dtype=object)


# ── Model registry ──────────────────────────────────────────────────────────
MODELS = {
    "Logistic Regression": LogisticRegression,
    "Random Forest": RandomForestClassifier,
    "Gradient Boosting": GradientBoostingClassifier,
    "AdaBoost": AdaBoostClassifier,
    "Decision Tree": DecisionTreeClassifier,
    "SVM": SVC,
    "KNN": KNeighborsClassifier,
}

if XGBOOST_AVAILABLE:
    MODELS["XGBoost"] = XGBClassifier
if LIGHTGBM_AVAILABLE:
    MODELS["LightGBM"] = LGBMClassifier


# ── Regression model registry ────────────────────────────────────────────────
REGRESSION_MODELS = {
    "Linear Regression": LinearRegression,
    "Ridge": Ridge,
    "Lasso": Lasso,
    "ElasticNet": ElasticNet,
    "Decision Tree": DecisionTreeRegressor,
    "Random Forest": RandomForestRegressor,
    "Gradient Boosting": GradientBoostingRegressor,
    "SVR": SVR,
    "KNN": KNeighborsRegressor,
}

if XGBOOST_AVAILABLE:
    try:
        from xgboost import XGBRegressor
        REGRESSION_MODELS["XGBoost"] = XGBRegressor
    except ImportError:
        pass

if LIGHTGBM_AVAILABLE:
    try:
        from lightgbm import LGBMRegressor
        REGRESSION_MODELS["LightGBM"] = LGBMRegressor
    except ImportError:
        pass


# ── Regression hyperparameter specifications ─────────────────────────────────
REGRESSION_MODEL_PARAMS = {
    "Linear Regression": {},
    "Ridge": {
        "alpha": {"type": "float", "default": 1.0, "min": 0.0001, "max": 100.0},
    },
    "Lasso": {
        "alpha": {"type": "float", "default": 1.0, "min": 0.0001, "max": 100.0},
        "max_iter": {"type": "int", "default": 1000, "min": 100, "max": 5000},
    },
    "ElasticNet": {
        "alpha": {"type": "float", "default": 1.0, "min": 0.0001, "max": 100.0},
        "l1_ratio": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0},
    },
    "Decision Tree": {
        "max_depth": {"type": "int_none", "default": 10, "min": 1, "max": 50},
        "min_samples_split": {"type": "int", "default": 2, "min": 2, "max": 20},
        "min_samples_leaf": {"type": "int", "default": 1, "min": 1, "max": 10},
    },
    "Random Forest": {
        "n_estimators": {"type": "int", "default": 100, "min": 10, "max": 500},
        "max_depth": {"type": "int_none", "default": 10, "min": 1, "max": 50},
        "min_samples_split": {"type": "int", "default": 2, "min": 2, "max": 20},
        "min_samples_leaf": {"type": "int", "default": 1, "min": 1, "max": 10},
        "max_features": {"type": "select", "default": "sqrt", "options": ["sqrt", "log2", "none"]},
    },
    "Gradient Boosting": {
        "n_estimators": {"type": "int", "default": 100, "min": 10, "max": 500},
        "learning_rate": {"type": "float", "default": 0.1, "min": 0.001, "max": 1.0},
        "max_depth": {"type": "int", "default": 3, "min": 1, "max": 10},
        "subsample": {"type": "float", "default": 1.0, "min": 0.1, "max": 1.0},
    },
    "SVR": {
        "C": {"type": "float", "default": 1.0, "min": 0.001, "max": 100.0},
        "kernel": {"type": "select", "default": "rbf", "options": ["rbf", "linear", "poly"]},
        "epsilon": {"type": "float", "default": 0.1, "min": 0.0, "max": 1.0},
    },
    "KNN": {
        "n_neighbors": {"type": "int", "default": 5, "min": 1, "max": 50},
        "weights": {"type": "select", "default": "uniform", "options": ["uniform", "distance"]},
        "metric": {"type": "select", "default": "minkowski", "options": ["minkowski", "euclidean", "manhattan"]},
    },
    "XGBoost": {
        "n_estimators": {"type": "int", "default": 100, "min": 10, "max": 500},
        "learning_rate": {"type": "float", "default": 0.1, "min": 0.001, "max": 1.0},
        "max_depth": {"type": "int", "default": 6, "min": 1, "max": 15},
        "subsample": {"type": "float", "default": 1.0, "min": 0.1, "max": 1.0},
    },
    "LightGBM": {
        "n_estimators": {"type": "int", "default": 100, "min": 10, "max": 500},
        "learning_rate": {"type": "float", "default": 0.1, "min": 0.001, "max": 1.0},
        "max_depth": {"type": "int", "default": -1, "min": -1, "max": 15},
        "num_leaves": {"type": "int", "default": 31, "min": 10, "max": 200},
    },
}


# ── Hyperparameter search grids for optimization ─────────────────────────────
OPTIMIZATION_PARAM_GRIDS = {
    "Logistic Regression": {
        "C": [0.01, 0.1, 1.0, 10.0, 100.0],
        "solver": ["lbfgs", "liblinear"],
        "max_iter": [1000],
    },
    "Random Forest": {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "max_features": ["sqrt", "log2"],
    },
    "Gradient Boosting": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [2, 3, 5],
        "subsample": [0.8, 1.0],
    },
    "AdaBoost": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.5, 1.0],
    },
    "Decision Tree": {
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "criterion": ["gini", "entropy"],
    },
    "KNN": {
        "n_neighbors": [3, 5, 7, 11, 15],
        "weights": ["uniform", "distance"],
        "metric": ["minkowski", "euclidean"],
    },
    "SVM": {
        "C": [0.1, 1.0, 10.0],
        "kernel": ["rbf", "linear"],
        "gamma": ["scale", "auto"],
    },
}
if XGBOOST_AVAILABLE:
    OPTIMIZATION_PARAM_GRIDS["XGBoost"] = {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 5, 7],
        "subsample": [0.8, 1.0],
    }
if LIGHTGBM_AVAILABLE:
    OPTIMIZATION_PARAM_GRIDS["LightGBM"] = {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.05, 0.1],
        "num_leaves": [20, 31, 50],
        "subsample": [0.8, 1.0],
        "verbose": [-1],
    }


# ── Hyperparameter search grids for regression optimization ──────────────────
REGRESSION_OPTIMIZATION_PARAM_GRIDS = {
    "Linear Regression": {},  # no hyperparameters to tune
    "Ridge": {
        "alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
        "solver": ["auto", "svd", "cholesky", "lsqr"],
    },
    "Lasso": {
        "alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
        "max_iter": [1000, 5000],
    },
    "ElasticNet": {
        "alpha": [0.01, 0.1, 1.0, 10.0],
        "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
        "max_iter": [1000, 5000],
    },
    "Random Forest Regressor": {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "max_features": ["sqrt", "log2"],
    },
    "Gradient Boosting Regressor": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [2, 3, 5],
        "subsample": [0.8, 1.0],
    },
    "SVR": {
        "C": [0.1, 1.0, 10.0, 100.0],
        "kernel": ["rbf", "linear"],
        "gamma": ["scale", "auto"],
        "epsilon": [0.01, 0.1, 0.5],
    },
    "KNN Regressor": {
        "n_neighbors": [3, 5, 7, 11, 15],
        "weights": ["uniform", "distance"],
        "metric": ["minkowski", "euclidean"],
    },
    "Decision Tree Regressor": {
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "criterion": ["squared_error", "absolute_error"],
    },
}


# ── Hyperparameter specifications ────────────────────────────────────────────
MODEL_PARAMS = {
    "Logistic Regression": {
        "C": {"type": "float", "default": 1.0, "min": 0.001, "max": 100.0},
        "max_iter": {"type": "int", "default": 1000, "min": 100, "max": 5000},
        "solver": {"type": "select", "default": "lbfgs", "options": ["lbfgs", "liblinear", "saga", "newton-cg"]},
    },
    "Random Forest": {
        "n_estimators": {"type": "int", "default": 100, "min": 10, "max": 500},
        "max_depth": {"type": "int_none", "default": 10, "min": 1, "max": 50},
        "min_samples_split": {"type": "int", "default": 2, "min": 2, "max": 20},
        "min_samples_leaf": {"type": "int", "default": 1, "min": 1, "max": 10},
        "max_features": {"type": "select", "default": "sqrt", "options": ["sqrt", "log2", "none"]},
    },
    "Gradient Boosting": {
        "n_estimators": {"type": "int", "default": 100, "min": 10, "max": 500},
        "learning_rate": {"type": "float", "default": 0.1, "min": 0.001, "max": 1.0},
        "max_depth": {"type": "int", "default": 3, "min": 1, "max": 10},
        "subsample": {"type": "float", "default": 1.0, "min": 0.1, "max": 1.0},
    },
    "AdaBoost": {
        "n_estimators": {"type": "int", "default": 50, "min": 10, "max": 500},
        "learning_rate": {"type": "float", "default": 1.0, "min": 0.001, "max": 2.0},
    },
    "Decision Tree": {
        "max_depth": {"type": "int_none", "default": 10, "min": 1, "max": 50},
        "min_samples_split": {"type": "int", "default": 2, "min": 2, "max": 20},
        "min_samples_leaf": {"type": "int", "default": 1, "min": 1, "max": 10},
        "criterion": {"type": "select", "default": "gini", "options": ["gini", "entropy", "log_loss"]},
    },
    "SVM": {
        "C": {"type": "float", "default": 1.0, "min": 0.001, "max": 100.0},
        "kernel": {"type": "select", "default": "rbf", "options": ["rbf", "linear", "poly", "sigmoid"]},
        "gamma": {"type": "select", "default": "scale", "options": ["scale", "auto"]},
    },
    "KNN": {
        "n_neighbors": {"type": "int", "default": 5, "min": 1, "max": 50},
        "weights": {"type": "select", "default": "uniform", "options": ["uniform", "distance"]},
        "metric": {"type": "select", "default": "minkowski", "options": ["minkowski", "euclidean", "manhattan"]},
    },
    "XGBoost": {
        "n_estimators": {"type": "int", "default": 100, "min": 10, "max": 500},
        "learning_rate": {"type": "float", "default": 0.1, "min": 0.001, "max": 1.0},
        "max_depth": {"type": "int", "default": 6, "min": 1, "max": 15},
        "subsample": {"type": "float", "default": 1.0, "min": 0.1, "max": 1.0},
        "colsample_bytree": {"type": "float", "default": 1.0, "min": 0.1, "max": 1.0},
    },
    "LightGBM": {
        "n_estimators": {"type": "int", "default": 100, "min": 10, "max": 500},
        "learning_rate": {"type": "float", "default": 0.1, "min": 0.001, "max": 1.0},
        "max_depth": {"type": "int", "default": -1, "min": -1, "max": 15},
        "num_leaves": {"type": "int", "default": 31, "min": 10, "max": 200},
        "subsample": {"type": "float", "default": 1.0, "min": 0.1, "max": 1.0},
    },
}


def encode_high_cardinality(X_train, X_test, high_card_cols, strategy, y_train=None):
    """
    Encode high-cardinality categorical columns using target or binary encoding.
    Encoding is fit on X_train/y_train only, then applied to both splits.

    Returns (X_train_out, X_test_out, encoded_col_names, encoder_obj)
    where X_train_out/X_test_out have the original high_card_cols replaced
    by the encoded numeric columns.
    """
    if not high_card_cols or strategy == "drop":
        return X_train, X_test, [], None

    # Fill missing values before encoding
    from sklearn.impute import SimpleImputer as _SI
    imp = _SI(strategy="most_frequent")
    train_hc = pd.DataFrame(
        imp.fit_transform(X_train[high_card_cols]),
        columns=high_card_cols,
        index=X_train.index,
    )
    test_hc = pd.DataFrame(
        imp.transform(X_test[high_card_cols]),
        columns=high_card_cols,
        index=X_test.index,
    )

    if strategy == "binary":
        enc = BinaryEncoder()
        enc.fit(train_hc)
        train_enc = enc.transform(train_hc)
        test_enc = enc.transform(test_hc)
        col_names = enc.get_feature_names_out(high_card_cols).tolist()
    else:  # target
        if SKLEARN_TARGET_ENCODER:
            enc = _SKTargetEncoder()
        else:
            enc = MeanTargetEncoder()
        enc.fit(train_hc, y_train)
        train_enc = enc.transform(train_hc)
        test_enc = enc.transform(test_hc)
        col_names = [f"{c}_te" for c in high_card_cols]

    train_enc_df = pd.DataFrame(train_enc, columns=col_names, index=X_train.index)
    test_enc_df = pd.DataFrame(test_enc, columns=col_names, index=X_test.index)

    X_train_out = pd.concat(
        [X_train.drop(columns=high_card_cols), train_enc_df], axis=1
    )
    X_test_out = pd.concat(
        [X_test.drop(columns=high_card_cols), test_enc_df], axis=1
    )
    return X_train_out, X_test_out, col_names, enc


def build_preprocessor(numeric_cols, categorical_cols, num_strategy="median",
                        cat_strategy="most_frequent", scaler_type="standard",
                        use_knn_imputer=False, knn_neighbors=5):
    """
    Build a ColumnTransformer preprocessor for numeric and categorical columns.
    High-cardinality columns should be pre-encoded via encode_high_cardinality()
    before calling this function; their encoded columns are passed in numeric_cols.
    """
    # Numeric pipeline
    if use_knn_imputer:
        num_imputer = KNNImputer(n_neighbors=knn_neighbors)
    else:
        num_imputer = SimpleImputer(strategy=num_strategy)

    if scaler_type == "standard":
        scaler = StandardScaler()
    elif scaler_type == "minmax":
        scaler = MinMaxScaler()
    elif scaler_type == "robust":
        scaler = RobustScaler()
    else:
        scaler = "passthrough"

    if scaler == "passthrough":
        numeric_pipeline = Pipeline([
            ("imputer", num_imputer),
        ])
    else:
        numeric_pipeline = Pipeline([
            ("imputer", num_imputer),
            ("scaler", scaler),
        ])

    # Categorical pipeline
    if cat_strategy == "drop":
        cat_imputer = SimpleImputer(strategy="most_frequent")
    else:
        cat_imputer = SimpleImputer(strategy=cat_strategy)

    categorical_pipeline = Pipeline([
        ("imputer", cat_imputer),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_pipeline, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_pipeline, categorical_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return preprocessor


def get_feature_names_after_preprocessing(preprocessor, numeric_cols, categorical_cols):
    """
    Extract feature names after preprocessing (handles OHE, binary, and target encoding).
    """
    feature_names = []
    for name, transformer, cols in preprocessor.transformers_:
        if name == "num":
            feature_names.extend(cols)
        elif name == "cat":
            try:
                encoder = transformer.named_steps["encoder"]
                enc_features = encoder.get_feature_names_out(cols).tolist()
                feature_names.extend(enc_features)
            except Exception:
                feature_names.extend(cols)
        elif name == "remainder" and transformer == "passthrough":
            if isinstance(cols, list):
                feature_names.extend(cols)
    return feature_names


def get_model_instance(model_name, params):
    """
    Instantiate a model by name with the given parameters.
    Filters out None values for certain params (like max_depth).
    """
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}")

    model_cls = MODELS[model_name]
    clean_params = {}
    for k, v in params.items():
        if v is not None and v != "none":
            clean_params[k] = v

    # Special handling
    if model_name == "SVM":
        clean_params["probability"] = True

    if model_name in ("XGBoost",) and XGBOOST_AVAILABLE:
        clean_params["eval_metric"] = "logloss"
        clean_params["use_label_encoder"] = False
        clean_params.pop("use_label_encoder", None)

    if model_name == "LightGBM" and LIGHTGBM_AVAILABLE:
        # Suppress verbose output during CV/training
        clean_params.setdefault("verbose", -1)
        # subsample < 1.0 requires bagging_freq > 0 in LightGBM
        if clean_params.get("subsample", 1.0) < 1.0:
            clean_params.setdefault("subsample_freq", 1)

    try:
        model = model_cls(**clean_params)
    except TypeError:
        # Fallback: instantiate with defaults
        model = model_cls()
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained model on the test set.
    Returns a dict with:
      y_pred, y_proba, report (dict), report_text (str),
      confusion_matrix, classes, roc_auc,
      roc_curve (dict with fpr/tpr/thresholds),
      pr_curve (dict with precision/recall/thresholds)
    """
    y_pred = model.predict(X_test)
    classes = model.classes_

    # Probabilities
    y_proba = None
    roc_auc = None
    roc_curve_data = None
    pr_curve_data = None

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)
        if len(classes) == 2:
            y_scores = y_proba[:, 1]
            try:
                roc_auc = roc_auc_score(y_test, y_scores)
                fpr, tpr, roc_thresholds = roc_curve(y_test, y_scores, pos_label=classes[1])
                roc_curve_data = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": roc_thresholds.tolist()}
                precision, recall, pr_thresholds = precision_recall_curve(y_test, y_scores, pos_label=classes[1])
                avg_prec = average_precision_score(y_test, y_scores, pos_label=classes[1])
                pr_curve_data = {
                    "precision": precision.tolist(),
                    "recall": recall.tolist(),
                    "thresholds": pr_thresholds.tolist(),
                    "average_precision": avg_prec,
                }
            except Exception:
                pass
        else:
            try:
                roc_auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
            except Exception:
                pass

    # Classification report
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    report_text = classification_report(y_test, y_pred, zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=classes)

    return {
        "y_pred": y_pred,
        "y_proba": y_proba,
        "report": report_dict,
        "report_text": report_text,
        "confusion_matrix": cm,
        "classes": classes,
        "roc_auc": roc_auc,
        "roc_curve": roc_curve_data,
        "pr_curve": pr_curve_data,
    }


def get_regression_model_instance(model_name, params):
    """Instantiate a regression model with the given parameters."""
    if model_name not in REGRESSION_MODELS:
        raise ValueError(f"Unknown regression model: {model_name}")
    model_cls = REGRESSION_MODELS[model_name]
    clean_params = {k: v for k, v in params.items() if v is not None and v != "none"}
    if model_name == "LightGBM":
        clean_params.setdefault("verbose", -1)
    if model_name == "XGBoost":
        clean_params.pop("use_label_encoder", None)
    try:
        return model_cls(**clean_params)
    except TypeError:
        return model_cls()


def evaluate_model_regression(model, X_test, y_test):
    """
    Evaluate a trained regression model.

    Returns a dict with:
      y_pred, mae, mse, rmse, r2, mape, residuals,
      scatter (dict with actual/predicted lists)
    """
    y_test_arr = np.asarray(y_test, dtype=float)
    y_pred = model.predict(X_test).astype(float)

    mae = float(mean_absolute_error(y_test_arr, y_pred))
    mse = float(mean_squared_error(y_test_arr, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_test_arr, y_pred))

    nonzero = y_test_arr != 0
    if nonzero.sum() > 0:
        mape = float(np.mean(np.abs((y_test_arr[nonzero] - y_pred[nonzero]) / y_test_arr[nonzero])) * 100)
    else:
        mape = None

    residuals = (y_test_arr - y_pred).tolist()

    return {
        "y_pred": y_pred,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "mape": mape,
        "residuals": residuals,
        "scatter": {
            "actual": y_test_arr.tolist(),
            "predicted": y_pred.tolist(),
        },
    }


def run_model_optimization(
    X_train,
    y_train,
    models_to_try=None,
    search_type="random",
    scoring="accuracy",
    cv=5,
    n_iter=20,
    sample_weights=None,
    random_state=42,
    progress_callback=None,
    task_type="classification",
):
    """
    Run GridSearchCV or RandomizedSearchCV across multiple models and their
    hyperparameter grids.

    Parameters
    ----------
    X_train, y_train : array-like
    models_to_try : list of str | None
        Model names to include (defaults to all available).
    search_type : 'grid' | 'random'
    scoring : str
        scikit-learn scoring string.
    cv : int
        Number of cross-validation folds.
    n_iter : int
        Iterations for RandomizedSearchCV (ignored for GridSearchCV).
    sample_weights : array-like | None
    random_state : int
    progress_callback : callable(current, total, model_name) | None
        Called after each model finishes so callers can update a progress bar.
    task_type : 'classification' | 'regression'

    Returns
    -------
    list of dict, sorted by mean_cv_score descending.
    Each dict: model_name, best_params, best_score, cv_results_df, best_estimator
    """
    from sklearn.model_selection import StratifiedKFold, KFold

    is_regression = task_type == "regression"
    param_grids = REGRESSION_OPTIMIZATION_PARAM_GRIDS if is_regression else OPTIMIZATION_PARAM_GRIDS
    model_registry = REGRESSION_MODELS if is_regression else MODELS
    get_model_fn = get_regression_model_instance if is_regression else get_model_instance

    if models_to_try is None:
        models_to_try = list(param_grids.keys())

    # Filter to models that are actually available
    models_to_try = [m for m in models_to_try if m in model_registry and m in param_grids]

    if is_regression:
        cv_splitter = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    else:
        cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    fit_params = {}
    if sample_weights is not None and not is_regression:
        fit_params["sample_weight"] = sample_weights

    results = []
    total = len(models_to_try)

    for idx, model_name in enumerate(models_to_try):
        if progress_callback:
            progress_callback(idx, total, model_name)

        try:
            base_model = get_model_fn(model_name, {})
            param_grid = param_grids[model_name]

            # Models with no hyperparameters (e.g. Linear Regression) — just CV score them
            if not param_grid:
                from sklearn.model_selection import cross_validate
                cv_scores = cross_validate(base_model, X_train, y_train, cv=cv_splitter, scoring=scoring)
                base_model.fit(X_train, y_train)
                mean_score = float(cv_scores["test_score"].mean())
                results.append({
                    "model_name": model_name,
                    "best_params": {},
                    "best_score": mean_score,
                    "cv_results_df": pd.DataFrame({"params": [{}], "mean_test_score": [mean_score],
                                                   "std_test_score": [cv_scores["test_score"].std()],
                                                   "rank_test_score": [1]}),
                    "best_estimator": base_model,
                })
                if progress_callback:
                    progress_callback(idx + 1, total, model_name)
                continue

            if search_type == "grid":
                searcher = GridSearchCV(
                    base_model,
                    param_grid,
                    cv=cv_splitter,
                    scoring=scoring,
                    n_jobs=-1 if model_name != "LightGBM" else 1,
                    refit=True,
                    error_score="raise",
                )
            else:
                # Compute a sensible n_iter: cap at total grid size
                grid_size = 1
                for vals in param_grid.values():
                    grid_size *= len(vals)
                effective_n_iter = min(n_iter, grid_size)
                searcher = RandomizedSearchCV(
                    base_model,
                    param_grid,
                    n_iter=effective_n_iter,
                    cv=cv_splitter,
                    scoring=scoring,
                    n_jobs=-1 if model_name != "LightGBM" else 1,
                    refit=True,
                    random_state=random_state,
                    error_score="raise",
                )

            if fit_params and hasattr(base_model, "fit"):
                # Pass sample_weight only if the model supports it
                try:
                    searcher.fit(X_train, y_train, **{f"estimator__{k}": v for k, v in fit_params.items()})
                except TypeError:
                    searcher.fit(X_train, y_train)
            else:
                searcher.fit(X_train, y_train)

            cv_res_df = pd.DataFrame(searcher.cv_results_)[
                ["params", "mean_test_score", "std_test_score", "rank_test_score"]
            ].sort_values("rank_test_score")

            results.append({
                "model_name": model_name,
                "best_params": searcher.best_params_,
                "best_score": float(searcher.best_score_),
                "cv_results_df": cv_res_df,
                "best_estimator": searcher.best_estimator_,
            })

        except Exception as exc:
            results.append({
                "model_name": model_name,
                "best_params": {},
                "best_score": float("nan"),
                "cv_results_df": pd.DataFrame(),
                "best_estimator": None,
                "error": str(exc),
            })

    if progress_callback:
        progress_callback(total, total, "Done")

    results.sort(key=lambda r: r["best_score"] if not np.isnan(r["best_score"]) else -1, reverse=True)
    return results
