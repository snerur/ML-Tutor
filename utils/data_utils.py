"""
Data utility functions for ML Fairness Studio.
"""

import pandas as pd
import numpy as np
import io


def load_data(uploaded_file):
    """
    Load data from a CSV or Excel uploaded file.
    Returns a DataFrame or raises an exception on failure.
    """
    filename = uploaded_file.name.lower()
    if filename.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif filename.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file)
    else:
        raise ValueError(f"Unsupported file type: {uploaded_file.name}. Please upload a CSV or Excel file.")
    return df


def get_column_types(df):
    """
    Analyze column types and return a dict with 'numeric', 'categorical', 'datetime' lists.
    """
    numeric_cols = []
    categorical_cols = []
    datetime_cols = []

    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            datetime_cols.append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            # If there are very few unique values relative to total rows, treat as categorical
            unique_ratio = df[col].nunique() / max(len(df), 1)
            if df[col].nunique() <= 10 and unique_ratio < 0.05:
                categorical_cols.append(col)
            else:
                numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    return {
        "numeric": numeric_cols,
        "categorical": categorical_cols,
        "datetime": datetime_cols,
    }


def analyze_missing(df):
    """
    Returns a DataFrame with columns ['Count', 'Percentage'] for missing values per column.
    Only includes columns that have at least one missing value.
    """
    missing_count = df.isnull().sum()
    missing_pct = (missing_count / len(df) * 100).round(2)
    result = pd.DataFrame({
        "Count": missing_count,
        "Percentage": missing_pct,
    })
    result = result[result["Count"] > 0].sort_values("Count", ascending=False)
    return result


def detect_outliers_iqr(df, col):
    """
    Detect outliers using IQR method for a numeric column.
    Returns (outliers_df, lower_bound, upper_bound).
    """
    series = df[col].dropna()
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    return outliers, lower_bound, upper_bound


def compute_class_distribution(df, target_col):
    """
    Returns a DataFrame with Count and Percentage for each class of the target column.
    """
    counts = df[target_col].value_counts()
    pct = (counts / counts.sum() * 100).round(2)
    result = pd.DataFrame({
        "Count": counts,
        "Percentage": pct,
    })
    result.index.name = target_col
    return result


def detect_task_type(df, target_col, max_classes_for_classification=20):
    """
    Auto-detect whether the target variable represents a classification or regression problem.

    Rules (in order):
    1. Non-numeric (object, category) → classification
    2. Boolean / 2 unique values → classification
    3. Numeric with ≤ max_classes distinct integer values → classification
    4. Numeric float with many distinct values → regression
    5. Numeric integer where unique_ratio > 5% → regression

    Returns
    -------
    'classification' or 'regression'
    """
    series = df[target_col].dropna()

    # Non-numeric → always classification
    if not pd.api.types.is_numeric_dtype(series):
        return "classification"

    n_unique = series.nunique()

    # Binary or very-few-classes → classification
    if n_unique <= 2:
        return "classification"

    if n_unique <= max_classes_for_classification:
        # Only call it classification if values look like labels (integers, no decimals)
        if pd.api.types.is_integer_dtype(series):
            return "classification"
        # Float but all values are whole numbers (e.g. 0.0, 1.0, 2.0)
        if (series % 1 == 0).all():
            return "classification"

    # Float with many distinct values → regression
    if pd.api.types.is_float_dtype(series) and n_unique > max_classes_for_classification:
        return "regression"

    # Integer with many unique values relative to dataset size → regression
    n_total = len(series)
    if n_unique / n_total > 0.05 and n_unique > max_classes_for_classification:
        return "regression"

    return "classification"


def suggest_protected_attributes(df):
    """
    Returns a list of column names likely to be protected attributes,
    based on common keywords.
    """
    keywords = [
        "gender", "sex", "age", "race", "ethnicity", "nationality",
        "religion", "disability", "marital", "income", "education",
        "national_origin", "color", "origin", "birth", "citizen",
        "language", "caste", "tribe", "veteran", "pregnancy",
    ]
    suggestions = []
    for col in df.columns:
        col_lower = col.lower()
        for kw in keywords:
            if kw in col_lower:
                suggestions.append(col)
                break
    return suggestions


def generate_adult_income_dataset(n_samples=1000, random_state=42):
    """
    Generate a synthetic adult income dataset with demographic features and introduced bias.
    """
    rng = np.random.RandomState(random_state)

    age = rng.randint(18, 70, n_samples)
    gender = rng.choice(["Male", "Female"], n_samples, p=[0.55, 0.45])
    race = rng.choice(["White", "Black", "Asian", "Hispanic", "Other"], n_samples,
                      p=[0.60, 0.15, 0.12, 0.10, 0.03])
    education_years = rng.randint(8, 20, n_samples)
    hours_per_week = rng.randint(10, 80, n_samples)
    occupation = rng.choice(
        ["Tech", "Management", "Sales", "Service", "Clerical", "Craft", "Transport"],
        n_samples
    )
    marital_status = rng.choice(
        ["Married", "Single", "Divorced", "Widowed"], n_samples, p=[0.45, 0.35, 0.15, 0.05]
    )
    capital_gain = rng.exponential(500, n_samples).astype(int)
    capital_loss = rng.exponential(100, n_samples).astype(int)

    # Introduce bias: income >50K is more likely for males, whites, higher education
    log_odds = (
        -3.0
        + 0.03 * (age - 18)
        + 0.20 * education_years
        + 0.01 * hours_per_week
        + 0.5 * (gender == "Male").astype(float)
        + 0.3 * (race == "White").astype(float)
        + 0.2 * (race == "Asian").astype(float)
        - 0.2 * (race == "Black").astype(float)
        + 0.00001 * capital_gain
    )
    prob_high_income = 1 / (1 + np.exp(-log_odds))
    income = (rng.rand(n_samples) < prob_high_income).astype(int)
    income_label = np.where(income == 1, ">50K", "<=50K")

    df = pd.DataFrame({
        "age": age,
        "gender": gender,
        "race": race,
        "education_years": education_years,
        "hours_per_week": hours_per_week,
        "occupation": occupation,
        "marital_status": marital_status,
        "capital_gain": capital_gain,
        "capital_loss": capital_loss,
        "income": income_label,
    })
    return df


def generate_credit_risk_dataset(n_samples=1000, random_state=42):
    """
    Generate a synthetic credit risk dataset with demographic features and introduced bias.
    """
    rng = np.random.RandomState(random_state)

    age = rng.randint(20, 75, n_samples)
    gender = rng.choice(["Male", "Female"], n_samples, p=[0.52, 0.48])
    race = rng.choice(["White", "Black", "Asian", "Hispanic", "Other"], n_samples,
                      p=[0.58, 0.17, 0.13, 0.09, 0.03])
    credit_score = rng.randint(300, 850, n_samples)
    loan_amount = rng.randint(1000, 100000, n_samples)
    income = rng.randint(15000, 200000, n_samples)
    employment_years = rng.randint(0, 40, n_samples)
    debt_to_income = (loan_amount / income * 100).round(2)
    num_credit_lines = rng.randint(0, 20, n_samples)
    previous_defaults = rng.randint(0, 5, n_samples)

    # Introduce bias: default risk is higher for lower credit scores, lower income
    # and slightly biased by race
    log_odds = (
        2.0
        - 0.005 * credit_score
        - 0.00001 * income
        + 0.01 * debt_to_income
        + 0.3 * previous_defaults
        - 0.02 * employment_years
        + 0.3 * (race == "Black").astype(float)
        + 0.2 * (race == "Hispanic").astype(float)
        - 0.1 * (gender == "Male").astype(float)
    )
    prob_default = 1 / (1 + np.exp(-log_odds))
    default = (rng.rand(n_samples) < prob_default).astype(int)
    default_label = np.where(default == 1, "Default", "No Default")

    df = pd.DataFrame({
        "age": age,
        "gender": gender,
        "race": race,
        "credit_score": credit_score,
        "loan_amount": loan_amount,
        "income": income,
        "employment_years": employment_years,
        "debt_to_income": debt_to_income,
        "num_credit_lines": num_credit_lines,
        "previous_defaults": previous_defaults,
        "default": default_label,
    })
    return df


def generate_compas_dataset(n_samples=1000, random_state=42):
    """
    Generate a synthetic COMPAS-like recidivism dataset.

    Mimics the structure of the ProPublica COMPAS dataset (Broward County, FL).
    Reproduces the documented racial disparities: the COMPAS score systematically
    over-predicted recidivism for Black defendants and under-predicted for white
    defendants even when controlling for prior record (ProPublica, 2016).

    Features
    --------
    age, age_cat, sex, race, juv_fel_count, juv_misd_count, juv_other_count,
    priors_count, c_charge_degree, days_b_screening_arrest, decile_score

    Target
    ------
    two_year_recid : 'Yes' / 'No'
    """
    rng = np.random.RandomState(random_state)

    race = rng.choice(
        ["African-American", "Caucasian", "Hispanic", "Other", "Asian", "Native American"],
        n_samples,
        p=[0.51, 0.34, 0.08, 0.04, 0.02, 0.01],
    )
    sex = rng.choice(["Male", "Female"], n_samples, p=[0.81, 0.19])
    age = rng.randint(18, 70, n_samples)
    age_cat = np.where(
        age < 25, "Less than 25",
        np.where(age <= 45, "25 - 45", "Greater than 45"),
    )

    priors_count = np.clip(rng.negative_binomial(1, 0.3, n_samples), 0, 38)
    juv_fel_count = np.clip(rng.negative_binomial(1, 0.85, n_samples), 0, 20)
    juv_misd_count = np.clip(rng.negative_binomial(1, 0.80, n_samples), 0, 15)
    juv_other_count = np.clip(rng.negative_binomial(1, 0.90, n_samples), 0, 10)
    c_charge_degree = rng.choice(["F", "M"], n_samples, p=[0.42, 0.58])
    days_b_screening_arrest = rng.randint(-30, 30, n_samples)

    # COMPAS decile score (1–10): includes documented racial over-scoring of
    # African-American defendants independent of criminal history
    score_base = (
        2.0
        + 0.15 * priors_count
        + 0.5 * juv_fel_count
        + 0.3 * juv_misd_count
        + (age < 25).astype(float) * 1.5
        + (c_charge_degree == "F").astype(float) * 0.5
        + rng.normal(0, 1.5, n_samples)
        + (race == "African-American").astype(float) * 1.2   # bias component
        + (race == "Hispanic").astype(float) * 0.3
    )
    decile_score = np.clip(np.round(score_base).astype(int), 1, 10)

    # Actual two-year recidivism (ground truth).
    # Racial disparities in actual recidivism reflect documented systemic factors
    # (e.g. concentrated policing, fewer diversion resources, higher re-arrest rates)
    # — not inherent differences. This mirrors real ProPublica COMPAS findings.
    log_odds_recid = (
        -0.5
        + 0.12 * priors_count
        + 0.4 * juv_fel_count
        + 0.2 * juv_misd_count
        + (age < 25).astype(float) * 0.8
        + (c_charge_degree == "F").astype(float) * 0.3
        - 0.01 * np.clip(age - 25, 0, None)
        + rng.normal(0, 0.5, n_samples)
        + (race == "African-American").astype(float) * 0.55   # systemic disparity
        + (race == "Hispanic").astype(float) * 0.30
        - (sex == "Female").astype(float) * 0.30              # lower female recidivism
    )
    prob_recid = 1 / (1 + np.exp(-log_odds_recid))
    two_year_recid = np.where(rng.rand(n_samples) < prob_recid, "Yes", "No")

    df = pd.DataFrame({
        "age": age,
        "age_cat": age_cat,
        "sex": sex,
        "race": race,
        "juv_fel_count": juv_fel_count,
        "juv_misd_count": juv_misd_count,
        "juv_other_count": juv_other_count,
        "priors_count": priors_count,
        "c_charge_degree": c_charge_degree,
        "days_b_screening_arrest": days_b_screening_arrest,
        "decile_score": decile_score,
        "two_year_recid": two_year_recid,
    })
    return df
