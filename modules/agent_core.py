import pandas as pd
import numpy as np


def normalize_col_name(col):
    return str(col).strip().lower().replace(" ", "_").replace("-", "_")


def is_id_like(df, col):
    name = normalize_col_name(col)
    unique_ratio = df[col].nunique(dropna=True) / max(len(df), 1)

    id_words = ["id", "uuid", "code", "number", "no", "serial"]

    if any(word == name or name.endswith("_" + word) or name.startswith(word + "_") for word in id_words):
        return True

    if unique_ratio > 0.95:
        return True

    return False


def is_datetime_like(df, col):
    name = normalize_col_name(col)

    date_keywords = ["date", "time", "timestamp", "created", "updated", "year", "month"]

    if not any(word in name for word in date_keywords):
        return False

    if pd.api.types.is_numeric_dtype(df[col]):
        return False

    converted = pd.to_datetime(df[col], errors="coerce")

    valid_ratio = converted.notna().sum() / max(len(df), 1)

    return valid_ratio > 0.70


def detect_column_types(df):
    numeric_cols = []
    categorical_cols = []
    date_cols = []
    id_cols = []

    for col in df.columns:
        if is_id_like(df, col):
            id_cols.append(col)
            continue

        if is_datetime_like(df, col):
            date_cols.append(col)
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    return numeric_cols, categorical_cols, date_cols, id_cols


def data_quality_score(df):
    total_cells = df.shape[0] * df.shape[1]
    missing = df.isnull().sum().sum()
    duplicate = df.duplicated().sum()

    missing_score = 100 - ((missing / total_cells) * 100 if total_cells else 0)
    duplicate_score = 100 - ((duplicate / len(df)) * 100 if len(df) else 0)

    return round((missing_score * 0.6) + (duplicate_score * 0.4), 2)


def suggest_target_column(df):
    numeric_cols, categorical_cols, date_cols, id_cols = detect_column_types(df)

    forbidden = set(id_cols + date_cols)

    target_priority = {
        "classification": [
            "target", "label", "class", "status", "outcome", "result",
            "churn", "fraud", "default", "returned", "approved",
            "risk", "disease", "diagnosis", "pass", "fail"
        ],
        "regression": [
            "sales", "profit", "revenue", "price", "amount", "cost",
            "score", "rating", "income", "salary", "value", "total",
            "quantity", "duration", "age"
        ]
    }

    candidates = []

    for col in df.columns:
        if col in forbidden:
            continue

        name = normalize_col_name(col)
        unique_count = df[col].nunique(dropna=True)
        unique_ratio = unique_count / max(len(df), 1)

        if unique_count <= 1:
            continue

        if unique_ratio > 0.90:
            continue

        score = 0

        for word in target_priority["classification"]:
            if word in name:
                score += 40

        for word in target_priority["regression"]:
            if word in name:
                score += 35

        if pd.api.types.is_numeric_dtype(df[col]):
            if unique_count > 20:
                score += 20
            else:
                score += 8
        else:
            if 2 <= unique_count <= 20:
                score += 25
            elif unique_count <= 50:
                score += 10

        if any(x in name for x in ["channel", "type", "category", "segment", "region", "city", "state", "country"]):
            score -= 20

        candidates.append((col, score))

    if not candidates:
        usable_cols = [c for c in df.columns if c not in forbidden]
        return usable_cols[-1] if usable_cols else df.columns[-1]

    candidates = sorted(candidates, key=lambda x: x[1], reverse=True)

    return candidates[0][0]


def detect_problem_type(df, target_col):
    y = df[target_col]

    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]

    unique_count = y.nunique(dropna=True)

    if pd.api.types.is_numeric_dtype(y):
        if unique_count <= 10:
            return "classification"
        return "regression"

    return "classification"