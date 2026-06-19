import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from modules.agent_core import detect_column_types


def make_unique_columns(df):
    df = df.copy()
    counts = {}
    new_cols = []

    for col in df.columns:
        if col not in counts:
            counts[col] = 0
            new_cols.append(col)
        else:
            counts[col] += 1
            new_cols.append(f"{col}_{counts[col]}")

    df.columns = new_cols
    return df


def feature_engineering_agent(df):
    engineered_df = make_unique_columns(df)
    log = []

    numeric_cols, categorical_cols, date_cols, id_cols = detect_column_types(engineered_df)

    # ID-like columns: keep original for report/dashboard, but do not engineer them.
    if id_cols:
        log.append(f"Detected ID-like columns and skipped feature engineering: {', '.join(id_cols)}.")

    # Date features only from true datetime-like columns
    for col in date_cols:
        converted = pd.to_datetime(engineered_df[col], errors="coerce")

        engineered_df[col + "_year"] = converted.dt.year
        engineered_df[col + "_month"] = converted.dt.month
        engineered_df[col + "_day"] = converted.dt.day
        engineered_df[col + "_weekday"] = converted.dt.weekday

        log.append(f"Extracted year, month, day, weekday from datetime column '{col}'.")

    # Categorical encoding
    for col in categorical_cols:
        if col in id_cols:
            continue

        unique_count = engineered_df[col].nunique(dropna=True)

        if unique_count <= 1:
            continue

        if unique_count <= 10:
            dummies = pd.get_dummies(engineered_df[col], prefix=col, drop_first=True)
            engineered_df = pd.concat([engineered_df, dummies], axis=1)
            log.append(f"One-hot encoding applied on low-cardinality categorical column '{col}'.")

        elif unique_count <= 50:
            le = LabelEncoder()
            engineered_df[col + "_encoded"] = le.fit_transform(engineered_df[col].astype(str))
            log.append(f"Label encoding applied on medium-cardinality categorical column '{col}'.")

        else:
            freq = engineered_df[col].value_counts()
            engineered_df[col + "_freq"] = engineered_df[col].map(freq)
            log.append(f"Frequency encoding applied on high-cardinality categorical column '{col}'.")

    # Numeric transformations only on original numeric columns, not engineered date parts/scaled/log columns
    skip_suffixes = (
        "_year", "_month", "_day", "_weekday",
        "_scaled", "_robust", "_log", "_encoded", "_freq"
    )

    base_numeric_cols = [
        col for col in numeric_cols
        if not col.endswith(skip_suffixes) and col not in id_cols
    ]

    for col in base_numeric_cols:
        if engineered_df[col].nunique(dropna=True) <= 1:
            continue

        skewness = engineered_df[col].skew()

        if abs(skewness) > 1:
            engineered_df[col + "_log"] = np.log1p(np.abs(engineered_df[col]))
            log.append(f"Log transform applied on skewed numeric column '{col}'.")

        q1 = engineered_df[col].quantile(0.25)
        q3 = engineered_df[col].quantile(0.75)
        iqr = q3 - q1

        if iqr == 0:
            continue

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outlier_count = ((engineered_df[col] < lower) | (engineered_df[col] > upper)).sum()

        if outlier_count > 0:
            scaler = RobustScaler()
            engineered_df[col + "_robust"] = scaler.fit_transform(engineered_df[[col]])
            log.append(f"Robust scaling applied on '{col}' because outliers were detected.")
        else:
            scaler = StandardScaler()
            engineered_df[col + "_scaled"] = scaler.fit_transform(engineered_df[[col]])
            log.append(f"Standard scaling applied on numeric column '{col}'.")

    engineered_df = make_unique_columns(engineered_df)

    return engineered_df, log