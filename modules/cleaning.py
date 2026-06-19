import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import zscore

def clean_data_agent(df):
    clean_df = df.copy()
    log = []

    clean_df.columns = (
        clean_df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )
    log.append("Standardized column names using strip, lowercase, and underscore formatting.")

    before = clean_df.shape[0]
    clean_df = clean_df.drop_duplicates()
    removed = before - clean_df.shape[0]
    log.append(f"Removed {removed} duplicate rows using drop_duplicates().")

    for col in clean_df.columns:
        missing_count = clean_df[col].isnull().sum()

        if missing_count > 0:
            missing_ratio = missing_count / len(clean_df)

            if missing_ratio > 0.40:
                clean_df = clean_df.dropna(subset=[col])
                log.append(f"Column '{col}' had more than 40% missing values, applied dropna on this column.")

            elif clean_df[col].dtype == "object":
                mode_value = clean_df[col].mode()
                fill_value = mode_value[0] if not mode_value.empty else "unknown"
                clean_df[col] = clean_df[col].fillna(fill_value)
                log.append(f"Filled missing categorical values in '{col}' using mode.")

            else:
                skewness = clean_df[col].skew()

                if abs(skewness) > 1:
                    clean_df[col] = clean_df[col].fillna(clean_df[col].median())
                    log.append(f"Filled missing numeric values in '{col}' using median because skewness was high.")
                else:
                    clean_df[col] = clean_df[col].fillna(clean_df[col].mean())
                    log.append(f"Filled missing numeric values in '{col}' using mean because distribution was balanced.")

    numeric_cols = clean_df.select_dtypes(include=np.number).columns.tolist()

    for col in numeric_cols:
        q1 = clean_df[col].quantile(0.25)
        q3 = clean_df[col].quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outlier_count = ((clean_df[col] < lower) | (clean_df[col] > upper)).sum()

        if outlier_count > 0:
            clean_df[col] = np.where(clean_df[col] < lower, lower, clean_df[col])
            clean_df[col] = np.where(clean_df[col] > upper, upper, clean_df[col])
            log.append(f"Capped {outlier_count} outliers in '{col}' using IQR winsorization.")

    for col in clean_df.columns:
        try:
            converted = pd.to_datetime(clean_df[col], errors="coerce")
            if converted.notna().sum() > len(clean_df) * 0.7:
                clean_df[col + "_year"] = converted.dt.year
                clean_df[col + "_month"] = converted.dt.month
                clean_df[col + "_day"] = converted.dt.day
                log.append(f"Extracted year, month, and day features from datetime column '{col}'.")
        except:
            pass

    return clean_df, log


def show_cleaning_studio(raw_df, cleaned_df, cleaning_log):
    st.header("🧹 Cleaning Studio")

    c1, c2, c3 = st.columns(3)
    c1.metric("Original Rows", raw_df.shape[0])
    c2.metric("Cleaned Rows", cleaned_df.shape[0])
    c3.metric("Removed Rows", raw_df.shape[0] - cleaned_df.shape[0])

    st.subheader("Cleaning Techniques Applied")
    techniques = [
        "Column standardization",
        "drop_duplicates()",
        "fillna(mean)",
        "fillna(median)",
        "fillna(mode)",
        "dropna()",
        "IQR outlier capping",
        "Datetime feature extraction"
    ]

    for tech in techniques:
        st.write(f"✅ {tech}")

    st.subheader("Agent Cleaning Log")
    for item in cleaning_log:
        st.write(f"🤖 {item}")

    st.subheader("Cleaned Dataset")
    st.dataframe(cleaned_df.head(30), width="stretch")

    csv = cleaned_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇ Download Cleaned Dataset", csv, "visionix_cleaned_dataset.csv", "text/csv")