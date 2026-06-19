import streamlit as st
import pandas as pd
import numpy as np


def validate_dataset(df):
    issues = []
    warnings = []
    recommendations = []

    # -------------------------
    # Missing Values
    # -------------------------
    for col in df.columns:

        missing_pct = (df[col].isnull().sum() / len(df)) * 100

        if missing_pct > 50:
            issues.append(
                f"{col} has {missing_pct:.1f}% missing values."
            )

            recommendations.append(
                f"Consider removing or heavily cleaning {col}."
            )

    # -------------------------
    # Constant Columns
    # -------------------------
    for col in df.columns:

        if df[col].nunique() <= 1:

            warnings.append(
                f"{col} contains only one unique value."
            )

            recommendations.append(
                f"Remove {col} because it provides no predictive value."
            )

    # -------------------------
    # Duplicate Rows
    # -------------------------
    duplicates = df.duplicated().sum()

    if duplicates > 0:

        warnings.append(
            f"{duplicates} duplicate rows detected."
        )

        recommendations.append(
            "Remove duplicate rows before analysis."
        )

    # -------------------------
    # Numeric Validation
    # -------------------------
    numeric_cols = df.select_dtypes(include=np.number).columns

    for col in numeric_cols:

        if "age" in col.lower():

            invalid = (df[col] < 0).sum()

            if invalid > 0:
                issues.append(
                    f"{col} contains {invalid} negative ages."
                )

        if "salary" in col.lower():

            invalid = (df[col] < 0).sum()

            if invalid > 0:
                issues.append(
                    f"{col} contains {invalid} negative salaries."
                )

    # -------------------------
    # ID Columns
    # -------------------------
    for col in df.columns:

        unique_ratio = df[col].nunique() / len(df)

        if unique_ratio > 0.95:

            warnings.append(
                f"{col} appears to be an ID-like column."
            )

            recommendations.append(
                f"Consider excluding {col} from ML training."
            )

    return {
        "issues": issues,
        "warnings": warnings,
        "recommendations": recommendations
    }


def show_validation_agent(df):

    st.header("🛡️ Data Validation Agent")

    results = validate_dataset(df)

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Critical Issues",
        len(results["issues"])
    )

    c2.metric(
        "Warnings",
        len(results["warnings"])
    )

    c3.metric(
        "Recommendations",
        len(results["recommendations"])
    )

    st.subheader("🚨 Critical Issues")

    if results["issues"]:
        for item in results["issues"]:
            st.error(item)
    else:
        st.success("No critical issues found.")

    st.subheader("⚠️ Warnings")

    if results["warnings"]:
        for item in results["warnings"]:
            st.warning(item)
    else:
        st.success("No warnings found.")

    st.subheader("💡 Recommendations")

    if results["recommendations"]:
        for item in results["recommendations"]:
            st.info(item)
    else:
        st.success("Dataset is ready for processing.")