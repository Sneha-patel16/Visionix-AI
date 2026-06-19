import streamlit as st
import numpy as np
import pandas as pd
from modules.agent_core import detect_column_types


def get_safe_numeric_columns(df):
    numeric_cols = []

    for col in df.columns:
        series = df[col]

        if pd.api.types.is_bool_dtype(series):
            continue

        if pd.api.types.is_numeric_dtype(series):
            if series.nunique(dropna=True) > 1:
                numeric_cols.append(col)

    return numeric_cols


def generate_agentic_insights(df):
    insights = []

    numeric_cols = get_safe_numeric_columns(df)

    try:
        numeric_detected, categorical_cols, date_cols, id_cols = detect_column_types(df)
    except Exception:
        categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()
        date_cols = []
        id_cols = []

    categorical_cols = [
        col for col in categorical_cols
        if col not in id_cols and df[col].nunique(dropna=True) > 1
    ]

    if numeric_cols:
        means = df[numeric_cols].mean(numeric_only=True)

        highest = means.idxmax()
        lowest = means.idxmin()

        insights.append(f"'{highest}' has the highest average numeric value.")
        insights.append(f"'{lowest}' has the lowest average numeric value.")

    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr().abs()
        np.fill_diagonal(corr.values, 0)

        stacked = corr.stack()

        if not stacked.empty:
            pair = stacked.idxmax()
            value = stacked.max()

            insights.append(
                f"Strongest numeric relationship is between '{pair[0]}' and '{pair[1]}' with correlation {value:.2f}."
            )

    for col in numeric_cols:
        try:
            series = pd.to_numeric(df[col], errors="coerce").dropna()

            if series.empty:
                continue

            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1

            if iqr == 0:
                continue

            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            outlier_count = ((series < lower) | (series > upper)).sum()

            if outlier_count > 0:
                insights.append(f"'{col}' contains {outlier_count} possible outliers.")

        except Exception:
            continue

    if categorical_cols:
        cat = categorical_cols[0]
        top_value = df[cat].value_counts().idxmax()
        top_count = df[cat].value_counts().max()

        insights.append(
            f"Most frequent category in '{cat}' is '{top_value}' with {top_count} records."
        )

    if date_cols:
        insights.append(
            f"Datetime pattern detected using: {', '.join(date_cols[:3])}."
        )

    if not insights:
        insights.append("Dataset looks stable and ready for business analysis.")

    return insights


def show_insight_engine(df):
    st.header("🤖 Agentic Insight Engine")
    st.caption("Insights are generated from cleaned and engineered data using safe generic rules.")

    insights = generate_agentic_insights(df)

    for insight in insights:
        st.write(f"💡 {insight}")

    st.subheader("Recommended Actions")
    st.write("✅ Use high-correlation features carefully to avoid redundancy.")
    st.write("✅ Investigate remaining outliers before final decision-making.")
    st.write("✅ Train multiple ML models in ML Studio and select the best performer.")
    st.write("✅ Use Dashboard Studio for stakeholder-ready business visuals.")