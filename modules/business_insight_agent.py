import streamlit as st
import numpy as np
import pandas as pd


def detect_outliers(df, col):
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1

    if iqr == 0:
        return 0

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    return int(((df[col] < lower) | (df[col] > upper)).sum())


def generate_automated_insights(df):
    insights = []

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    missing_total = df.isnull().sum().sum()
    duplicate_total = df.duplicated().sum()

    if missing_total == 0:
        insights.append("Data quality is strong: no missing values found.")
    else:
        insights.append(f"Data contains {missing_total} missing values.")

    if duplicate_total == 0:
        insights.append("No duplicate rows found.")
    else:
        insights.append(f"{duplicate_total} duplicate rows found.")

    for col in numeric_cols[:20]:
        outliers = detect_outliers(df, col)
        if outliers > 0:
            insights.append(f"Column '{col}' has {outliers} possible outliers.")

    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr().abs()
        np.fill_diagonal(corr.values, 0)

        pair = corr.stack().idxmax()
        value = corr.stack().max()

        if value >= 0.80:
            insights.append(
                f"Strong relationship detected between '{pair[0]}' and '{pair[1]}' with correlation {value:.2f}."
            )
        elif value >= 0.50:
            insights.append(
                f"Moderate relationship detected between '{pair[0]}' and '{pair[1]}' with correlation {value:.2f}."
            )

    for col in categorical_cols[:10]:
        top_ratio = df[col].value_counts(normalize=True).iloc[0]
        if top_ratio > 0.80:
            insights.append(
                f"Column '{col}' is imbalanced: one category covers {top_ratio:.1%} of records."
            )

    if "target_col" in st.session_state and st.session_state["target_col"] in df.columns:
        target = st.session_state["target_col"]

        if target in numeric_cols:
            correlations = df[numeric_cols].corr()[target].abs().sort_values(ascending=False)
            correlations = correlations.drop(target, errors="ignore")

            if not correlations.empty:
                top_feature = correlations.index[0]
                insights.append(
                    f"Feature '{top_feature}' has the strongest numeric relationship with target '{target}'."
                )

        else:
            target_balance = df[target].value_counts(normalize=True)
            if len(target_balance) > 1 and target_balance.iloc[0] > 0.80:
                insights.append(
                    f"Target '{target}' is imbalanced. Consider class balancing techniques."
                )

    if "model_results" in st.session_state:
        problem_type = st.session_state.get("problem_type")

        if problem_type == "classification":
            best = st.session_state["model_results"]["F1 Score"].max()
            insights.append(f"Best classification F1 score achieved: {best:.2f}.")
        else:
            best = st.session_state["model_results"]["R2 Score"].max()
            insights.append(f"Best regression R² score achieved: {best:.2f}.")

    return insights


def show_business_insight_agent(df):
    st.header("🤖 Automated Business Insight Agent")
    st.caption("Universal automated insights from data quality, statistics, correlations, target behavior, and ML results.")

    insights = generate_automated_insights(df)

    for insight in insights:
        st.write(f"💡 {insight}")

    st.subheader("Recommended Actions")
    st.write("✅ Remove leakage or ID-like features before final deployment.")
    st.write("✅ Investigate highly correlated features.")
    st.write("✅ Check outliers before business decisions.")
    st.write("✅ Use XAI Agent to understand model behavior.")