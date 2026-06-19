import streamlit as st
import pandas as pd
import numpy as np


def is_numeric_column(df, col):
    return pd.api.types.is_numeric_dtype(df[col])


def safe_round(value, digits=2):
    try:
        if pd.isna(value):
            return None
        return round(float(value), digits)
    except Exception:
        return value


def detect_kpi_columns(df):
    cols = df.columns.tolist()

    kpi_keywords = {
        "revenue_sales": ["revenue", "sales", "sale_amount", "amount", "income", "earning", "turnover"],
        "profit": ["profit", "margin", "gain"],
        "cost": ["cost", "expense", "spend"],
        "customer": ["customer", "client", "user", "buyer"],
        "order": ["order", "invoice", "transaction", "purchase"],
        "quantity": ["quantity", "qty", "units", "volume"],
        "discount": ["discount", "offer", "rebate"],
        "rating": ["rating", "score", "review"],
        "risk": ["risk", "churn", "fraud", "loss", "default", "return"]
    }

    detected = {}

    for kpi_name, keywords in kpi_keywords.items():
        matched_cols = []

        for col in cols:
            col_lower = col.lower()

            for word in keywords:
                if word in col_lower:
                    matched_cols.append(col)
                    break

        detected[kpi_name] = list(dict.fromkeys(matched_cols))

    return detected


def summarize_column(df, col):
    series = df[col]

    summary = {
        "column": col,
        "dtype": str(series.dtype),
        "missing": int(series.isnull().sum()),
        "unique": int(series.nunique())
    }

    if is_numeric_column(df, col):
        summary.update({
            "type": "numeric",
            "total": safe_round(series.sum()),
            "average": safe_round(series.mean()),
            "median": safe_round(series.median()),
            "minimum": safe_round(series.min()),
            "maximum": safe_round(series.max()),
            "std_dev": safe_round(series.std())
        })
    else:
        mode_values = series.dropna().mode()
        top_value = mode_values.iloc[0] if not mode_values.empty else "N/A"

        try:
            top_count = int(series.value_counts().iloc[0])
        except Exception:
            top_count = 0

        summary.update({
            "type": "categorical",
            "top_value": top_value,
            "top_count": top_count
        })

    return summary


def generate_kpi_summary(df):
    detected = detect_kpi_columns(df)
    summary = {}

    for kpi_name, columns in detected.items():
        valid_summaries = []

        for col in columns:
            try:
                valid_summaries.append(summarize_column(df, col))
            except Exception:
                continue

        if valid_summaries:
            summary[kpi_name] = valid_summaries

    return summary


def generate_kpi_insights(df):
    summary = generate_kpi_summary(df)
    insights = []

    for kpi_name, summaries in summary.items():
        readable_name = kpi_name.replace("_", " ").title()

        for item in summaries:
            col = item["column"]

            if item["type"] == "numeric":
                insights.append(
                    f"{readable_name} KPI detected in '{col}'. Total: {item['total']}, Average: {item['average']}, Range: {item['minimum']} to {item['maximum']}."
                )
            else:
                insights.append(
                    f"{readable_name} KPI detected in '{col}'. It has {item['unique']} unique values. Most frequent value: '{item['top_value']}'."
                )

    if not insights:
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

        if numeric_cols:
            best_numeric = df[numeric_cols].mean().idxmax()
            insights.append(
                f"No named business KPI detected, but '{best_numeric}' appears important based on numeric average."
            )

        if categorical_cols:
            best_cat = categorical_cols[0]
            insights.append(
                f"No named business KPI detected, but '{best_cat}' can be used for segmentation analysis."
            )

        if not numeric_cols and not categorical_cols:
            insights.append("No useful KPI signals detected in this dataset.")

    return insights


def flatten_kpi_summary(summary):
    rows = []

    for kpi_name, summaries in summary.items():
        for item in summaries:
            row = {
                "KPI": kpi_name.replace("_", " ").title(),
                "Column": item.get("column"),
                "Type": item.get("type"),
                "Missing": item.get("missing"),
                "Unique": item.get("unique")
            }

            if item.get("type") == "numeric":
                row.update({
                    "Total": item.get("total"),
                    "Average": item.get("average"),
                    "Median": item.get("median"),
                    "Min": item.get("minimum"),
                    "Max": item.get("maximum"),
                    "Std Dev": item.get("std_dev")
                })
            else:
                row.update({
                    "Top Value": item.get("top_value"),
                    "Top Count": item.get("top_count")
                })

            rows.append(row)

    return pd.DataFrame(rows)


def show_kpi_agent(df):
    st.header("📌 Business KPI Agent")
    st.caption("Generic KPI detection that works across sales, finance, customer, risk, healthcare, education, and technical datasets.")

    summary = generate_kpi_summary(df)

    if not summary:
        st.warning("No direct KPI keyword columns found. Showing fallback dataset signals.")

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", df.shape[0])
        c2.metric("Columns", df.shape[1])
        c3.metric("Numeric Columns", len(numeric_cols))
        c4.metric("Categorical Columns", len(categorical_cols))

        for insight in generate_kpi_insights(df):
            st.write(f"💡 {insight}")

        return

    flat_df = flatten_kpi_summary(summary)

    numeric_kpis = flat_df[flat_df["Type"] == "numeric"] if not flat_df.empty else pd.DataFrame()
    categorical_kpis = flat_df[flat_df["Type"] == "categorical"] if not flat_df.empty else pd.DataFrame()

    st.subheader("Executive KPI Cards")

    cards = st.columns(4)
    card_index = 0

    if not numeric_kpis.empty:
        for _, row in numeric_kpis.head(8).iterrows():
            with cards[card_index % 4]:
                st.metric(
                    label=f"{row['KPI']} - {row['Column']}",
                    value=row.get("Total", "N/A")
                )
            card_index += 1

    if card_index == 0 and not categorical_kpis.empty:
        for _, row in categorical_kpis.head(8).iterrows():
            with cards[card_index % 4]:
                st.metric(
                    label=f"{row['KPI']} - {row['Column']}",
                    value=row.get("Unique", "N/A")
                )
            card_index += 1

    st.subheader("Detected KPI Table")
    st.dataframe(flat_df, width="stretch")

    st.subheader("KPI Insights")
    for insight in generate_kpi_insights(df):
        st.write(f"💡 {insight}")

    st.subheader("KPI Recommendations")
    st.write("✅ Use numeric KPI totals for business performance tracking.")
    st.write("✅ Use categorical KPI columns for segmentation analysis.")
    st.write("✅ Check missing values before making final decisions.")
    st.write("✅ Use Dashboard Studio to visualize KPI trends and comparisons.")