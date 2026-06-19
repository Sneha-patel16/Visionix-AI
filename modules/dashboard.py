import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from modules.agent_core import detect_column_types
from modules.kpi_agent import generate_kpi_summary, generate_kpi_insights


def show_dashboard_studio(df):
    st.header("🚀 Executive Dashboard Studio v2")
    st.caption("Power BI-style executive dashboard generated from cleaned and engineered dataset.")

    numeric_cols, categorical_cols, date_cols, id_cols = detect_column_types(df)
    kpi_summary = generate_kpi_summary(df)

    st.subheader("📌 Executive KPI Cards")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Numeric Features", len(numeric_cols))
    c4.metric("Categorical Features", len(categorical_cols))

    if kpi_summary:
        st.subheader("💼 Business KPIs")

        kpi_cols = st.columns(4)
        i = 0

        for kpi, values in kpi_summary.items():
            with kpi_cols[i % 4]:
                if "total" in values:
                    st.metric(kpi.upper(), values["total"])
                elif "unique" in values:
                    st.metric(kpi.upper(), values["unique"])
                else:
                    st.metric(kpi.upper(), "Detected")
            i += 1

    st.divider()

    st.subheader("📊 Agent-Generated Visuals")

    if numeric_cols:
        main_metric = numeric_cols[0]

        fig_hist = px.histogram(
            df,
            x=main_metric,
            title=f"Distribution of {main_metric}"
        )
        st.plotly_chart(fig_hist, width="stretch", key="exec_hist")

    if categorical_cols:
        cat = categorical_cols[0]
        temp = df[cat].value_counts().head(10).reset_index()
        temp.columns = [cat, "count"]

        fig_bar = px.bar(
            temp,
            x=cat,
            y="count",
            title=f"Top Categories by {cat}"
        )
        st.plotly_chart(fig_bar, width="stretch", key="exec_bar")

    if len(numeric_cols) >= 2:
        fig_scatter = px.scatter(
            df,
            x=numeric_cols[0],
            y=numeric_cols[1],
            title=f"Relationship: {numeric_cols[0]} vs {numeric_cols[1]}"
        )
        st.plotly_chart(fig_scatter, width="stretch", key="exec_scatter")

        corr = df[numeric_cols].corr()
        fig_corr = px.imshow(
            corr,
            text_auto=True,
            aspect="auto",
            title="Correlation Heatmap"
        )
        st.plotly_chart(fig_corr, width="stretch", key="exec_corr")

    if date_cols and numeric_cols:
        date_col = date_cols[0]
        metric = numeric_cols[0]

        temp = df.copy()
        temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
        trend = temp.groupby(date_col)[metric].mean().reset_index()

        fig_line = px.line(
            trend,
            x=date_col,
            y=metric,
            title=f"{metric} Trend Over Time"
        )
        st.plotly_chart(fig_line, width="stretch", key="exec_trend")

    st.divider()

    st.subheader("💡 Executive Insights")

    for insight in generate_kpi_insights(df):
        st.write(f"💡 {insight}")

    st.divider()

    st.subheader("🎛️ Manual Interactive Chart Builder")

    chart_type = st.selectbox(
        "Chart Type",
        ["Histogram", "Bar", "Scatter", "Line", "Box", "Pie"],
        key="manual_chart_type_v2"
    )

    x_col = st.selectbox("X Axis", df.columns, key="manual_x_v2")
    y_col = st.selectbox(
        "Y Axis",
        numeric_cols if numeric_cols else df.columns,
        key="manual_y_v2"
    )

    try:
        if chart_type == "Histogram":
            fig = px.histogram(df, x=x_col)

        elif chart_type == "Bar":
            temp = df.groupby(x_col)[y_col].mean().reset_index()
            fig = px.bar(temp, x=x_col, y=y_col)

        elif chart_type == "Scatter":
            fig = px.scatter(df, x=x_col, y=y_col)

        elif chart_type == "Line":
            fig = px.line(df, x=x_col, y=y_col)

        elif chart_type == "Box":
            fig = px.box(df, x=x_col, y=y_col)

        else:
            temp = df[x_col].value_counts().head(10).reset_index()
            temp.columns = [x_col, "count"]
            fig = px.pie(temp, names=x_col, values="count")

        st.plotly_chart(fig, width="stretch", key="manual_dashboard_v2")

    except Exception as e:
        st.error(f"Chart generation failed: {e}")