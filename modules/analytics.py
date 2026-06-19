import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

def show_analytics_studio(df):
    st.header("📊 Analytics Studio")
    st.caption("All analytics are performed on cleaned dataset.")

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    if numeric_cols:
        st.subheader("Advanced Descriptive Statistics")

        stats = df[numeric_cols].agg(["mean", "median", "std", "var", "min", "max", "skew", "kurtosis"]).T
        stats["mode"] = df[numeric_cols].mode().iloc[0]
        st.dataframe(stats, width="stretch")

        selected = st.selectbox("Select Numeric Column", numeric_cols)

        fig = px.histogram(df, x=selected, title=f"Distribution Analysis: {selected}")
        st.plotly_chart(fig, width="stretch", key="analytics_distribution")

        fig_box = px.box(df, y=selected, title=f"Outlier Analysis: {selected}")
        st.plotly_chart(fig_box, width="stretch", key="analytics_box")

        if len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr(method="pearson")
            fig_corr = px.imshow(corr, text_auto=True, aspect="auto", title="Pearson Correlation Heatmap")
            st.plotly_chart(fig_corr, width="stretch", key="analytics_corr")

    if categorical_cols:
        st.subheader("Categorical Frequency Analysis")
        selected_cat = st.selectbox("Select Categorical Column", categorical_cols)
        freq = df[selected_cat].value_counts().head(15).reset_index()
        freq.columns = [selected_cat, "count"]

        fig = px.bar(freq, x=selected_cat, y="count", title=f"Top Categories in {selected_cat}")
        st.plotly_chart(fig, width="stretch", key="analytics_cat")