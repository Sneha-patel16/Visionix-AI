import streamlit as st
import pandas as pd

def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)

def show_data_hub(raw_df, cleaned_df):

    st.header("📂 Data Hub")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Raw Rows", raw_df.shape[0])
    c2.metric("Raw Columns", raw_df.shape[1])
    c3.metric("Clean Rows", cleaned_df.shape[0])
    c4.metric("Clean Columns", cleaned_df.shape[1])

    st.subheader("Raw Dataset Preview")
    st.dataframe(raw_df.head(20), use_container_width=True)

    st.subheader("Cleaned Dataset Preview")
    st.dataframe(cleaned_df.head(20), use_container_width=True)

    profile = pd.DataFrame({
        "Column": raw_df.columns,
        "Data Type": raw_df.dtypes.astype(str),
        "Missing Values": raw_df.isnull().sum().values,
        "Unique Values": raw_df.nunique().values
    })

    st.subheader("Column Profile")
    st.dataframe(profile, use_container_width=True)