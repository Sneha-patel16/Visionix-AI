import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import google.generativeai as genai

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error
api_key = os.getenv("GEMINI_API_KEY")
st.write(os.getenv("GEMINI_API_KEY"))
st.set_page_config(page_title="DataMind AI", layout="wide")

st.title("🤖 DataMind AI")
st.caption("Agentic Data Analyst AI — Cleaning, EDA, ML, Chat & Report")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])


def data_quality_score(df):
    total_cells = df.shape[0] * df.shape[1]
    missing = df.isnull().sum().sum()
    duplicate = df.duplicated().sum()

    missing_score = 100 - ((missing / total_cells) * 100 if total_cells else 0)
    duplicate_score = 100 - ((duplicate / len(df)) * 100 if len(df) else 0)

    return round((missing_score * 0.6) + (duplicate_score * 0.4), 2)


def clean_data(df):
    clean_df = df.copy()

    clean_df.columns = (
        clean_df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )

    clean_df = clean_df.drop_duplicates()

    for col in clean_df.columns:
        if clean_df[col].dtype == "object":
            mode_val = clean_df[col].mode()
            clean_df[col] = clean_df[col].fillna(mode_val[0] if not mode_val.empty else "unknown")
        else:
            clean_df[col] = clean_df[col].fillna(clean_df[col].median())

    return clean_df


def suggest_targets(df):
    suggestions = []

    for col in df.columns:
        name = col.lower()
        unique_count = df[col].nunique()

        if any(word in name for word in [
            "target", "label", "class", "result", "status",
            "price", "sales", "profit", "crime", "act",
            "churn", "risk", "score"
        ]):
            suggestions.append(col)
        elif 1 < unique_count <= 20:
            suggestions.append(col)

    return list(dict.fromkeys(suggestions))[:5]


def generate_dataset_answer(df, question):
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return (
            "❌ Gemini API key not found.\n\n"
            "Please run this command in terminal:\n\n"
            "`set GEMINI_API_KEY=your_api_key_here`\n\n"
            "Then restart Streamlit."
        )

    genai.configure(api_key=api_key)

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    numeric_summary = (
        df[numeric_cols].describe().to_string()
        if numeric_cols else "No numeric columns found."
    )

    sample_data = df.head(15).to_string()

    dataset_context = f"""
Dataset Overview:
Rows: {df.shape[0]}
Columns: {df.shape[1]}

Column Names:
{list(df.columns)}

Numeric Columns:
{numeric_cols}

Categorical Columns:
{categorical_cols}

Missing Values:
{df.isnull().sum().to_dict()}

Duplicate Rows:
{int(df.duplicated().sum())}

Data Types:
{df.dtypes.astype(str).to_dict()}

Numeric Summary:
{numeric_summary}

Sample Data:
{sample_data}
"""

    prompt = f"""
You are DataMind AI, an expert Agentic Data Analyst.

Rules:
- Analyze only the provided dataset context.
- Give practical, data-driven insights.
- If the user asks business questions, give actionable recommendations.
- If data is insufficient, clearly mention what extra columns are needed.
- Do not make fake claims beyond the dataset.
- Answer in simple Hinglish/English mix.

Dataset Context:
{dataset_context}

User Question:
{question}

Answer:
"""

    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
        return f"❌ AI Error: {e}"


def generate_auto_report(df, score):
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    report = []

    report.append("## 📄 DataMind AI Auto Report")
    report.append(f"Rows: {df.shape[0]}")
    report.append(f"Columns: {df.shape[1]}")
    report.append(f"Data Quality Score: {score}/100")
    report.append(f"Duplicate Rows: {df.duplicated().sum()}")
    report.append(f"Total Missing Values: {df.isnull().sum().sum()}")

    report.append("\n## Column Types")
    report.append(f"Numeric Columns: {', '.join(numeric_cols) if numeric_cols else 'None'}")
    report.append(f"Categorical Columns: {', '.join(categorical_cols) if categorical_cols else 'None'}")

    report.append("\n## Key Insights")

    if df.duplicated().sum() > 0:
        report.append("- Dataset contains duplicate rows. Cleaning is recommended.")

    if df.isnull().sum().sum() > 0:
        report.append("- Dataset contains missing values. Missing value treatment is required.")

    if numeric_cols:
        highest = df[numeric_cols].mean().idxmax()
        lowest = df[numeric_cols].mean().idxmin()
        report.append(f"- '{highest}' has the highest average value.")
        report.append(f"- '{lowest}' has the lowest average value.")

    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr().abs()
        np.fill_diagonal(corr.values, 0)
        pair = corr.stack().idxmax()
        value = corr.stack().max()
        report.append(
            f"- Strongest numeric relationship: {pair[0]} and {pair[1]} with correlation {value:.2f}."
        )

    if score >= 85:
        report.append("- Dataset quality is good for analysis.")
    elif score >= 60:
        report.append("- Dataset needs cleaning before serious ML work.")
    else:
        report.append("- Dataset quality is poor. Cleaning is highly recommended.")

    report.append("\n## Recommended Next Steps")
    report.append("- Clean duplicate and missing values.")
    report.append("- Explore important numerical relationships.")
    report.append("- Select a proper target column for ML.")
    report.append("- Use AI Chat to generate business/domain-specific insights.")

    return "\n".join(report)
def detect_columns(df):
    cols = df.columns.tolist()
    lower_cols = {col.lower(): col for col in cols}

    date_cols = []
    for col in cols:
        try:
            converted = pd.to_datetime(df[col], errors="coerce")
            if converted.notna().sum() > len(df) * 0.5:
                date_cols.append(col)
        except:
            pass

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    return date_cols, numeric_cols, categorical_cols


def show_auto_dashboard(df):
    st.subheader("🚀 Interactive Smart Dashboard")

    date_cols, numeric_cols, categorical_cols = detect_columns(df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Rows", df.shape[0])
    c2.metric("Total Columns", df.shape[1])
    c3.metric("Numeric Columns", len(numeric_cols))
    c4.metric("Categorical Columns", len(categorical_cols))

    st.divider()

    st.subheader("🎛️ Dashboard Controls")

    col1, col2, col3 = st.columns(3)

    with col1:
        chart_type = st.selectbox(
            "Select Chart Type",
            ["Histogram", "Scatter Plot", "Bar Chart", "Line Chart", "Box Plot", "Pie Chart"],
            key="dash_chart_type"
        )

    with col2:
        x_col = st.selectbox(
            "Select X-axis",
            df.columns,
            key="dash_x_col"
        )

    with col3:
        y_col = st.selectbox(
            "Select Y-axis / Value",
            numeric_cols if numeric_cols else df.columns,
            key="dash_y_col"
        )

    filtered_df = df.copy()

    if categorical_cols:
        filter_col = st.selectbox(
            "Optional Category Filter",
            ["None"] + categorical_cols,
            key="dash_filter_col"
        )

        if filter_col != "None":
            selected_values = st.multiselect(
                f"Select values from {filter_col}",
                filtered_df[filter_col].dropna().unique().tolist(),
                key="dash_filter_values"
            )

            if selected_values:
                filtered_df = filtered_df[filtered_df[filter_col].isin(selected_values)]

    agg_func = st.selectbox(
        "Aggregation",
        ["None", "sum", "mean", "count", "max", "min"],
        key="dash_agg"
    )

    st.write(f"Showing {filtered_df.shape[0]} rows after filtering.")

    try:
        if chart_type == "Histogram":
            fig = px.histogram(
                filtered_df,
                x=x_col,
                title=f"Histogram of {x_col}"
            )

        elif chart_type == "Scatter Plot":
            fig = px.scatter(
                filtered_df,
                x=x_col,
                y=y_col,
                title=f"{x_col} vs {y_col}"
            )

        elif chart_type == "Bar Chart":
            if agg_func != "None":
                temp = filtered_df.groupby(x_col)[y_col].agg(agg_func).reset_index()
                fig = px.bar(temp, x=x_col, y=y_col, title=f"{agg_func} of {y_col} by {x_col}")
            else:
                temp = filtered_df[x_col].value_counts().head(20).reset_index()
                temp.columns = [x_col, "count"]
                fig = px.bar(temp, x=x_col, y="count", title=f"Count by {x_col}")

        elif chart_type == "Line Chart":
            if agg_func != "None":
                temp = filtered_df.groupby(x_col)[y_col].agg(agg_func).reset_index()
                fig = px.line(temp, x=x_col, y=y_col, title=f"{agg_func} of {y_col} by {x_col}")
            else:
                fig = px.line(filtered_df, x=x_col, y=y_col, title=f"{y_col} over {x_col}")

        elif chart_type == "Box Plot":
            fig = px.box(
                filtered_df,
                x=x_col,
                y=y_col,
                title=f"Box Plot of {y_col} by {x_col}"
            )

        elif chart_type == "Pie Chart":
            temp = filtered_df[x_col].value_counts().head(10).reset_index()
            temp.columns = [x_col, "count"]
            fig = px.pie(temp, names=x_col, values="count", title=f"Pie Chart of {x_col}")

        st.plotly_chart(fig, use_container_width=True, key="interactive_dashboard_chart")

    except Exception as e:
        st.error(f"Chart could not be generated: {e}")

    st.subheader("📌 Filtered Data Preview")
    st.dataframe(filtered_df.head(20), use_container_width=True)
    
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    score = data_quality_score(df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Duplicate Rows", int(df.duplicated().sum()))
    c4.metric("Data Quality Score", f"{score}/100")

    if score >= 85:
        st.success("✅ Dataset quality is good.")
    elif score >= 60:
        st.warning("⚠️ Dataset needs cleaning.")
    else:
        st.error("❌ Dataset quality is poor. Cleaning is strongly recommended.")

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📊 Preview",
        "🧹 Cleaning",
        "📈 EDA",
        "🗺️ Map",
        "🤖 ML Model",
        "💬 AI Chat",
        "📄 Report",
        "🚀 Auto Dashboard"

    ])

    with tab1:
        st.subheader("Dataset Preview")
        st.dataframe(df.head(20), use_container_width=True)

        info_df = pd.DataFrame({
            "Column": df.columns,
            "Data Type": df.dtypes.astype(str).values,
            "Missing Values": df.isnull().sum().values,
            "Unique Values": df.nunique().values
        })

        st.subheader("Column Information")
        st.dataframe(info_df, use_container_width=True)

    with tab2:
        st.subheader("Auto Data Cleaning")

        if st.button("🧹 Clean Dataset"):
            cleaned_df = clean_data(df)

            st.success("✅ Dataset cleaned successfully!")

            c1, c2, c3 = st.columns(3)
            c1.metric("Old Rows", df.shape[0])
            c2.metric("New Rows", cleaned_df.shape[0])
            c3.metric("Removed Duplicates", df.shape[0] - cleaned_df.shape[0])

            st.dataframe(cleaned_df.head(20), use_container_width=True)

            csv = cleaned_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                "⬇ Download Cleaned Dataset",
                csv,
                "cleaned_dataset.csv",
                "text/csv"
            )

    with tab3:
        st.subheader("Exploratory Data Analysis")

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

        if numeric_cols:
            selected_num = st.selectbox("Select numeric column", numeric_cols)

            fig = px.histogram(df, x=selected_num, title=f"Distribution of {selected_num}")
            st.plotly_chart(fig, use_container_width=True)

            if len(numeric_cols) >= 2:
                st.subheader("Correlation Heatmap")
                corr = df[numeric_cols].corr()
                fig_corr = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Heatmap")
                st.plotly_chart(fig_corr, use_container_width=True)

        if categorical_cols:
            selected_cat = st.selectbox("Select categorical column", categorical_cols)
            value_df = df[selected_cat].value_counts().reset_index()
            value_df.columns = [selected_cat, "count"]

            fig_bar = px.bar(value_df, x=selected_cat, y="count", title=f"Count of {selected_cat}")
            st.plotly_chart(fig_bar, use_container_width=True)

    with tab4:
        st.subheader("Map Visualization")

        lower_cols = [col.lower() for col in df.columns]

        if "latitude" in lower_cols and "longitude" in lower_cols:
            lat_col = df.columns[lower_cols.index("latitude")]
            lon_col = df.columns[lower_cols.index("longitude")]

            map_df = df[[lat_col, lon_col]].dropna()

            if not map_df.empty:
                st.map(map_df.rename(columns={lat_col: "latitude", lon_col: "longitude"}))
            else:
                st.warning("Latitude and longitude columns exist, but values are empty.")
        else:
            st.info("No latitude/longitude columns found. Map is skipped for this dataset.")

    with tab5:
        st.subheader("Smart ML Model Training")

        suggestions = suggest_targets(df)

        if suggestions:
            st.info(f"Suggested target columns: {', '.join(suggestions)}")

        target_col = st.selectbox("Select target column", df.columns)

        if st.button("🚀 Train Model"):
            ml_df = df.copy().dropna()

            if ml_df[target_col].nunique() <= 1:
                st.error("Target column has only one unique value. Choose another target.")
            else:
                X = ml_df.drop(columns=[target_col])
                y = ml_df[target_col]

                X = pd.get_dummies(X, drop_first=True)

                try:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42
                    )

                    if y.dtype == "object" or y.nunique() <= 20:
                        model = RandomForestClassifier(random_state=42)
                        model.fit(X_train, y_train)
                        preds = model.predict(X_test)

                        acc = accuracy_score(y_test, preds)

                        st.success("✅ Classification model trained successfully.")
                        st.metric("Accuracy", f"{acc:.2f}")

                    else:
                        model = RandomForestRegressor(random_state=42)
                        model.fit(X_train, y_train)
                        preds = model.predict(X_test)

                        r2 = r2_score(y_test, preds)
                        mae = mean_absolute_error(y_test, preds)

                        st.success("✅ Regression model trained successfully.")
                        st.metric("R² Score", f"{r2:.2f}")
                        st.metric("Mean Absolute Error", f"{mae:.2f}")

                    importance_df = pd.DataFrame({
                        "Feature": X.columns,
                        "Importance": model.feature_importances_
                    }).sort_values(by="Importance", ascending=False).head(10)

                    st.subheader("Top Important Features")
                    st.dataframe(importance_df, use_container_width=True)

                    fig_imp = px.bar(
                        importance_df,
                        x="Importance",
                        y="Feature",
                        orientation="h",
                        title="Feature Importance"
                    )
                    st.plotly_chart(fig_imp, use_container_width=True)

                except Exception as e:
                    st.error(f"Error: {e}")

    with tab6:
        st.subheader("💬 Chat with Dataset")

        user_question = st.text_area(
            "Ask anything about your dataset",
            placeholder="Example: How can sales grow? What are the main problems? Give 5 insights."
        )

        if st.button("Ask DataMind AI"):
            if user_question.strip():
                with st.spinner("DataMind AI is analyzing your dataset..."):
                    answer = generate_dataset_answer(df, user_question)
                    st.markdown(answer)
            else:
                st.warning("Please enter a question.")

        st.info(
            "Try: How can sales grow? What are the main problems? "
            "Give 5 business insights. Which column affects performance? "
            "What should I improve in this dataset?"
        )

    with tab7:
        st.subheader("Auto AI Report")

        report = generate_auto_report(df, score)

        st.markdown(report)

        st.download_button(
            "⬇ Download Report",
            report,
            "DataMind_AI_Report.md",
            "text/markdown"
        )
    with tab8:
        show_auto_dashboard(df)  

else:
    st.info("Please upload a CSV file to start analysis.")