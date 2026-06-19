import streamlit as st
import plotly.express as px


def show_model_comparison_dashboard():
    st.header("📊 Model Comparison Dashboard")
    st.caption("Compares all trained ML models and highlights the strongest performer.")

    if "model_results" not in st.session_state or "problem_type" not in st.session_state:
        st.info("Run ML Studio first.")
        return

    result_df = st.session_state["model_results"]
    problem_type = st.session_state["problem_type"]

    st.dataframe(result_df, width="stretch")

    if problem_type == "classification":
        metric = st.selectbox(
            "Select Metric",
            ["Accuracy", "Precision", "Recall", "F1 Score"]
        )
    else:
        metric = st.selectbox(
            "Select Metric",
            ["R2 Score", "MAE", "RMSE"]
        )

    fig = px.bar(
        result_df,
        x="Model",
        y=metric,
        title=f"Model Comparison by {metric}"
    )

    st.plotly_chart(fig, width="stretch", key="model_comparison_dashboard")

    if metric in ["MAE", "RMSE"]:
        best_row = result_df.loc[result_df[metric].idxmin()]
        st.success(f"🏆 Best model by lowest {metric}: {best_row['Model']}")
    else:
        best_row = result_df.loc[result_df[metric].idxmax()]
        st.success(f"🏆 Best model by highest {metric}: {best_row['Model']}")