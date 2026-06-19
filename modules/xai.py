import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from modules.feature_importance_agent import get_importance


def model_readiness_score(results, problem_type):
    if results.empty:
        return 0

    if problem_type == "classification":
        score = results["F1 Score"].max()
    else:
        score = max(0, results["R2 Score"].max())

    return round(min(score * 10, 10), 2)


def show_xai_agent(best_model, result_df, problem_type, feature_names):
    st.header("🧠 Explainable AI Agent")
    st.caption("Explains model selection, feature drivers, and deployment readiness.")

    readiness = model_readiness_score(result_df, problem_type)

    c1, c2 = st.columns(2)
    c1.metric("Model Readiness Score", f"{readiness}/10")

    if readiness >= 8:
        c2.success("High readiness")
    elif readiness >= 5:
        c2.warning("Moderate readiness")
    else:
        c2.error("Low readiness")

    st.subheader("Model Selection Explanation")

    if problem_type == "classification":
        best_row = result_df.loc[result_df["F1 Score"].idxmax()]
        metric = "F1 Score"
    else:
        best_row = result_df.loc[result_df["R2 Score"].idxmax()]
        metric = "R2 Score"

    st.write(f"🏆 Best Model: **{best_row['Model']}**")
    st.write(f"✅ Selected because it achieved the highest **{metric}** among tested models.")

    if "X_test" not in st.session_state or "y_test" not in st.session_state:
        st.info("Run ML Studio again to generate feature explanations.")
        return

    importance_df = get_importance(
        best_model,
        st.session_state["X_test"],
        st.session_state["y_test"],
        feature_names
    )

    st.subheader("Feature Drivers")
    st.dataframe(importance_df, width="stretch")

    fig = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title="XAI Feature Drivers"
    )
    st.plotly_chart(fig, width="stretch", key="xai_feature_drivers")

    st.subheader("Business/Domain Interpretation")
    top = importance_df.head(5)["Feature"].tolist()

    st.write(f"💡 Predictions are mainly driven by: {', '.join(top)}.")
    st.write("💡 These features should be reviewed carefully before making final decisions.")
    st.write("💡 If any top feature is ID-like or leakage-related, remove it before deployment.")