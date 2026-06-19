import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.inspection import permutation_importance


def get_importance(model, X_test, y_test, feature_names):
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_

    elif hasattr(model, "coef_"):
        coef = model.coef_
        if len(coef.shape) > 1:
            values = np.mean(np.abs(coef), axis=0)
        else:
            values = np.abs(coef)

    else:
        result = permutation_importance(
            model,
            X_test,
            y_test,
            n_repeats=5,
            random_state=42,
            n_jobs=-1
        )
        values = result.importances_mean

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": values
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    ).head(20)

    return importance_df


def show_feature_importance_agent():
    st.header("🧬 Feature Importance Agent")
    st.caption("Universal feature importance using tree importance, coefficients, or permutation importance.")

    required = ["best_model", "X_test", "y_test", "feature_names"]

    if not all(key in st.session_state for key in required):
        st.info("Run ML Studio first.")
        return

    model = st.session_state["best_model"]
    X_test = st.session_state["X_test"]
    y_test = st.session_state["y_test"]
    feature_names = st.session_state["feature_names"]

    importance_df = get_importance(model, X_test, y_test, feature_names)

    st.subheader("Top Important Features")
    st.dataframe(importance_df, width="stretch")

    fig = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Top Feature Importance"
    )

    st.plotly_chart(fig, width="stretch", key="feature_importance_agent_chart")

    top_features = importance_df.head(5)["Feature"].tolist()

    st.subheader("Agent Interpretation")
    st.write(f"💡 The model is mostly influenced by: {', '.join(top_features)}.")
    st.write("💡 Higher importance means the feature strongly affects prediction behavior.")