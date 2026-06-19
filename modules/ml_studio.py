import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR

from modules.agent_core import suggest_target_column, detect_problem_type


def make_unique_columns(df):
    df = df.copy()
    counts = {}
    new_cols = []

    for col in df.columns:
        if col not in counts:
            counts[col] = 0
            new_cols.append(col)
        else:
            counts[col] += 1
            new_cols.append(f"{col}_{counts[col]}")

    df.columns = new_cols
    return df


def prepare_ml_data(df, target_col):
    df = make_unique_columns(df)

    y = df[target_col]
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]

    X = df.drop(columns=[target_col])
    X = X.loc[:, ~X.columns.duplicated()]
    X = pd.get_dummies(X, drop_first=True)

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

    return X, y


def show_ml_studio(df):
    st.header("🧠 Agentic AutoML Studio")
    st.caption("Universal AutoML for classification and regression.")

    df = make_unique_columns(df)

    auto_target = suggest_target_column(df)

    target_col = st.selectbox(
        "Select Target Column",
        df.columns,
        index=list(df.columns).index(auto_target) if auto_target in df.columns else 0
    )

    problem_type = detect_problem_type(df, target_col)

    st.info(f"🤖 Suggested Target: {auto_target}")
    st.info(f"🤖 Detected Problem Type: {problem_type.upper()}")

    if st.button("🚀 Run Agentic AutoML"):
        X, y = prepare_ml_data(df, target_col)

        if y.nunique() <= 1:
            st.error("Target column has only one unique value.")
            return

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        results = []
        trained_models = {}

        if problem_type == "classification":
            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Decision Tree": DecisionTreeClassifier(random_state=42),
                "Random Forest": RandomForestClassifier(random_state=42),
                "KNN": KNeighborsClassifier(),
                "SVM": SVC(),
                "Gradient Boosting": GradientBoostingClassifier(random_state=42),
            }

            for name, model in models.items():
                try:
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)

                    results.append({
                        "Model": name,
                        "Accuracy": accuracy_score(y_test, preds),
                        "Precision": precision_score(y_test, preds, average="weighted", zero_division=0),
                        "Recall": recall_score(y_test, preds, average="weighted", zero_division=0),
                        "F1 Score": f1_score(y_test, preds, average="weighted", zero_division=0),
                    })

                    trained_models[name] = model

                except Exception as e:
                    st.warning(f"{name} skipped: {e}")

            result_df = pd.DataFrame(results)

            if result_df.empty:
                st.error("No model trained successfully.")
                return

            best_idx = result_df["F1 Score"].idxmax()
            best_metric = "F1 Score"

        else:
            models = {
                "Linear Regression": LinearRegression(),
                "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
                "Random Forest Regressor": RandomForestRegressor(random_state=42),
                "KNN Regressor": KNeighborsRegressor(),
                "SVR": SVR(),
                "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=42),
            }

            for name, model in models.items():
                try:
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)

                    results.append({
                        "Model": name,
                        "R2 Score": r2_score(y_test, preds),
                        "MAE": mean_absolute_error(y_test, preds),
                        "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
                    })

                    trained_models[name] = model

                except Exception as e:
                    st.warning(f"{name} skipped: {e}")

            result_df = pd.DataFrame(results)

            if result_df.empty:
                st.error("No model trained successfully.")
                return

            best_idx = result_df["R2 Score"].idxmax()
            best_metric = "R2 Score"

        best_model_name = result_df.loc[best_idx, "Model"]
        best_model = trained_models[best_model_name]

        st.success(f"🏆 Best Model: {best_model_name}")
        st.dataframe(result_df, width="stretch")

        fig = px.bar(
            result_df,
            x="Model",
            y=best_metric,
            title="Model Comparison Dashboard"
        )
        st.plotly_chart(fig, width="stretch", key="model_comparison_chart")

        st.session_state["best_model"] = best_model
        st.session_state["best_model_name"] = best_model_name
        st.session_state["model_results"] = result_df
        st.session_state["problem_type"] = problem_type
        st.session_state["feature_names"] = X.columns.tolist()
        st.session_state["X_test"] = X_test
        st.session_state["y_test"] = y_test
        st.session_state["target_col"] = target_col

        st.success("✅ Results stored for XAI, Feature Agent, and Insight Agent.")