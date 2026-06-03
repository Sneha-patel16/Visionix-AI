# modules/insight_agent.py
def generate_basic_insights(df):
    insights = []

    missing = df.isnull().sum().sum()
    if missing > 0:
        insights.append(f"Dataset has {missing} missing values.")

    duplicates = df.duplicated().sum()
    if duplicates > 0:
        insights.append(f"Dataset has {duplicates} duplicate rows.")

    numeric_cols = df.select_dtypes(include="number").columns
    if len(numeric_cols) > 0:
        highest_mean_col = df[numeric_cols].mean().idxmax()
        insights.append(f"{highest_mean_col} has the highest average value.")

    if not insights:
        insights.append("Dataset looks clean and ready for analysis.")

    return insights