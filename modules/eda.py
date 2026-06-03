# modules/eda.py
import plotly.express as px

def numeric_summary(df):
    return df.describe()

def create_histogram(df, column):
    return px.histogram(df, x=column, title=f"Distribution of {column}")

def create_bar_chart(df, column):
    return px.bar(df[column].value_counts().reset_index(),
                  x=column, y="count",
                  title=f"Count of {column}")