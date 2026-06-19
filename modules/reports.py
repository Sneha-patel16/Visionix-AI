import streamlit as st
import numpy as np
from io import BytesIO

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

from modules.insights import generate_agentic_insights
from modules.kpi_agent import generate_kpi_summary, generate_kpi_insights


def generate_report_text(df, cleaning_log):
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    insights = generate_agentic_insights(df)
    kpi_summary = generate_kpi_summary(df)
    kpi_insights = generate_kpi_insights(df)

    report = "# VisionIX Enterprise Executive Report\n\n"

    report += "## 1. Executive Summary\n"
    report += "VisionIX analyzed the uploaded dataset using agentic cleaning, feature engineering, analytics, ML-readiness checks, KPI detection, dashboard intelligence, and insight generation.\n\n"

    report += "## 2. Dataset Overview\n"
    report += f"- Rows after processing: {df.shape[0]}\n"
    report += f"- Columns after processing: {df.shape[1]}\n"
    report += f"- Numeric Columns: {len(numeric_cols)}\n"
    report += f"- Categorical Columns: {len(categorical_cols)}\n"
    report += f"- Missing Values: {df.isnull().sum().sum()}\n"
    report += f"- Duplicate Rows: {df.duplicated().sum()}\n\n"

    report += "## 3. Cleaning & Feature Engineering Summary\n"
    for item in cleaning_log:
        report += f"- {item}\n"

    report += "\n## 4. Business KPI Analysis\n"
    if kpi_summary:
        for kpi, values in kpi_summary.items():
            report += f"- {kpi.upper()}: {values}\n"
    else:
        report += "- No clear business KPI columns detected.\n"

    report += "\n## 5. KPI Insights\n"
    for insight in kpi_insights:
        report += f"- {insight}\n"

    report += "\n## 6. Analytics & Pattern Insights\n"
    for insight in insights:
        report += f"- {insight}\n"

    report += "\n## 7. Risk Analysis\n"
    report += "- Review outliers and high-correlation features before making business decisions.\n"
    report += "- Validate target column quality before deploying machine learning models.\n"
    report += "- Check business KPI columns for abnormal spikes, drops, or missing values.\n\n"

    report += "## 8. Growth Opportunities\n"
    report += "- Use KPI trends to identify high-performing segments.\n"
    report += "- Use feature importance from XAI Agent to focus on high-impact variables.\n"
    report += "- Use Dashboard Studio visuals for stakeholder decision-making.\n\n"

    report += "## 9. Final Recommendations\n"
    report += "- Use cleaned and engineered data for all analysis.\n"
    report += "- Compare multiple models before deployment.\n"
    report += "- Use XAI Agent to explain model behavior.\n"
    report += "- Monitor key business KPIs regularly.\n"

    return report


def create_pdf_report(df, cleaning_log):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    insights = generate_agentic_insights(df)
    kpi_summary = generate_kpi_summary(df)
    kpi_insights = generate_kpi_insights(df)

    story.append(Paragraph("VisionIX Enterprise Executive Report", styles["Title"]))
    story.append(Spacer(1, 18))
    story.append(Paragraph("Agentic AI Decision Intelligence Platform", styles["Heading2"]))
    story.append(Spacer(1, 20))

    summary = [
        ["Metric", "Value"],
        ["Rows after processing", str(df.shape[0])],
        ["Columns after processing", str(df.shape[1])],
        ["Numeric Columns", str(len(numeric_cols))],
        ["Categorical Columns", str(len(categorical_cols))],
        ["Missing Values", str(df.isnull().sum().sum())],
        ["Duplicate Rows", str(df.duplicated().sum())],
    ]

    table = Table(summary)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.black),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 1, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 8),
    ]))

    story.append(Paragraph("Dataset Overview", styles["Heading2"]))
    story.append(table)
    story.append(Spacer(1, 18))

    story.append(Paragraph("Cleaning & Feature Engineering Summary", styles["Heading2"]))
    for item in cleaning_log[:15]:
        story.append(Paragraph(f"- {item}", styles["Normal"]))
        story.append(Spacer(1, 5))

    story.append(PageBreak())

    story.append(Paragraph("Business KPI Analysis", styles["Heading2"]))

    if kpi_summary:
        kpi_table_data = [["KPI", "Detected Details"]]
        for kpi, values in kpi_summary.items():
            kpi_table_data.append([kpi.upper(), str(values)])

        kpi_table = Table(kpi_table_data)
        kpi_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.black),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 1, colors.grey),
            ("PADDING", (0, 0), (-1, -1), 6),
        ]))

        story.append(kpi_table)
    else:
        story.append(Paragraph("No strong KPI columns detected.", styles["Normal"]))

    story.append(Spacer(1, 18))

    story.append(Paragraph("Premium KPI Insights", styles["Heading2"]))
    for insight in kpi_insights:
        story.append(Paragraph(f"- {insight}", styles["Normal"]))
        story.append(Spacer(1, 5))

    story.append(Spacer(1, 12))

    story.append(Paragraph("Analytics & Pattern Insights", styles["Heading2"]))
    for insight in insights:
        story.append(Paragraph(f"- {insight}", styles["Normal"]))
        story.append(Spacer(1, 5))

    story.append(Spacer(1, 12))

    story.append(Paragraph("Risk Analysis", styles["Heading2"]))
    risks = [
        "Review outliers before final decisions.",
        "Validate high-correlation features to avoid redundancy.",
        "Check model explainability before deployment.",
        "Monitor detected KPI columns regularly."
    ]

    for risk in risks:
        story.append(Paragraph(f"- {risk}", styles["Normal"]))
        story.append(Spacer(1, 5))

    story.append(Spacer(1, 12))

    story.append(Paragraph("Final Recommendations", styles["Heading2"]))
    recs = [
        "Use cleaned and engineered data for all analytics and ML.",
        "Use Dashboard Studio for stakeholder-ready visuals.",
        "Use XAI Agent to explain ML behavior.",
        "Use KPI Agent to track business performance.",
        "Compare multiple ML models before deployment."
    ]

    for rec in recs:
        story.append(Paragraph(f"- {rec}", styles["Normal"]))
        story.append(Spacer(1, 5))

    doc.build(story)
    buffer.seek(0)
    return buffer


def show_report_studio(df, cleaning_log):
    st.header("📄 Enterprise Reports")

    report = generate_report_text(df, cleaning_log)

    st.markdown(report)

    st.download_button(
        "⬇ Download Enterprise Markdown Report",
        report,
        "VisionIX_Enterprise_Report.md",
        "text/markdown"
    )

    pdf = create_pdf_report(df, cleaning_log)

    st.download_button(
        "⬇ Download Enterprise PDF Report",
        pdf,
        "VisionIX_Enterprise_Report.pdf",
        "application/pdf"
    )