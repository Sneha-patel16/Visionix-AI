import streamlit as st

from modules.data_hub import load_data, show_data_hub
from modules.data_validation import show_validation_agent
from modules.cleaning import clean_data_agent, show_cleaning_studio
from modules.feature_engineering import feature_engineering_agent
from modules.analytics import show_analytics_studio
from modules.kpi_agent import show_kpi_agent
from modules.ml_studio import show_ml_studio
from modules.xai import show_xai_agent
from modules.model_comparison import show_model_comparison_dashboard
from modules.feature_importance_agent import show_feature_importance_agent
from modules.dashboard import show_dashboard_studio
from modules.insights import show_insight_engine
from modules.business_insight_agent import show_business_insight_agent
from modules.reports import show_report_studio


st.set_page_config(
    page_title="VisionIX",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed"
)


st.markdown("""
<style>
[data-testid="stSidebar"], [data-testid="collapsedControl"] {
    display: none;
}

.block-container {
    max-width: 1280px;
    padding-top: 1rem;
    padding-bottom: 3rem;
}

.stApp {
    background: linear-gradient(-45deg, #f8fafc, #eef2ff, #f1f5f9, #e0f2fe);
    background-size: 400% 400%;
    animation: gradientShift 18s ease infinite;
    color: #0f172a;
}

@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.ambient-orb-one {
    position: fixed;
    width: 420px;
    height: 420px;
    border-radius: 50%;
    background: rgba(37,99,235,0.12);
    filter: blur(45px);
    top: -140px;
    left: -120px;
    z-index: -1;
    animation: floatOne 12s ease-in-out infinite;
}

.ambient-orb-two {
    position: fixed;
    width: 380px;
    height: 380px;
    border-radius: 50%;
    background: rgba(14,165,233,0.12);
    filter: blur(45px);
    bottom: -120px;
    right: -100px;
    z-index: -1;
    animation: floatTwo 14s ease-in-out infinite;
}

@keyframes floatOne {
    0%, 100% { transform: translate(0, 0); }
    50% { transform: translate(60px, 40px); }
}

@keyframes floatTwo {
    0%, 100% { transform: translate(0, 0); }
    50% { transform: translate(-50px, -35px); }
}

h1, h2, h3 {
    color: #0f172a;
    letter-spacing: -0.035em;
}

p, span, label {
    color: #334155;
}

.app-shell {
    background: rgba(255,255,255,0.78);
    border: 1px solid rgba(203,213,225,0.85);
    border-radius: 28px;
    padding: 20px 24px;
    box-shadow: 0 24px 80px rgba(15,23,42,0.08);
    backdrop-filter: blur(18px);
    margin-bottom: 22px;
}

.top-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.brand {
    display: flex;
    align-items: center;
    gap: 14px;
}

.brand-mark {
    width: 46px;
    height: 46px;
    border-radius: 16px;
    background: linear-gradient(135deg, #1e40af, #0f172a);
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-size: 22px;
    font-weight: 900;
}

.brand-name {
    font-size: 24px;
    font-weight: 900;
    color: #0f172a;
    line-height: 1;
}

.brand-caption {
    font-size: 13px;
    color: #64748b;
    margin-top: 4px;
}

.top-badge {
    padding: 9px 15px;
    border-radius: 999px;
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    color: #1d4ed8;
    font-weight: 800;
    font-size: 13px;
}

.hero {
    position: relative;
    overflow: hidden;
    border-radius: 34px;
    padding: 56px 54px;
    background:
        linear-gradient(135deg, rgba(15,23,42,0.98), rgba(30,41,59,0.97)),
        #020617;
    box-shadow: 0 30px 90px rgba(15,23,42,0.28);
    margin-bottom: 24px;
    border: 1px solid rgba(255,255,255,0.10);
}

.hero:before {
    content: "";
    position: absolute;
    width: 520px;
    height: 520px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(96,165,250,0.30), transparent 64%);
    right: -180px;
    top: -180px;
}

.hero:after {
    content: "";
    position: absolute;
    width: 420px;
    height: 420px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(14,165,233,0.16), transparent 65%);
    left: -160px;
    bottom: -180px;
}

.hero-content {
    position: relative;
    z-index: 1;
}

.hero-kicker {
    color: #93c5fd;
    font-size: 14px;
    font-weight: 850;
    text-transform: uppercase;
    letter-spacing: 0.10em;
    margin-bottom: 18px;
}

.vision-title {
    font-size: 92px;
    line-height: 0.92;
    font-weight: 1000;
    color: #ffffff;
    letter-spacing: -0.06em;
}

.vision-title span {
    color: #60a5fa;
}

.tagline {
    margin-top: 24px;
    font-size: 24px;
    line-height: 1.55;
    color: #dbeafe;
    max-width: 900px;
    font-weight: 450;
}

.hero-metrics {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
    margin-top: 34px;
}

.hero-metric {
    background: rgba(255,255,255,0.075);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 20px;
    padding: 18px;
}

.hero-metric-title {
    color: #ffffff;
    font-size: 21px;
    font-weight: 850;
}

.hero-metric-label {
    color: #94a3b8;
    font-size: 13px;
    margin-top: 6px;
}

.upload-panel {
    background: rgba(255,255,255,0.84);
    border: 1px solid rgba(203,213,225,0.9);
    border-radius: 24px;
    padding: 22px;
    box-shadow: 0 18px 55px rgba(15,23,42,0.07);
    margin-bottom: 18px;
}

.nav-panel {
    background: rgba(255,255,255,0.78);
    border: 1px solid rgba(203,213,225,0.9);
    border-radius: 24px;
    padding: 16px 18px;
    box-shadow: 0 18px 55px rgba(15,23,42,0.06);
    margin-bottom: 24px;
}

.capability-card {
    background: rgba(255,255,255,0.88);
    border: 1px solid #e2e8f0;
    border-radius: 24px;
    padding: 26px;
    min-height: 165px;
    box-shadow: 0 14px 45px rgba(15,23,42,0.055);
}

.capability-card h3 {
    margin-bottom: 10px;
}

.capability-card p {
    color: #475569;
    font-size: 15px;
    line-height: 1.62;
}

.status {
    background: #ecfdf5;
    color: #065f46;
    border: 1px solid #a7f3d0;
    padding: 12px 16px;
    border-radius: 16px;
    font-weight: 800;
    margin-bottom: 20px;
}

.stButton>button {
    background: #0f172a;
    color: white;
    border-radius: 12px;
    border: none;
    padding: 10px 18px;
    font-weight: 800;
}

.stDownloadButton>button {
    background: #2563eb;
    color: white;
    border-radius: 12px;
    border: none;
    padding: 10px 18px;
    font-weight: 800;
}

div[data-baseweb="select"] > div {
    background: #ffffff;
    border-radius: 12px;
    border-color: #cbd5e1;
}

[data-testid="stMetric"] {
    background: rgba(255,255,255,0.9);
    border: 1px solid #e2e8f0;
    border-radius: 18px;
    padding: 18px;
    box-shadow: 0 10px 30px rgba(15,23,42,0.05);
}

[data-testid="stMetricValue"] {
    color: #0f172a;
}

[data-testid="stMetricLabel"] {
    color: #64748b;
}
</style>

<div class="ambient-orb-one"></div>
<div class="ambient-orb-two"></div>
""", unsafe_allow_html=True)


st.markdown("""
<div class="app-shell">
    <div class="top-row">
        <div class="brand">
            <div class="brand-mark">V</div>
            <div>
                <div class="brand-name">VisionIX</div>
                <div class="brand-caption">Enterprise AI Analytics Platform</div>
            </div>
        </div>
        <div class="top-badge">Universal Data Intelligence</div>
    </div>
</div>
""", unsafe_allow_html=True)


st.markdown("""
<div class="hero">
    <div class="hero-content">
        <div class="hero-kicker">Enterprise AI Analytics Platform</div>
        <div class="vision-title">Vision<span>IX</span></div>
        <div class="tagline">
            Transform raw datasets into intelligent insights, machine learning models,
            executive dashboards, explainable AI, and business-ready reports.
        </div>
        <div class="hero-metrics">
            <div class="hero-metric"><div class="hero-metric-title">Validate</div><div class="hero-metric-label">Data quality intelligence</div></div>
            <div class="hero-metric"><div class="hero-metric-title">Model</div><div class="hero-metric-label">AutoML comparison</div></div>
            <div class="hero-metric"><div class="hero-metric-title">Explain</div><div class="hero-metric-label">XAI and feature drivers</div></div>
            <div class="hero-metric"><div class="hero-metric-title">Report</div><div class="hero-metric-label">Executive outputs</div></div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


st.markdown('<div class="upload-panel">', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload CSV dataset", type=["csv"])
st.markdown('</div>', unsafe_allow_html=True)


pages = [
    "Overview",
    "Data",
    "Analytics",
    "AI Models",
    "Dashboard",
    "Insights",
    "Reports"
]

st.markdown('<div class="nav-panel">', unsafe_allow_html=True)
page = st.radio("Navigation", pages, horizontal=True)
st.markdown('</div>', unsafe_allow_html=True)


def show_overview():
    st.markdown("## Platform capabilities")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="capability-card">
            <h3>Data Intelligence</h3>
            <p>Automated validation, cleaning, feature engineering, duplicate handling, missing value treatment, and outlier management.</p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="capability-card">
            <h3>AI Modeling</h3>
            <p>Universal AutoML workflow for classification and regression with model comparison, feature importance, and explainability.</p>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="capability-card">
            <h3>Decision Reporting</h3>
            <p>Interactive dashboards, KPI detection, automated insights, and professional executive report generation.</p>
        </div>
        """, unsafe_allow_html=True)


if uploaded_file:
    raw_df = load_data(uploaded_file)
    cleaned_df, cleaning_log = clean_data_agent(raw_df)
    engineered_df, feature_log = feature_engineering_agent(cleaned_df)
    process_log = cleaning_log + feature_log

    st.markdown('<div class="status">Dataset processed successfully. Cleaned and engineered data is ready.</div>', unsafe_allow_html=True)

    if page == "Overview":
        show_overview()

    elif page == "Data":
        tab1, tab2, tab3 = st.tabs(["Data Hub", "Validation", "Cleaning"])
        with tab1:
            show_data_hub(raw_df, cleaned_df)
        with tab2:
            show_validation_agent(raw_df)
        with tab3:
            show_cleaning_studio(raw_df, cleaned_df, cleaning_log)
            st.subheader("Feature Engineering Log")
            for item in feature_log:
                st.write(f"• {item}")
            st.subheader("Engineered Dataset Preview")
            st.dataframe(engineered_df.head(20), width="stretch")

    elif page == "Analytics":
        tab1, tab2 = st.tabs(["Analytics Studio", "KPI Agent"])
        with tab1:
            show_analytics_studio(engineered_df)
        with tab2:
            show_kpi_agent(engineered_df)

    elif page == "AI Models":
        tab1, tab2, tab3, tab4 = st.tabs(["ML Studio", "Model Comparison", "XAI Agent", "Feature Importance"])
        with tab1:
            show_ml_studio(engineered_df)
        with tab2:
            show_model_comparison_dashboard()
        with tab3:
            if (
                "best_model" in st.session_state
                and "model_results" in st.session_state
                and "problem_type" in st.session_state
                and "feature_names" in st.session_state
            ):
                show_xai_agent(
                    st.session_state["best_model"],
                    st.session_state["model_results"],
                    st.session_state["problem_type"],
                    st.session_state["feature_names"]
                )
            else:
                st.info("Run ML Studio first to generate model explanation.")
        with tab4:
            show_feature_importance_agent()

    elif page == "Dashboard":
        show_dashboard_studio(engineered_df)

    elif page == "Insights":
        tab1, tab2 = st.tabs(["Insight Engine", "Insight Pro"])
        with tab1:
            show_insight_engine(engineered_df)
        with tab2:
            show_business_insight_agent(engineered_df)

    elif page == "Reports":
        show_report_studio(engineered_df, process_log)

else:
    show_overview()
    st.info("Upload a CSV file above to start VisionIX.")