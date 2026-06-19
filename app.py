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
    max-width: 1180px;
    padding-top: 2.2rem;
    padding-bottom: 2.8rem;
}

.stApp {
    background:
        radial-gradient(circle at top right, rgba(37,99,235,0.14), transparent 32%),
        radial-gradient(circle at bottom left, rgba(14,165,233,0.10), transparent 28%),
        #0E1117;
    color: #FAFAFA;
}

h1, h2, h3 {
    color: #FAFAFA;
    letter-spacing: -0.035em;
}

p, span, label {
    color: #D1D5DB;
}

.topbar {
    background: #161A22;
    border: 1px solid #2A2F3A;
    border-radius: 20px;
    padding: 14px 20px;
    box-shadow: 0 16px 45px rgba(0,0,0,0.30);
    margin-bottom: 22px;
}

.topbar-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.brand {
    display: flex;
    align-items: center;
    gap: 12px;
}

.brand-logo {
    width: 40px;
    height: 40px;
    border-radius: 12px;
    background: linear-gradient(135deg, #2563EB, #0F172A);
    color: white;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 850;
    font-size: 18px;
}

.brand-title {
    font-size: 22px;
    font-weight: 850;
    line-height: 1;
    color: #FAFAFA;
}

.brand-caption {
    font-size: 12px;
    color: #9CA3AF;
    margin-top: 3px;
}

.badge {
    background: rgba(37,99,235,0.12);
    color: #93C5FD;
    border: 1px solid rgba(96,165,250,0.35);
    padding: 7px 13px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 750;
}

.hero {
    border-radius: 26px;
    padding: 42px 44px;
    background:
        linear-gradient(135deg, #111827 0%, #0F172A 45%, #1E293B 100%);
    box-shadow: 0 28px 80px rgba(0,0,0,0.38);
    margin-bottom: 22px;
    position: relative;
    overflow: hidden;
    border: 1px solid #334155;
}

.hero::after {
    content: "";
    position: absolute;
    width: 390px;
    height: 390px;
    right: -130px;
    top: -160px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(96,165,250,0.22), transparent 65%);
}

.hero-content {
    position: relative;
    z-index: 2;
}

.hero-kicker {
    color: #93C5FD;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 14px;
}

.hero-title {
    font-size: 52px;
    line-height: 0.98;
    font-weight: 900;
    color: #FFFFFF;
    letter-spacing: -0.055em;
}

.hero-title span {
    color: #60A5FA;
}

.hero-subtitle {
    margin-top: 18px;
    max-width: 780px;
    font-size: 17px;
    line-height: 1.65;
    color: #CBD5E1;
}

.hero-thought {
    margin-top: 18px;
    max-width: 760px;
    color: #93C5FD;
    font-size: 14px;
    font-weight: 650;
}

.hero-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-top: 26px;
}

.hero-card {
    background: rgba(255,255,255,0.055);
    border: 1px solid rgba(148,163,184,0.20);
    border-radius: 16px;
    padding: 15px;
}

.hero-card-title {
    color: white;
    font-size: 17px;
    font-weight: 800;
}

.hero-card-sub {
    color: #94A3B8;
    font-size: 12px;
    margin-top: 5px;
}

.panel {
    background: #161A22;
    border: 1px solid #2A2F3A;
    border-radius: 20px;
    padding: 18px;
    box-shadow: 0 16px 45px rgba(0,0,0,0.26);
    margin-bottom: 18px;
}

.nav-panel {
    background: #161A22;
    border: 1px solid #2A2F3A;
    border-radius: 18px;
    padding: 12px 16px;
    box-shadow: 0 12px 35px rgba(0,0,0,0.22);
    margin-bottom: 22px;
}

.capability-card {
    background: #161A22;
    border: 1px solid #2A2F3A;
    border-radius: 20px;
    padding: 22px;
    min-height: 150px;
    box-shadow: 0 14px 38px rgba(0,0,0,0.22);
}

.capability-card h3 {
    font-size: 20px;
    margin-bottom: 10px;
    color: #FAFAFA;
}

.capability-card p {
    font-size: 14px;
    line-height: 1.65;
    color: #B8C0CC;
}

.status {
    background: rgba(34,197,94,0.10);
    color: #86EFAC;
    border: 1px solid rgba(34,197,94,0.30);
    padding: 11px 14px;
    border-radius: 14px;
    font-weight: 700;
    font-size: 14px;
    margin-bottom: 18px;
}

.stButton>button {
    background: #2563EB;
    color: white;
    border-radius: 11px;
    border: none;
    padding: 9px 16px;
    font-weight: 750;
}

.stDownloadButton>button {
    background: #2563EB;
    color: white;
    border-radius: 11px;
    border: none;
    padding: 9px 16px;
    font-weight: 750;
}

div[data-baseweb="select"] > div {
    background: #1F2430;
    border-radius: 11px;
    border-color: #374151;
    color: #FAFAFA;
}

[data-testid="stMetric"] {
    background: #161A22;
    border: 1px solid #2A2F3A;
    border-radius: 16px;
    padding: 15px;
    box-shadow: 0 10px 26px rgba(0,0,0,0.22);
}

[data-testid="stMetricValue"] {
    color: #FAFAFA;
    font-size: 24px;
}

[data-testid="stMetricLabel"] {
    color: #9CA3AF;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}

.stTabs [data-baseweb="tab"] {
    background: #161A22;
    border-radius: 12px;
    color: #D1D5DB;
    border: 1px solid #2A2F3A;
    padding: 8px 14px;
}

.stTabs [aria-selected="true"] {
    background: #2563EB !important;
    color: white !important;
}

[data-testid="stFileUploader"] {
    color: #D1D5DB;
}

[data-testid="stDataFrame"] {
    border-radius: 14px;
    overflow: hidden;
}
</style>
""", unsafe_allow_html=True)


st.markdown("""
<div class="topbar">
    <div class="topbar-row">
        <div class="brand">
            <div class="brand-logo">V</div>
            <div>
                <div class="brand-title">VisionIX</div>
                <div class="brand-caption">Enterprise AI Analytics Platform</div>
            </div>
        </div>
        <div class="badge">Universal Data Intelligence</div>
    </div>
</div>
""", unsafe_allow_html=True)


st.markdown("""
<div class="hero">
    <div class="hero-content">
        <div class="hero-kicker">Enterprise AI Analytics Platform</div>
        <div class="hero-title">Vision<span>IX</span></div>
        <div class="hero-subtitle">
            Transform raw datasets into clean data, intelligent insights, machine learning models,
            interactive dashboards, explainable AI, and business-ready reports.
        </div>
        <div class="hero-thought">
            Built for modern decision-makers who want clarity from data, not complexity.
        </div>
        <div class="hero-grid">
            <div class="hero-card"><div class="hero-card-title">Validate</div><div class="hero-card-sub">Data quality checks</div></div>
            <div class="hero-card"><div class="hero-card-title">Model</div><div class="hero-card-sub">AutoML comparison</div></div>
            <div class="hero-card"><div class="hero-card-title">Explain</div><div class="hero-card-sub">XAI + feature drivers</div></div>
            <div class="hero-card"><div class="hero-card-title">Report</div><div class="hero-card-sub">Executive outputs</div></div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


st.markdown('<div class="panel">', unsafe_allow_html=True)
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