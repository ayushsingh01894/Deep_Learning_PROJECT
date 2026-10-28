import json
import os
import sys
import textwrap
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.booking_prediction.predict import artifacts_available as booking_ready
from src.common.config import METRICS_DIR
from src.demand_forecasting.predict import artifacts_available as demand_ready
from src.food_classification.predict import artifacts_available as food_ready
from src.sentiment_analysis.predict import artifacts_available as sentiment_ready

st.set_page_config(
    page_title="Smart Hospitality Management System",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

def _load_metric(filename, key):
    path = os.path.join(METRICS_DIR, filename)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get(key)
    except (json.JSONDecodeError, OSError):
        return None

st.markdown(
    textwrap.dedent("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600&family=Inter:wght@400;500;600;700&display=swap');

    :root {
        --bg: var(--background-color);
        --surface: var(--secondary-background-color);
        --text: var(--text-color);
        --muted: rgba(128,128,128,0.9);
        --border: rgba(128,128,128,0.25);
        --accent: #C7A24A;
        --accent-soft: rgba(199,162,74,0.13);
        --success: #2E8B57;
        --warning: #D98E3B;
    }

    html, body, [class*="css"] {
        font-family: "Inter", sans-serif;
    }

    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"] {
        background: var(--bg);
        color: var(--text);
    }

    h1, h2, h3 {
        font-family: "Fraunces", Georgia, serif !important;
        color: var(--text) !important;
    }

    header {
        background: transparent !important;
    }

    #MainMenu,
    footer {
        visibility: hidden;
    }

    [data-testid="stSidebar"] {
        background: var(--surface);
        border-right: 1px solid var(--border);
        min-width: 260px;
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1rem;
        overflow-y: auto !important;
        scrollbar-width: thin;
        scrollbar-color: rgba(128,128,128,0.55) transparent;
    }

    [data-testid="stSidebar"] > div:first-child::-webkit-scrollbar {
        width: 8px;
    }

    [data-testid="stSidebar"] > div:first-child::-webkit-scrollbar-track {
        background: transparent;
    }

    [data-testid="stSidebar"] > div:first-child::-webkit-scrollbar-thumb {
        background: rgba(128,128,128,0.45);
        border-radius: 20px;
    }

    [data-testid="stSidebar"] > div:first-child::-webkit-scrollbar-thumb:hover {
        background: rgba(128,128,128,0.75);
    }

    [data-testid="stSidebarNav"] {
        padding-bottom: 1rem;
    }

    [data-testid="stSidebarNav"] a {
        border-radius: 8px;
        margin: 4px 8px;
        transition: all 0.2s ease;
    }

    [data-testid="stSidebarNav"] a:hover {
        background: var(--accent-soft);
    }

    [data-testid="stSidebarNav"] a[aria-current="page"] {
        background: var(--accent-soft);
        border-left: 3px solid var(--accent);
    }

    html {
        scrollbar-width: thin;
        scrollbar-color: rgba(128,128,128,0.55) transparent;
    }

    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: transparent;
    }

    ::-webkit-scrollbar-thumb {
        background: rgba(128,128,128,0.45);
        border-radius: 20px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: rgba(128,128,128,0.75);
    }

    .hero {
        padding: 0.5rem 0 1.75rem 0;
        border-bottom: 1px solid var(--border);
        margin-bottom: 2rem;
    }

    .hero-kicker {
        color: var(--accent) !important;
        font-size: 0.95rem;
        font-weight: 600;
        margin-bottom: 0.4rem;
    }

    .hero-title {
        font-family: "Fraunces", Georgia, serif;
        font-weight: 600;
        font-size: clamp(2rem, 4vw, 2.7rem);
        color: var(--text) !important;
        line-height: 1.1;
        margin: 0 0 0.65rem 0;
    }

    .hero-sub {
        color: var(--muted) !important;
        font-size: 1.05rem;
        max-width: 720px;
        line-height: 1.65;
        margin-bottom: 1.5rem;
    }

    .factstrip {
        display: flex;
        flex-wrap: wrap;
    }

    .fact {
        padding: 0 1.6rem;
        border-left: 1px solid var(--border);
    }

    .fact:first-child {
        padding-left: 0;
        border-left: none;
    }

    .fact-value {
        font-family: "Fraunces", Georgia, serif;
        font-size: 1.5rem;
        color: var(--text) !important;
        font-weight: 600;
    }

    .fact-label {
        color: var(--muted) !important;
        font-size: 0.85rem;
        margin-top: 0.15rem;
    }

    .ledger-heading {
        color: var(--text) !important;
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 0.2rem;
    }

    .ledger-subheading {
        color: var(--muted) !important;
        font-size: 0.92rem;
        margin-bottom: 1.2rem;
    }

    .ledger-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        padding: 1.15rem 0;
        border-bottom: 1px solid var(--border);
        transition: all 0.2s ease;
    }

    .ledger-row:hover {
        background: var(--accent-soft);
        border-radius: 8px;
        padding-left: 0.6rem;
        padding-right: 0.6rem;
    }

    .ledger-left {
        display: flex;
        align-items: flex-start;
        gap: 1rem;
        min-width: 0;
    }

    .ledger-num {
        font-family: "Fraunces", Georgia, serif;
        color: var(--muted) !important;
        opacity: 0.55;
        font-size: 1.4rem;
        font-weight: 600;
        min-width: 2rem;
    }

    .ledger-title {
        color: var(--text) !important;
        font-size: 1.08rem;
        font-weight: 600;
        margin-bottom: 0.15rem;
    }

    .ledger-desc {
        color: var(--muted) !important;
        font-size: 0.92rem;
        max-width: 560px;
        line-height: 1.5;
    }

    .ledger-right {
        display: flex;
        align-items: center;
        gap: 0.9rem;
        flex-shrink: 0;
    }

    .arch-tag {
        color: var(--muted) !important;
        font-size: 0.85rem;
        white-space: nowrap;
    }

    .pill {
        padding: 0.3rem 0.8rem;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 600;
        white-space: nowrap;
    }

    .pill-ready {
        background: rgba(46,139,87,0.12);
        color: var(--success) !important;
        border: 1px solid rgba(46,139,87,0.35);
    }

    .pill-pending {
        background: rgba(217,142,59,0.12);
        color: var(--warning) !important;
        border: 1px solid rgba(217,142,59,0.35);
    }

    .about-heading {
        color: var(--text) !important;
        font-size: 0.98rem;
        font-weight: 600;
        margin-bottom: 0.6rem;
    }

    .about-list {
        color: var(--muted) !important;
        font-size: 0.9rem;
        line-height: 1.75;
        padding-left: 1.1rem;
        margin: 0;
    }

    .about-list li::marker {
        color: var(--accent);
    }

    .about-list code {
        background: var(--accent-soft);
        color: var(--text) !important;
        padding: 0.15rem 0.35rem;
        border-radius: 4px;
    }

    @media (max-width: 900px) {
        .ledger-row {
            align-items: flex-start;
        }

        .ledger-right {
            flex-direction: column;
            align-items: flex-end;
            gap: 0.4rem;
        }

        .fact {
            padding: 0.5rem 1rem;
        }

        .fact:first-child {
            padding-left: 0;
        }
    }

    @media (max-width: 640px) {
        .hero-title {
            font-size: 2rem;
        }

        .hero-sub {
            font-size: 0.95rem;
        }

        .ledger-row {
            flex-direction: column;
        }

        .ledger-right {
            width: 100%;
            flex-direction: row;
            justify-content: flex-start;
        }

        .ledger-desc {
            max-width: 100%;
        }

        .fact {
            padding-left: 0;
            border-left: none;
            min-width: 45%;
        }
    }
    </style>
    """),
    unsafe_allow_html=True
)

ready_flags = {
    "booking": booking_ready(),
    "sentiment": sentiment_ready(),
    "demand": demand_ready(),
    "food": food_ready()
}

ready_count = sum(ready_flags.values())

booking_test_n = _load_metric(
    "booking_prediction_metrics.json",
    "test_set_size"
)

sentiment_test_n = _load_metric(
    "sentiment_analysis_metrics.json",
    "test_set_size"
)

demand_days_n = _load_metric(
    "demand_forecasting_metrics.json",
    "test_set_size_days"
)

hero_html = f"""
<div class="hero">
    <div class="hero-kicker">Final Year Project · Deep Learning</div>
    <div class="hero-title">Smart Hospitality Management System</div>
    <div class="hero-sub">
        Four Deep Learning modules, unified under one hospitality platform —
        booking risk, guest sentiment, room demand, and menu imagery —
        built on real, publicly sourced hotel data rather than treated as
        four separate projects.
    </div>
    <div class="factstrip">
        <div class="fact">
            <div class="fact-value">{ready_count} of 4</div>
            <div class="fact-label">modules trained on real data</div>
        </div>
        <div class="fact">
            <div class="fact-value">{f"{booking_test_n:,}" if booking_test_n else "—"}</div>
            <div class="fact-label">held-out bookings evaluated</div>
        </div>
        <div class="fact">
            <div class="fact-value">{f"{sentiment_test_n:,}" if sentiment_test_n else "—"}</div>
            <div class="fact-label">held-out reviews evaluated</div>
        </div>
        <div class="fact">
            <div class="fact-value">{demand_days_n if demand_days_n else "—"}</div>
            <div class="fact-label">days of demand history</div>
        </div>
    </div>
</div>
"""

st.markdown(
    textwrap.dedent(hero_html),
    unsafe_allow_html=True
)

st.markdown(
    '<div class="ledger-heading">System Directory</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="ledger-subheading">Open a module from the sidebar to use it.</div>',
    unsafe_allow_html=True
)

modules = [
    (
        "01",
        "Hotel Booking Prediction",
        "ANN",
        "Predicts whether an incoming booking is likely to be canceled, from booking details.",
        ready_flags["booking"]
    ),
    (
        "02",
        "Hotel Review Sentiment Analysis",
        "BiLSTM",
        "Classifies a guest review as positive or negative sentiment.",
        ready_flags["sentiment"]
    ),
    (
        "03",
        "Room Demand Forecasting",
        "LSTM",
        "Forecasts upcoming daily room demand from historical booking patterns.",
        ready_flags["demand"]
    ),
    (
        "04",
        "Food Image Classification",
        "CNN · Transfer Learning",
        "Classifies an uploaded food photo into a hotel-menu food category.",
        ready_flags["food"]
    )
]

for num, title, arch, desc, ready in modules:
    pill_class = "pill-ready" if ready else "pill-pending"
    pill_text = "Ready" if ready else "Pending"

    module_html = f"""
    <div class="ledger-row">
        <div class="ledger-left">
            <div class="ledger-num">{num}</div>
            <div>
                <div class="ledger-title">{title}</div>
                <div class="ledger-desc">{desc}</div>
            </div>
        </div>
        <div class="ledger-right">
            <div class="arch-tag">{arch}</div>
            <div class="pill {pill_class}">{pill_text}</div>
        </div>
    </div>
    """

    st.markdown(
        textwrap.dedent(module_html),
        unsafe_allow_html=True
    )

st.write("")
st.write("")

about_col1, about_col2 = st.columns(2)

with about_col1:
    st.markdown(
        '<div class="about-heading">Data sources</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <ul class="about-list">
            <li>Modules 01 &amp; 03 share one dataset — Hotel Booking Demand (Kaggle) — used per-booking for cancellation risk, and aggregated into a daily series for demand forecasting.</li>
            <li>Module 02 uses the 515K Hotel Reviews Data in Europe dataset (Kaggle).</li>
            <li>Module 04 uses a documented 12-class subset of Food-101 relevant to a hotel menu and breakfast buffet.</li>
        </ul>
        """,
        unsafe_allow_html=True
    )

with about_col2:
    st.markdown(
        '<div class="about-heading">Module shows "Pending"</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <ul class="about-list">
            <li>Download that module's dataset (see the project README for exact links).</li>
            <li>Run its <code>preprocess.py</code>, then <code>train.py</code>, then <code>evaluate.py</code>.</li>
            <li>Reload this page — the directory above reflects live model status.</li>
        </ul>
        """,
        unsafe_allow_html=True
    )