"""
app.py
------
Smart Data Analyzer – No-Code AI Tool
Main Streamlit application entry point.

Run with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import os
from dotenv import load_dotenv

# ── Load environment variables (.env file) ────────────────────────────────────
load_dotenv()

# ── Internal modules ──────────────────────────────────────────────────────────
from utils.data_processor import (
    load_csv,
    get_basic_info,
    get_statistics,
    get_numeric_columns,
    get_categorical_columns,
    build_summary_text,
    detect_important_columns,
    get_column_value_counts,
)
from utils.visualizer import (
    plot_histogram,
    plot_bar_chart,
    plot_correlation_heatmap,
    plot_missing_values,
    plot_scatter,
)
from utils.ai_helper import generate_insights, chat_with_data

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG — must be the FIRST Streamlit call
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Smart Data Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS — dark SaaS dashboard theme
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

/* ── Root palette ── */
:root {
    --bg-primary:   #0d0f1a;
    --bg-secondary: #131627;
    --bg-card:      #1a1d2e;
    --bg-card-hover:#1f2340;
    --border:       #252840;
    --accent:       #6366f1;
    --accent-2:     #8b5cf6;
    --accent-3:     #ec4899;
    --text-primary: #f1f5f9;
    --text-muted:   #94a3b8;
    --success:      #10b981;
    --warning:      #f59e0b;
    --danger:       #ef4444;
}

/* ── Global reset ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: var(--bg-primary);
    color: var(--text-primary);
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem 3rem; max-width: 1400px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] .stMarkdown p {
    color: var(--text-muted);
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

/* ── Sidebar nav items ── */
.nav-item {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.6rem 0.9rem;
    border-radius: 10px;
    cursor: pointer;
    color: var(--text-muted);
    font-size: 0.9rem;
    font-weight: 500;
    transition: all 0.2s;
    margin-bottom: 0.25rem;
    border: 1px solid transparent;
    text-decoration: none;
}
.nav-item:hover, .nav-item.active {
    background: rgba(99,102,241,0.12);
    color: var(--text-primary);
    border-color: rgba(99,102,241,0.3);
}
.nav-item.active { color: #818cf8; }

/* ── Metric cards ── */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.25rem 1.5rem;
    transition: all 0.2s;
}
.metric-card:hover { border-color: var(--accent); background: var(--bg-card-hover); }
.metric-label {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-muted);
    margin-bottom: 0.4rem;
}
.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: var(--text-primary);
    font-family: 'Space Mono', monospace;
    line-height: 1.1;
}
.metric-sub {
    font-size: 0.75rem;
    color: var(--text-muted);
    margin-top: 0.25rem;
}

/* ── Section headers ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin: 2rem 0 1.2rem;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid var(--border);
}
.section-title {
    font-size: 1.2rem;
    font-weight: 700;
    color: var(--text-primary);
}
.section-badge {
    background: rgba(99,102,241,0.15);
    color: #818cf8;
    padding: 0.15rem 0.5rem;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 600;
    border: 1px solid rgba(99,102,241,0.3);
}

/* ── Data tables ── */
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, var(--accent), var(--accent-2));
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.5rem 1.4rem;
    font-weight: 600;
    font-size: 0.88rem;
    letter-spacing: 0.02em;
    transition: all 0.2s;
    box-shadow: 0 4px 15px rgba(99,102,241,0.25);
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(99,102,241,0.4);
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: var(--bg-card);
    border: 2px dashed var(--border);
    border-radius: 16px;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover { border-color: var(--accent); }

/* ── Text input & textarea ── */
.stTextInput input, .stTextArea textarea {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
    font-size: 0.9rem !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(99,102,241,0.15) !important;
}

/* ── Selectbox ── */
.stSelectbox div[data-baseweb="select"] > div {
    background: var(--bg-card) !important;
    border-color: var(--border) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
}

/* ── Insight box ── */
.insight-box {
    background: linear-gradient(135deg, rgba(99,102,241,0.08), rgba(139,92,246,0.08));
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 16px;
    padding: 1.5rem 1.75rem;
    line-height: 1.75;
    font-size: 0.93rem;
    color: var(--text-primary);
}

/* ── Chat bubbles ── */
.chat-user {
    background: linear-gradient(135deg, rgba(99,102,241,0.2), rgba(139,92,246,0.2));
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 16px 16px 4px 16px;
    padding: 0.8rem 1.1rem;
    margin: 0.5rem 0 0.5rem 3rem;
    font-size: 0.9rem;
    color: var(--text-primary);
}
.chat-ai {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px 16px 16px 4px;
    padding: 0.8rem 1.1rem;
    margin: 0.5rem 3rem 0.5rem 0;
    font-size: 0.9rem;
    color: var(--text-primary);
    line-height: 1.65;
}
.chat-label {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.25rem;
}
.chat-label.user { color: #818cf8; }
.chat-label.ai   { color: var(--success); }

/* ── Logo / brand ── */
.brand {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.5rem 0 1.5rem;
}
.brand-icon {
    width: 36px; height: 36px;
    background: linear-gradient(135deg, var(--accent), var(--accent-3));
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.1rem;
}
.brand-name {
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--text-primary);
    line-height: 1.1;
}
.brand-tagline {
    font-size: 0.68rem;
    color: var(--text-muted);
    font-weight: 400;
}

/* ── Status pill ── */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.2rem 0.7rem;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
}
.status-pill.ok {
    background: rgba(16,185,129,0.12);
    color: var(--success);
    border: 1px solid rgba(16,185,129,0.25);
}
.status-pill.warn {
    background: rgba(245,158,11,0.12);
    color: var(--warning);
    border: 1px solid rgba(245,158,11,0.25);
}
.status-pill.err {
    background: rgba(239,68,68,0.12);
    color: var(--danger);
    border: 1px solid rgba(239,68,68,0.25);
}

/* ── Tabs ── */
[data-baseweb="tab-list"] { background: var(--bg-secondary) !important; border-radius: 12px !important; padding: 4px !important; gap: 4px !important; }
[data-baseweb="tab"]      { border-radius: 9px !important; color: var(--text-muted) !important; font-size: 0.85rem !important; font-weight: 500 !important; }
[aria-selected="true"]    { background: var(--bg-card) !important; color: var(--text-primary) !important; }

/* ── Expander ── */
[data-testid="stExpander"] {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE INITIALISATION
# ══════════════════════════════════════════════════════════════════════════════
for key, default in {
    "df": None,
    "data_summary": None,
    "insights": None,
    "chat_history": [],
    "current_page": "upload",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:

    # Brand logo
    st.markdown("""
    <div class="brand">
        <div class="brand-icon">🧠</div>
        <div>
            <div class="brand-name">DataSense AI</div>
            <div class="brand-tagline">No-Code Data Intelligence</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p style="color:#94a3b8;font-size:0.7rem;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.5rem">Navigation</p>', unsafe_allow_html=True)

    nav_items = [
        ("upload",   "📂", "Upload Dataset"),
        ("overview", "📊", "Data Overview"),
        ("viz",      "📈", "Visualizations"),
        ("ai",       "🤖", "AI Insights"),
        ("chat",     "💬", "Chat with Data"),
    ]

    for page_id, icon, label in nav_items:
        is_active = st.session_state.current_page == page_id
        if st.button(
            f"{icon}  {label}",
            key=f"nav_{page_id}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
        ):
            st.session_state.current_page = page_id
            st.rerun()

    st.markdown("---")

    # Dataset status
    if st.session_state.df is not None:
        df = st.session_state.df
        st.markdown("""
        <p style="color:#94a3b8;font-size:0.7rem;text-transform:uppercase;
        letter-spacing:0.1em;margin-bottom:0.5rem">Dataset</p>
        """, unsafe_allow_html=True)

        missing_total = int(df.isnull().sum().sum())
        st.markdown(f"""
        <div style="background:#1a1d2e;border:1px solid #252840;border-radius:12px;padding:0.9rem 1rem">
            <div style="font-size:0.78rem;color:#f1f5f9;font-weight:600;margin-bottom:0.5rem">📄 Loaded</div>
            <div style="font-size:0.75rem;color:#94a3b8">{df.shape[0]:,} rows × {df.shape[1]} cols</div>
            <div style="margin-top:0.5rem">
                <span class="status-pill {'warn' if missing_total > 0 else 'ok'}">
                    {'⚠ ' + str(missing_total) + ' missing' if missing_total > 0 else '✓ No missing values'}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.7rem;color:#475569;text-align:center;padding:0.5rem 0">
        Built with ❤️ · Powered by Claude AI<br>
        <span style="color:#6366f1">DataSense AI v1.0</span>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HELPER: Section Header
# ══════════════════════════════════════════════════════════════════════════════
def section_header(icon, title, badge=None):
    badge_html = f'<span class="section-badge">{badge}</span>' if badge else ""
    st.markdown(f"""
    <div class="section-header">
        <span style="font-size:1.3rem">{icon}</span>
        <span class="section-title">{title}</span>
        {badge_html}
    </div>
    """, unsafe_allow_html=True)


def metric_card(label, value, sub=None):
    sub_html = f'<div class="metric-sub">{sub}</div>' if sub else ""
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {sub_html}
    </div>
    """


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — UPLOAD
# ══════════════════════════════════════════════════════════════════════════════
def page_upload():
    # Hero banner
    st.markdown("""
    <div style="background:linear-gradient(135deg,rgba(99,102,241,0.12),rgba(236,72,153,0.08));
                border:1px solid rgba(99,102,241,0.2);border-radius:20px;
                padding:2.5rem 2.5rem 2rem;margin-bottom:2rem">
        <div style="font-size:2.2rem;font-weight:800;color:#f1f5f9;margin-bottom:0.4rem">
            Upload Your Dataset
        </div>
        <div style="font-size:1rem;color:#94a3b8;max-width:520px;line-height:1.6">
            Drop in any CSV file and let our AI instantly analyze it — no code required.
            Get statistics, charts, and natural-language insights in seconds.
        </div>
        <div style="display:flex;gap:0.75rem;margin-top:1.2rem;flex-wrap:wrap">
            <span class="status-pill ok">✓ Instant Analysis</span>
            <span class="status-pill ok">✓ AI-Powered Insights</span>
            <span class="status-pill ok">✓ Interactive Charts</span>
            <span class="status-pill ok">✓ Chat Interface</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_upload, col_tip = st.columns([2, 1], gap="large")

    with col_upload:
        section_header("📂", "Upload CSV File")
        uploaded_file = st.file_uploader(
            "Drag & drop your CSV here, or click to browse",
            type=["csv"],
            help="Maximum file size: 200MB. Supports UTF-8 encoded CSV files.",
        )

        if uploaded_file is not None:
            try:
                with st.spinner("🔍 Reading your dataset..."):
                    df = load_csv(uploaded_file)
                    st.session_state.df = df
                    st.session_state.data_summary = build_summary_text(df)
                    st.session_state.insights = None  # reset on new upload
                    st.session_state.chat_history = []

                st.success(f"✅ File loaded successfully — **{uploaded_file.name}**")

                # Preview
                section_header("👁️", "Dataset Preview", f"First 5 rows")
                st.dataframe(
                    df.head(5),
                    use_container_width=True,
                    hide_index=False,
                )

                # Quick stats
                info = get_basic_info(df)
                num_cols = get_numeric_columns(df)
                cat_cols = get_categorical_columns(df)
                missing = sum(info["missing_values"].values())

                c1, c2, c3, c4 = st.columns(4)
                cards = [
                    (c1, "Rows",        f"{info['num_rows']:,}",    "Total records"),
                    (c2, "Columns",     str(info["num_cols"]),       "Features"),
                    (c3, "Numeric",     str(len(num_cols)),          "Numeric columns"),
                    (c4, "Missing",     str(missing),               "Null values"),
                ]
                for col, label, val, sub in cards:
                    col.markdown(metric_card(label, val, sub), unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("🚀  Start Full Analysis", use_container_width=True):
                    st.session_state.current_page = "overview"
                    st.rerun()

            except ValueError as e:
                st.error(f"❌ {str(e)}")
            except Exception as e:
                st.error(f"❌ Unexpected error: {str(e)}")

    with col_tip:
        section_header("💡", "Quick Tips")
        st.markdown("""
        <div style="background:#1a1d2e;border:1px solid #252840;border-radius:14px;padding:1.2rem">
            <div style="font-size:0.85rem;color:#94a3b8;line-height:1.9">
                <b style="color:#f1f5f9">Supported formats</b><br>
                Standard CSV (comma-separated)<br><br>
                <b style="color:#f1f5f9">For best results</b><br>
                • First row = column headers<br>
                • UTF-8 encoding<br>
                • Remove password protection<br><br>
                <b style="color:#f1f5f9">Works great for</b><br>
                • Sales & financial data<br>
                • Survey results<br>
                • Scientific datasets<br>
                • Customer records<br>
                • Log files
            </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DATA OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
def page_overview():
    df = st.session_state.df
    info = get_basic_info(df)
    stats = get_statistics(df)
    num_cols = get_numeric_columns(df)
    cat_cols = get_categorical_columns(df)

    section_header("📊", "Data Overview", f"{info['num_rows']:,} rows")

    # Top metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    missing_total = sum(info["missing_values"].values())
    dupes = info["duplicate_rows"]
    cards = [
        (c1, "Total Rows",    f"{info['num_rows']:,}", None),
        (c2, "Total Columns", str(info["num_cols"]),   None),
        (c3, "Numeric Cols",  str(len(num_cols)),      None),
        (c4, "Missing Vals",  str(missing_total),      "⚠ review" if missing_total > 0 else "✓ clean"),
        (c5, "Duplicates",    str(dupes),               "⚠ review" if dupes > 0 else "✓ none"),
    ]
    for col, label, val, sub in cards:
        col.markdown(metric_card(label, val, sub), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📋  Column Info", "📐  Statistics", "🔍  Missing Values"])

    # ── Tab 1: Column info ─────────────────────────────────────────────────
    with tab1:
        col_data = []
        for col in df.columns:
            missing_n   = int(df[col].isnull().sum())
            missing_pct = round(missing_n / len(df) * 100, 1)
            unique_n    = int(df[col].nunique())
            dtype       = str(df[col].dtype)
            sample      = str(df[col].dropna().iloc[0]) if not df[col].dropna().empty else "—"
            col_data.append({
                "Column":      col,
                "Type":        dtype,
                "Missing":     f"{missing_n} ({missing_pct}%)",
                "Unique Values": unique_n,
                "Sample Value":  sample[:50],
            })
        col_df = pd.DataFrame(col_data)
        st.dataframe(col_df, use_container_width=True, hide_index=True, height=380)

    # ── Tab 2: Statistics ──────────────────────────────────────────────────
    with tab2:
        if stats is not None:
            st.dataframe(stats.style.background_gradient(cmap="Blues", axis=0),
                         use_container_width=True, height=380)
        else:
            st.info("No numeric columns found in this dataset.")

    # ── Tab 3: Missing values ──────────────────────────────────────────────
    with tab3:
        missing_cols = {k: v for k, v in info["missing_values"].items() if v > 0}
        if not missing_cols:
            st.markdown("""
            <div style="text-align:center;padding:2rem;color:#10b981;font-size:1rem">
                🎉 No missing values detected — your dataset is clean!
            </div>
            """, unsafe_allow_html=True)
        else:
            fig_miss = plot_missing_values(df)
            if fig_miss:
                st.pyplot(fig_miss)

            miss_df = pd.DataFrame({
                "Column":       list(missing_cols.keys()),
                "Missing Count": list(missing_cols.values()),
                "Missing %":    [f"{v/len(df)*100:.1f}%" for v in missing_cols.values()],
                "Severity":     ["🔴 High" if v/len(df) > 0.3 else "🟡 Medium" if v/len(df) > 0.1 else "🟢 Low"
                                 for v in missing_cols.values()],
            })
            st.dataframe(miss_df, use_container_width=True, hide_index=True)

    # ── Important columns ──────────────────────────────────────────────────
    important = detect_important_columns(df)
    if important["numeric"] or important["categorical"]:
        section_header("⭐", "Auto-Detected Important Columns")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">High Variance Numeric</div>
                <div style="margin-top:0.5rem">
                    {''.join([f'<span class="status-pill ok" style="margin:2px">{c}</span>'
                               for c in important["numeric"]]) or '<span style="color:#475569">None</span>'}
                </div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Good Categorical Groups</div>
                <div style="margin-top:0.5rem">
                    {''.join([f'<span class="status-pill warn" style="margin:2px">{c}</span>'
                               for c in important["categorical"][:5]]) or '<span style="color:#475569">None</span>'}
                </div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════════════════
def page_visualizations():
    df = st.session_state.df
    num_cols = get_numeric_columns(df)
    cat_cols = get_categorical_columns(df)

    section_header("📈", "Visualizations")

    tab_hist, tab_bar, tab_corr, tab_scatter = st.tabs(
        ["📊  Histograms", "📋  Bar Charts", "🔥  Correlation", "🔵  Scatter Plot"]
    )

    # ── Histograms ─────────────────────────────────────────────────────────
    with tab_hist:
        if not num_cols:
            st.info("No numeric columns found.")
        else:
            selected_num = st.selectbox(
                "Select a numeric column:", num_cols, key="hist_col"
            )
            if selected_num:
                with st.spinner("Generating histogram..."):
                    fig = plot_histogram(df, selected_num)
                    st.pyplot(fig)

                # Quick stats below chart
                data = df[selected_num].dropna()
                c1, c2, c3, c4 = st.columns(4)
                c1.markdown(metric_card("Mean",   f"{data.mean():.2f}"),   unsafe_allow_html=True)
                c2.markdown(metric_card("Median", f"{data.median():.2f}"), unsafe_allow_html=True)
                c3.markdown(metric_card("Std Dev", f"{data.std():.2f}"),   unsafe_allow_html=True)
                c4.markdown(metric_card("Skewness", f"{data.skew():.3f}"), unsafe_allow_html=True)

    # ── Bar charts ─────────────────────────────────────────────────────────
    with tab_bar:
        if not cat_cols:
            st.info("No categorical columns found.")
        else:
            selected_cat = st.selectbox(
                "Select a categorical column:", cat_cols, key="bar_col"
            )
            top_n = st.slider("Show top N values:", 5, 25, 10, key="bar_topn")
            if selected_cat:
                with st.spinner("Generating bar chart..."):
                    fig = plot_bar_chart(df, selected_cat, top_n=top_n)
                    st.pyplot(fig)

                counts = get_column_value_counts(df, selected_cat, top_n)
                st.dataframe(
                    counts.rename("Count").reset_index().rename(columns={"index": selected_cat}),
                    use_container_width=True, hide_index=True
                )

    # ── Correlation heatmap ────────────────────────────────────────────────
    with tab_corr:
        if len(num_cols) < 2:
            st.info("Need at least 2 numeric columns for a correlation heatmap.")
        else:
            with st.spinner("Computing correlations..."):
                fig = plot_correlation_heatmap(df)
            if fig:
                st.pyplot(fig)

                # Top correlations table
                corr_matrix = df[num_cols].corr().abs()
                upper = corr_matrix.where(
                    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
                )
                top_corr = (
                    upper.stack()
                    .sort_values(ascending=False)
                    .head(10)
                    .reset_index()
                )
                top_corr.columns = ["Column A", "Column B", "Correlation"]
                top_corr["Correlation"] = top_corr["Correlation"].round(4)
                st.markdown("**Top 10 Correlations**")
                st.dataframe(top_corr, use_container_width=True, hide_index=True)

    # ── Scatter plot ───────────────────────────────────────────────────────
    with tab_scatter:
        if len(num_cols) < 2:
            st.info("Need at least 2 numeric columns for a scatter plot.")
        else:
            c1, c2 = st.columns(2)
            col_x = c1.selectbox("X-axis:", num_cols, key="scatter_x")
            col_y = c2.selectbox("Y-axis:", num_cols,
                                  index=min(1, len(num_cols) - 1), key="scatter_y")
            if col_x and col_y and col_x != col_y:
                with st.spinner("Plotting scatter..."):
                    fig = plot_scatter(df, col_x, col_y)
                    st.pyplot(fig)

                corr_val = df[col_x].corr(df[col_y])
                st.markdown(f"""
                <div class="status-pill {'ok' if abs(corr_val) > 0.5 else 'warn'}">
                    Pearson r = {corr_val:.4f}
                    {'  · Strong correlation' if abs(corr_val) > 0.7
                     else '  · Moderate' if abs(corr_val) > 0.4
                     else '  · Weak'}
                </div>
                """, unsafe_allow_html=True)
            elif col_x == col_y:
                st.warning("Please select two different columns.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — AI INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
def page_ai_insights():
    df = st.session_state.df

    section_header("🤖", "AI Insights", "Powered by Claude")

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")

    if not api_key:
        st.warning("""
        ⚠️ **API Key Missing** — Add your `ANTHROPIC_API_KEY` to a `.env` file
        in the project root to enable AI features.
        ```
        ANTHROPIC_API_KEY=your_key_here
        ```
        """)
        return

    c1, c2 = st.columns([2, 1], gap="large")

    with c1:
        if st.session_state.insights:
            st.markdown('<div class="insight-box">', unsafe_allow_html=True)
            st.markdown(st.session_state.insights)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            col_btn, col_dl = st.columns([1, 1])
            with col_btn:
                if st.button("🔄  Regenerate Insights"):
                    st.session_state.insights = None
                    st.rerun()
            with col_dl:
                insights_bytes = st.session_state.insights.encode("utf-8")
                st.download_button(
                    "⬇️  Download Insights (.txt)",
                    data=insights_bytes,
                    file_name="ai_insights.txt",
                    mime="text/plain",
                    use_container_width=True,
                )
        else:
            st.markdown("""
            <div style="background:#1a1d2e;border:2px dashed #252840;border-radius:16px;
                        padding:2.5rem;text-align:center;color:#94a3b8">
                <div style="font-size:2.5rem;margin-bottom:0.75rem">🤖</div>
                <div style="font-size:1rem;font-weight:600;color:#f1f5f9;margin-bottom:0.4rem">
                    Ready to Analyze
                </div>
                <div style="font-size:0.85rem;margin-bottom:1.2rem">
                    Click the button below to generate AI-powered insights<br>
                    about your dataset in plain English.
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("✨  Generate AI Insights", use_container_width=True):
                with st.spinner("🧠 Analyzing your dataset with Claude AI..."):
                    try:
                        insights = generate_insights(st.session_state.data_summary)
                        st.session_state.insights = insights
                        st.rerun()
                    except EnvironmentError as e:
                        st.error(str(e))
                    except Exception as e:
                        st.error(f"AI error: {str(e)}")

    with c2:
        section_header("📝", "Dataset Summary")
        st.markdown(f"""
        <div style="background:#1a1d2e;border:1px solid #252840;border-radius:14px;
                    padding:1.1rem;font-size:0.78rem;color:#94a3b8;line-height:1.8;
                    max-height:400px;overflow-y:auto;font-family:'Space Mono',monospace">
            {st.session_state.data_summary.replace('. ', '.<br>')}
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — CHAT
# ══════════════════════════════════════════════════════════════════════════════
def page_chat():
    df = st.session_state.df

    section_header("💬", "Chat with Your Data", "AI Assistant")

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        st.warning("⚠️ API Key required. See AI Insights page for setup instructions.")
        return

    # Chat history display
    chat_container = st.container()
    with chat_container:
        if not st.session_state.chat_history:
            st.markdown("""
            <div style="background:linear-gradient(135deg,rgba(16,185,129,0.08),rgba(99,102,241,0.08));
                        border:1px solid rgba(16,185,129,0.2);border-radius:16px;
                        padding:1.5rem;text-align:center;color:#94a3b8;margin-bottom:1rem">
                <div style="font-size:1.5rem;margin-bottom:0.4rem">💬</div>
                <div style="color:#f1f5f9;font-weight:600;margin-bottom:0.3rem">Ask anything about your data</div>
                <div style="font-size:0.83rem">Try: "What is the average of [column]?" or "Which column has the most missing values?"</div>
            </div>
            """, unsafe_allow_html=True)

            # Suggestion chips
            suggestions = [
                "What columns are in this dataset?",
                "What is the average of the numeric columns?",
                "Are there any anomalies in the data?",
                "What trends can you identify?",
            ]
            cols = st.columns(2)
            for i, s in enumerate(suggestions):
                if cols[i % 2].button(f"💡 {s}", key=f"sug_{i}", use_container_width=True):
                    st.session_state._pending_question = s
                    st.rerun()
        else:
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    st.markdown(f"""
                    <div class="chat-user">
                        <div class="chat-label user">You</div>
                        {msg["content"]}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="chat-ai">
                        <div class="chat-label ai">🤖 DataSense AI</div>
                        {msg["content"]}
                    </div>
                    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Input
    col_input, col_send = st.columns([5, 1])
    with col_input:
        user_question = st.text_input(
            "Ask a question:",
            placeholder="e.g. Which column has the highest variance?",
            label_visibility="collapsed",
            key="chat_input",
        )
    with col_send:
        send_btn = st.button("Send ➤", use_container_width=True)

    # Handle pending question (from suggestion chips)
    if hasattr(st.session_state, "_pending_question"):
        user_question = st.session_state._pending_question
        del st.session_state._pending_question
        send_btn = True

    if send_btn and user_question.strip():
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        with st.spinner("🤔 Thinking..."):
            try:
                answer = chat_with_data(
                    question=user_question,
                    data_summary=st.session_state.data_summary,
                    chat_history=st.session_state.chat_history[:-1],
                )
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": f"⚠️ Error: {str(e)}"
                })
        st.rerun()

    # Clear chat
    if st.session_state.chat_history:
        if st.button("🗑️  Clear Chat", use_container_width=False):
            st.session_state.chat_history = []
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# ROUTING — Guard: require upload for all pages except upload
# ══════════════════════════════════════════════════════════════════════════════
page = st.session_state.current_page

if page != "upload" and st.session_state.df is None:
    st.markdown("""
    <div style="background:rgba(245,158,11,0.08);border:1px solid rgba(245,158,11,0.3);
                border-radius:16px;padding:2rem;text-align:center">
        <div style="font-size:2rem;margin-bottom:0.5rem">📂</div>
        <div style="font-size:1rem;font-weight:600;color:#f1f5f9;margin-bottom:0.3rem">No Dataset Loaded</div>
        <div style="font-size:0.85rem;color:#94a3b8">Please upload a CSV file first.</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("⬅️  Go to Upload"):
        st.session_state.current_page = "upload"
        st.rerun()

elif page == "upload":
    page_upload()

elif page == "overview":
    page_overview()

elif page == "viz":
    page_visualizations()

elif page == "ai":
    page_ai_insights()

elif page == "chat":
    page_chat()
