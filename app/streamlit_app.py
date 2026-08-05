"""
app/streamlit_app.py — v4: Unified Job Market Intelligence
4 tabs: Market Explorer (v2/v3 toggle) | Salary Predictor | AI Coach | 🌍 Live Job Finder

Run: streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from src.utils import DATA_PROCESSED, get_deepseek_config, get_tavily_config
from src.rag_engine import create_rag_chain, load_vector_store, ask
from src.tavily_job_finder import load_live_jobs, collect_live_jobs, summary_by_region

# ── Page ──
st.set_page_config(page_title="Job Market Intelligence v4", page_icon="🌍", layout="wide")

# ── Load data ──
@st.cache_data
def load_data():
    """Load both datasets."""
    all_path = DATA_PROCESSED / "all_jobs.csv"
    dsml_path = DATA_PROCESSED / "ds_ml_dl_jobs.csv"
    df_all = pd.read_csv(all_path) if all_path.exists() else None
    df_dsml = pd.read_csv(dsml_path) if dsml_path.exists() else None
    return df_all, df_dsml

@st.cache_resource
def get_rag():
    try:
        vs = load_vector_store("job_market_v4")
        return create_rag_chain(vs)
    except Exception:
        return None

df_all, df_dsml = load_data()

# ── Tabs ──
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Market Explorer", "💰 Salary Predictor",
    "🤖 AI Career Coach", "🌍 Live Job Finder",
])

# ══════════════════════════════════════════════
# TAB 1: Market Explorer
# ══════════════════════════════════════════════
with tab1:
    st.header("📊 Job Market Explorer")

    # ── Data scope toggle (inside this tab only) ──
    col_scope, col_exp, col_jobs = st.columns([3, 1, 1])
    with col_scope:
        version = st.radio(
            "📊 Data Scope",
            ["v2: All Data Jobs", "v3: DS / ML / DL Only"],
            index=1,
            horizontal=True,
        )
    use_dsml = "v3" in version
    df = df_dsml if use_dsml else df_all

    if df is not None:
        with col_exp:
            if "experience_level" in df.columns:
                exps = ["All"] + sorted(df["experience_level"].dropna().unique().tolist())
                sel_exp = st.selectbox("🎯 Experience", exps)
                filtered = df if sel_exp == "All" else df[df["experience_level"] == sel_exp]
            else:
                filtered = df
        with col_jobs:
            st.metric("Jobs", len(filtered))
    else:
        filtered = None
        st.warning("Run data pipeline first: `python build_data.py`")

    scope_label = "DS / ML / DL" if use_dsml else "All Data Jobs"

    if filtered is not None and not filtered.empty:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Median Salary", f"${filtered['salary_in_usd'].median():,.0f}")
        c2.metric("Mean Salary", f"${filtered['salary_in_usd'].mean():,.0f}")
        c3.metric("Unique Roles", filtered["job_title"].nunique())
        c4.metric("Countries", filtered["company_location"].nunique())

        st.markdown("---")

        col_l, col_r = st.columns(2)
        with col_l:
            st.subheader("Salary by Experience")
            exp_data = filtered.groupby("experience_level")["salary_in_usd"].median().sort_values()
            fig = px.bar(x=exp_data.values, y=exp_data.index, orientation="h",
                         color=exp_data.values, color_continuous_scale="blues",
                         labels={"x": "Median Salary (USD)", "y": "Experience"})
            st.plotly_chart(fig, use_container_width=True)
        with col_r:
            st.subheader("Salary by Company Size")
            sz = filtered.groupby("company_size")["salary_in_usd"].median().sort_values()
            fig = px.bar(x=sz.index, y=sz.values, color=sz.values,
                         color_continuous_scale="greens",
                         labels={"x": "Company Size", "y": "Median Salary (USD)"})
            st.plotly_chart(fig, use_container_width=True)

        # DS/ML field breakdown (only in v3 mode)
        if use_dsml and "ds_ml_field" in filtered.columns:
            st.subheader("Salary by DS / ML Field")
            field_data = filtered.groupby("ds_ml_field")["salary_in_usd"].median().sort_values()
            fig = px.bar(x=field_data.values, y=field_data.index, orientation="h",
                         color=field_data.values, color_continuous_scale="viridis",
                         labels={"x": "Median Salary (USD)", "y": "Field"})
            st.plotly_chart(fig, use_container_width=True)

        # Top roles
        st.subheader("Top 15 Job Titles")
        top = filtered["job_title"].value_counts().head(15)
        fig = px.bar(x=top.values, y=top.index, orientation="h",
                     color=top.values, color_continuous_scale="purples",
                     labels={"x": "Count", "y": "Job Title"})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data loaded. Run the data pipeline first.")

# ══════════════════════════════════════════════
# TAB 2: Salary Predictor
# ══════════════════════════════════════════════
with tab2:
    st.header("💰 Salary Predictor")
    st.caption("Using the DS / ML / DL dataset (switch to **Market Explorer** to explore all jobs)")
    # Use DS/ML/DL dataset by default (richer data)
    pred_df = df_dsml if df_dsml is not None else df_all
    if pred_df is not None and not pred_df.empty:
        c1, c2, c3, _ = st.columns([19, 12, 12, 53])
        role = c1.selectbox("Job Role", sorted(pred_df["job_title"].dropna().unique()))
        exp = c2.selectbox("Experience", sorted(pred_df["experience_level"].dropna().unique()))
        loc = c3.selectbox("Location", sorted(pred_df["company_location"].dropna().unique()))

        if st.button("Predict Salary", type="primary"):
            mask = (
                (pred_df["job_title"] == role) &
                (pred_df["experience_level"] == exp) &
                (pred_df["company_location"] == loc)
            )
            subset = pred_df[mask]
            if not subset.empty:
                st.metric("Estimated Salary (USD)", f"${subset['salary_in_usd'].median():,.0f}")
            else:
                st.info("Not enough data for exact match. Overall median:")
                st.metric("Overall Median", f"${pred_df['salary_in_usd'].median():,.0f}")

# ══════════════════════════════════════════════
# TAB 3: AI Career Coach (RAG)
# ══════════════════════════════════════════════
with tab3:
    st.header("🤖 AI Career Coach")
    cfg = get_deepseek_config()
    if not cfg["api_key"]:
        st.warning("Set DEEPSEEK_API_KEY in .env")
    else:
        st.success(f"DeepSeek ready ({cfg['model']})")

    if "msgs" not in st.session_state:
        st.session_state.msgs = []
    for m in st.session_state.msgs:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if prompt := st.chat_input("Ask about salaries, skills, career paths..."):
        st.session_state.msgs.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        qa = get_rag()
        if qa:
            with st.spinner("Thinking..."):
                result = ask(qa, prompt)
                ans = result["answer"]
        else:
            ans = "Run the data pipeline + RAG build first."
        st.session_state.msgs.append({"role": "assistant", "content": ans})
        with st.chat_message("assistant"):
            st.markdown(ans)

# ══════════════════════════════════════════════
# TAB 4: 🌍 Live Job Finder (Tavily)
# ══════════════════════════════════════════════
with tab4:
    st.header("🌍 Live DS/ML/DL Job Finder")
    st.markdown("Search the web for **real-time job postings** across 6 regions using Tavily.")

    tavily_cfg = get_tavily_config()
    has_tavily = bool(tavily_cfg["api_key"])

    if not has_tavily:
        st.warning("⚠️ Add `TAVILY_API_KEY` to your `.env` file to enable live job search.")

    # Show existing jobs
    live_jobs = load_live_jobs()

    col_a, col_b, col_c = st.columns([1, 1, 2])
    with col_a:
        if has_tavily:
            if st.button("🔍 Collect New Jobs", type="primary"):
                with st.spinner("Searching 6 regions for DS/ML/DL jobs... This may take 30-60 seconds."):
                    live_jobs = collect_live_jobs()
                    st.cache_data.clear()
                    st.rerun()

    with col_b:
        if not live_jobs.empty:
            st.metric("Total Jobs Found", len(live_jobs))
    with col_c:
        if not live_jobs.empty:
            st.caption("Jobs from Europe, Middle East, China, Russia, South America & East Asia")

    if live_jobs.empty:
        st.info("No jobs collected yet. Click 'Collect New Jobs Now' to search the web.")
        st.markdown("**Regions searched:** Europe · Middle East · China · Russia · South America · East Asia")
    else:
        # Summary by region
        st.subheader("Jobs by Region")
        region_summary = summary_by_region(live_jobs)
        if not region_summary.empty:
            fig = px.bar(region_summary, x="region", y="job_count",
                         color="region", labels={"job_count": "Jobs Found"},
                         color_discrete_sequence=px.colors.qualitative.Bold)
            st.plotly_chart(fig, use_container_width=True)

        # Filter by region
        st.subheader("Browse Jobs")
        regions = ["All"] + sorted(live_jobs["region"].dropna().unique().tolist())
        sel_region = st.selectbox("Filter by Region", regions)
        display = live_jobs if sel_region == "All" else live_jobs[live_jobs["region"] == sel_region]

        for _, row in display.iterrows():
            with st.expander(f"🌍 [{row['region']}] {row['title'][:100]}"):
                st.markdown(f"**Region:** {row['region']}")
                st.markdown(f"**Snippet:** {row['snippet'][:500]}")
                st.markdown(f"🔗 [Open Link]({row['url']})")
                st.caption(f"Collected: {row.get('collected_at', 'N/A')[:19]}")

        # Export
        csv = live_jobs.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download All Jobs (CSV)", csv, "live_ds_ml_dl_jobs.csv", "text/csv")

st.markdown("---")
st.caption("Job Market Intelligence v4 | Ironhack Final Project | DeepSeek + Tavily + Kaggle")
