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
from src.tavily_job_finder import collect_live_jobs
from src.applications import (load_jobs, save_jobs, mark_applied, list_pdfs,
                              send_application, get_smtp_config, extract_email)

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
            st.plotly_chart(fig, width="stretch")
        with col_r:
            st.subheader("Salary by Company Size")
            sz = filtered.groupby("company_size")["salary_in_usd"].median().sort_values()
            fig = px.bar(x=sz.index, y=sz.values, color=sz.values,
                         color_continuous_scale="greens",
                         labels={"x": "Company Size", "y": "Median Salary (USD)"})
            st.plotly_chart(fig, width="stretch")

        # DS/ML field breakdown (only in v3 mode)
        if use_dsml and "ds_ml_field" in filtered.columns:
            st.subheader("Salary by DS / ML Field")
            field_data = filtered.groupby("ds_ml_field")["salary_in_usd"].median().sort_values()
            fig = px.bar(x=field_data.values, y=field_data.index, orientation="h",
                         color=field_data.values, color_continuous_scale="viridis",
                         labels={"x": "Median Salary (USD)", "y": "Field"})
            st.plotly_chart(fig, width="stretch")

        # Top roles
        st.subheader("Top 15 Job Titles")
        top = filtered["job_title"].value_counts().head(15)
        fig = px.bar(x=top.values, y=top.index, orientation="h",
                     color=top.values, color_continuous_scale="purples",
                     labels={"x": "Count", "y": "Job Title"})
        st.plotly_chart(fig, width="stretch")
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
        loc_options = sorted(set(pred_df["company_location"].dropna().unique()) | {"Tunisia"})
        loc = c3.selectbox("Location", loc_options)

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

    # Narrow the chat input + send button to 50% of the page width
    st.markdown(
        """
        <style>
        [data-testid="stChatInput"] { max-width: 50%; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    cfg = get_deepseek_config()
    if not cfg["api_key"]:
        st.warning("Set DEEPSEEK_API_KEY in .env")
    else:
        # Green "ready" box, 50% width — matches the chat input
        st.markdown(
            f"""
            <div style="max-width:50%; padding:0.6rem 1rem; border-radius:0.5rem;
                        border:1px solid #2e7d32; background:#e8f5e9; color:#1b5e20;
                        font-size:0.9rem;">
              ✅ <b>AI Coach ready</b> · {cfg['model']} @ {cfg['base_url']}
            </div>
            """,
            unsafe_allow_html=True,
        )

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
        backend = ""
        if qa:
            with st.spinner("Thinking..."):
                try:
                    result = ask(qa, prompt)
                    ans = result["answer"]
                    backend = result.get("endpoint", "")
                except Exception as exc:  # noqa: BLE001 — never crash on a dead backend
                    ans = (
                        f"⚠️ **Couldn't reach the AI backend.**\n\n"
                        f"`{type(exc).__name__}: {exc}`\n\n"
                        "Start the local LLM (`~/Documents/llamacpp/serve.sh qwen3-next`) "
                        "or check your `.env` API keys."
                    )
        else:
            ans = "Run the data pipeline + RAG build first."
        st.session_state.msgs.append({"role": "assistant", "content": ans})
        with st.chat_message("assistant"):
            st.markdown(ans)
            if backend:
                st.caption(f"via {backend}")

# ══════════════════════════════════════════════
# TAB 4: 🌍 Live Job Finder (Tavily)
# ══════════════════════════════════════════════
with tab4:
    st.header("🌍 Live Job Finder")
    st.markdown("Search the web for **real-time job postings** across 7 regions — DS/ML/DL & Aerospace engineering.")
    st.caption("Two views: **🔍 Search** collects & explores jobs · **✉️ Apply** sends applications. "
               "Jobs are stored locally and never applied to twice.")

    tavily_cfg = get_tavily_config()
    has_tavily = bool(tavily_cfg["api_key"])

    if not has_tavily:
        st.warning("⚠️ Add `TAVILY_API_KEY` to your `.env` file to enable live job search.")

    # ── Shared: classification + local DB ──
    ds_keywords = ["data scientist", "machine learning", "deep learning", "AI engineer",
                   "ml engineer", "dl engineer", "nlp", "computer vision", "data science",
                   "mlops", "data engineer", "research scientist"]
    aero_keywords = ["aircraft", "aerospace", "airplane", "airframe", "fuselage",
                     "wing", "composites", "structural design", "maintenance engineer",
                     "mechanic", "repair", "overhaul", "aviation"]

    def classify_field(title, snippet):
        text = f"{title} {snippet}".lower()
        if any(k in text for k in aero_keywords):
            return "🛩️ Aerospace"
        if any(k in text for k in ds_keywords):
            return "💻 DS / ML / DL"
        return "📋 Other"

    # Jobs are persisted in a local SQLite DB (data/job_applications.db)
    live_jobs = load_jobs()
    if not live_jobs.empty:
        live_jobs["field"] = live_jobs.apply(
            lambda r: classify_field(r["title"], r["snippet"]), axis=1
        )
        # Backfill contact emails (aggressive extraction) where none were found
        empty = live_jobs["email"].fillna("").astype(str).str.strip() == ""
        if empty.any():
            live_jobs.loc[empty, "email"] = live_jobs[empty].apply(
                lambda r: extract_email(f"{r['title']} {r['snippet']}"), axis=1
            )

    applied_count = int(live_jobs["applied"].sum()) if not live_jobs.empty else 0

    search_tab, apply_tab = st.tabs(["🔍 Search Jobs", "✉️ Apply to Jobs"])

    # ═══════════════ SEARCH ═══════════════
    with search_tab:

        # ── Top bar: collect + metrics ──
        col_a, col_b, col_c = st.columns([1, 1, 3])
        with col_a:
            if has_tavily:
                if st.button("🔍 Collect New Jobs", type="primary"):
                    with st.spinner("Searching 7 regions for DS/ML/DL & Aerospace jobs... This may take 30-60 seconds."):
                        collected = collect_live_jobs()
                        if not collected.empty:
                            collected["field"] = collected.apply(
                                lambda r: classify_field(r["title"], r["snippet"]), axis=1
                            )
                            save_jobs(collected)
                        st.cache_data.clear()
                        st.rerun()
        with col_b:
            st.metric("Jobs in DB", len(live_jobs))
        with col_c:
            st.metric("✅ Applied", applied_count)

        if live_jobs.empty:
            st.info("No jobs collected yet. Click 'Collect New Jobs' to search the web.")
            st.markdown("**Regions searched:** Europe · Middle East · China · Russia · South America · East Asia · Aerospace")
        else:
            # ── Filters ──
            st.markdown("---")
            filt_col1, filt_col2, filt_col3 = st.columns([1, 1, 5])
            with filt_col1:
                field_options = ["🌐 All Fields", "💻 DS / ML / DL", "🛩️ Aerospace"]
                sel_field = st.selectbox("🎯 Filter by Field", field_options)
            with filt_col2:
                geo_regions = sorted([r for r in live_jobs["region"].dropna().unique()
                                      if r != "Aerospace"])
                regions = ["🌍 All Regions"] + geo_regions
                sel_region = st.selectbox("📍 Filter by Region", regions)

            # Apply filters
            display = live_jobs.copy()
            if sel_field != "🌐 All Fields":
                display = display[display["field"] == sel_field]
            if sel_region != "🌍 All Regions":
                display = display[display["region"] == sel_region]

            with filt_col3:
                st.caption(f"Showing **{len(display)}** of {len(live_jobs)} jobs")

            # Status filter — default hides jobs you already applied to
            status_filter = st.radio(
                "Status:", ["Not applied", "All", "Applied"], horizontal=True, index=0
            )
            if status_filter == "Not applied":
                display = display[display["applied"] == 0]
            elif status_filter == "Applied":
                display = display[display["applied"] == 1]

            # ── Graph: Jobs by Region (geographic regions only) ──
            # Aerospace is a FIELD, not a region — jobs collected under the
            # "Aerospace" bucket are reported as a count, never as a region.
            # Total = In geographic regions + Other regions, always.
            aero_count = int((display["region"] == "Aerospace").sum())
            geo = display[display["region"] != "Aerospace"]

            # Compact inline summary: Total = In geographic regions + Other regions
            st.markdown(
                f"""
                <div style="display:flex; flex-wrap:wrap; gap:24px; align-items:baseline;
                            font-size:0.85rem; color:#808495; margin:8px 0 2px 0;">
                  <span>Total jobs <b style="font-size:1.15rem; color:#262730;">{len(display)}</b></span>
                  <span style="border-left:1px solid #e0e0e0; padding-left:24px;">
                    In geographic regions <b style="font-size:1.15rem; color:#262730;">{len(geo)}</b></span>
                  <span style="border-left:1px solid #e0e0e0; padding-left:24px;">
                    🌍 Other regions <b style="font-size:1.15rem; color:#262730;">{aero_count}</b></span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.subheader("Jobs by Region")
            graph_col, _ = st.columns([3, 2])
            with graph_col:
                # Include non-geographic jobs as an "Other regions" bar (always last)
                plot_df = display.copy()
                if aero_count:
                    plot_df["region"] = plot_df["region"].replace("Aerospace", "Other regions")
                if not plot_df.empty:
                    region_order = (
                        sorted(r for r in plot_df["region"].unique() if r != "Other regions")
                        + (["Other regions"] if aero_count else [])
                    )
                    if sel_field == "🌐 All Fields":
                        grouped = plot_df.groupby(["region", "field"]).size().reset_index(name="job_count")
                        max_count = int(grouped.groupby("region")["job_count"].sum().max()) if not grouped.empty else 1
                        fig = px.bar(grouped, x="region", y="job_count", color="field",
                                     category_orders={"region": region_order},
                                     labels={"job_count": "Jobs Found", "field": "Field"},
                                     color_discrete_map={
                                         "💻 DS / ML / DL": "#636EFA",
                                         "🛩️ Aerospace": "#EF553B",
                                         "📋 Other": "#B6B6B6",
                                     },
                                     barmode="stack")
                    else:
                        grouped = plot_df.groupby("region").size().reset_index(name="job_count")
                        max_count = int(grouped["job_count"].max())
                        fig = px.bar(grouped, x="region", y="job_count",
                                     category_orders={"region": region_order},
                                     color="region", labels={"job_count": "Jobs Found"},
                                     color_discrete_sequence=px.colors.qualitative.Bold)
                    fig.update_yaxes(dtick=1, range=[0, max_count + 1])
                    fig.update_layout(height=300, margin=dict(t=10, b=10))
                    st.plotly_chart(fig, width="content")
                else:
                    st.info("No jobs match this filter.")

            # ── Browse ──
            st.markdown("---")
            st.subheader("Browse Jobs")
            for _, row in display.iterrows():
                region_label = f" [{row['region']}]" if row["region"] != "Aerospace" else ""
                applied_badge = "✅ " if row.get("applied") else ""
                with st.expander(f"{applied_badge}{row['field']}{region_label} {row['title'][:100]}"):
                    st.markdown(f"**Field:** {row['field']}")
                    if row["region"] != "Aerospace":
                        st.markdown(f"**Region:** {row['region']}")
                    st.markdown(f"**Snippet:** {row['snippet'][:500]}")
                    st.markdown(f"🔗 [Open Link]({row['url']})")
                    if row.get("applied"):
                        st.success(f"✅ Applied on {str(row.get('applied_at', ''))[:19]}")
                    st.caption(f"Collected: {row.get('collected_at', 'N/A')[:19]}")

            # ── Export ──
            st.markdown("---")
            csv = live_jobs.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download All Jobs (CSV)", csv, "live_jobs.csv", "text/csv")

    # ═══════════════ APPLY ═══════════════
    with apply_tab:
        st.caption("Tick the jobs you want to apply to, write your email, point to your CV folder, then send. "
                   "Applied jobs are saved locally and **never applied to twice**.")

        apply_status = st.radio(
            "Show jobs:", ["Not applied", "All", "Applied"], horizontal=True, index=0
        )
        candidates = live_jobs.copy()
        if apply_status == "Not applied":
            candidates = candidates[candidates["applied"] == 0]
        elif apply_status == "Applied":
            candidates = candidates[candidates["applied"] == 1]

        if candidates.empty:
            st.info("No jobs here yet. Collect jobs in the **🔍 Search** tab first (or change the filter).")
        else:
            editable = candidates[["job_id", "field", "region", "title", "email", "url", "applied"]].copy()
            editable["Apply"] = False
            editable["email"] = editable["email"].fillna("").astype(str)
            editable["applied"] = editable["applied"].map({1: "✅ Applied", 0: "—"})

            edited = st.data_editor(
                editable,
                column_config={
                    "Apply": st.column_config.CheckboxColumn("Apply", default=False),
                    "field": st.column_config.TextColumn("Field", disabled=True),
                    "region": st.column_config.TextColumn("Region", disabled=True),
                    "title": st.column_config.TextColumn("Job title", width="large", disabled=True),
                    "email": st.column_config.TextColumn("Contact email (edit to override)"),
                    "url": st.column_config.LinkColumn("Link", disabled=True),
                    "applied": st.column_config.TextColumn("Status", disabled=True),
                },
                disabled=["job_id", "field", "region", "title", "url", "applied"],
                hide_index=True,
                num_rows="fixed",
                height=380,
            )

            selected = edited[edited["Apply"] & (edited["applied"] != "✅ Applied")]
            n_selected = len(selected)

            with st.expander("📧 Application email & attachments", expanded=True):
                subject = st.text_input("Email subject", value="Application for {title} position")
                body = st.text_area(
                    "Email body",
                    height=240,
                    value=(
                        "Dear Hiring Team,\n\n"
                        "I am writing to apply for the {title} position. "
                        "Please find my CV attached.\n\n"
                        "Thank you for your consideration.\n\nBest regards"
                    ),
                )
                pdf_dir = st.text_input(
                    "📁 PDF folder (CV / resume)", placeholder="/home/a/Documents/CV"
                )
                pdfs = list_pdfs(pdf_dir) if pdf_dir else []
                if pdf_dir and not pdfs:
                    st.warning("No PDF files found in that folder.")
                elif pdfs:
                    st.caption("Will attach: " + ", ".join(Path(p).name for p in pdfs))

            if st.button(
                f"📤 Send applications to {n_selected} selected job(s)",
                type="primary",
                disabled=n_selected == 0,
            ):
                smtp = get_smtp_config()
                if not (smtp["host"] and smtp["from_email"]):
                    st.error("SMTP is not configured. Add SMTP_HOST / SMTP_PORT / SMTP_USER / SMTP_PASS / EMAIL_FROM to .env")
                else:
                    results, ok_urls = [], []
                    for _, row in selected.iterrows():
                        recipient = (row["email"] or "").strip()
                        subj = subject.replace("{title}", str(row["title"]))
                        text = body.replace("{title}", str(row["title"]))
                        ok, msg = send_application(recipient, subj, text, pdfs)
                        results.append(f"{row['title'][:60]} → {msg}")
                        if ok:
                            ok_urls.append(row["url"])
                    for r in results:
                        st.write(r)
                    if ok_urls:
                        mark_applied(ok_urls)
                        st.success(f"Marked {len(ok_urls)} job(s) as applied — they won't be re-applied.")
                        st.rerun()

st.markdown("---")
st.caption("Job Market Intelligence v4 | Ironhack Final Project | DeepSeek + Tavily + HuggingFace")
