"""
PPP Candidate Briefing Agent
Streamlit application entry point.
Run with: streamlit run app.py
"""

from __future__ import annotations
import json
import logging
import os

import pandas as pd
import streamlit as st
import anthropic
from dotenv import load_dotenv
from tavily import TavilyClient

from agent.config import CLAUDE_MODEL
from agent.pipeline import run_pipeline, DEFAULT_ROLE_SPEC
from agent.utils import parse_csv, briefings_to_summary_df, validate_and_serialize
from agent.models import CandidateBriefing, OutputSchema

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="PPP · Candidate Briefing Agent",
    page_icon="PPP-Primary-Logo.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# PPP BRAND CSS
# ──────────────────────────────────────────────
st.markdown(
    """
<style>
/* ═══════════════════════════════════════════════
   PPP BRAND DESIGN SYSTEM  v2.0
   Polished · Animated · Responsive
════════════════════════════════════════════════ */

/* ── Colour & shadow tokens ── */
:root {
    --ppp-blue:      #6B9BB8;
    --ppp-blue-d:    #4d7a96;
    --ppp-blue-l:    #e8f1f7;
    --ppp-blue-xl:   #f0f6fa;
    --ppp-dark:      #2C3E50;
    --ppp-mid:       #546e7a;
    --ppp-light:     #78909c;
    --ppp-bg:        #F0F4F8;
    --ppp-border:    #D0DDE8;
    --ppp-white:     #FFFFFF;
    --ppp-ok:        #2E7D52;
    --ppp-err:       #b94040;

    /* Elevation shadows */
    --shadow-xs:  0 1px 3px rgba(44,62,80,0.08);
    --shadow-sm:  0 2px 8px rgba(107,155,184,0.12), 0 1px 3px rgba(44,62,80,0.08);
    --shadow-md:  0 4px 16px rgba(107,155,184,0.16), 0 2px 6px rgba(44,62,80,0.08);
    --shadow-lg:  0 8px 32px rgba(107,155,184,0.20), 0 4px 12px rgba(44,62,80,0.10);
    --shadow-btn: 0 2px 8px rgba(107,155,184,0.35);

    /* Transition presets */
    --ease-fast:  all 0.15s ease;
    --ease-std:   all 0.22s ease;
    --ease-slow:  all 0.35s ease;
}

/* ── Entry animation ── */
@keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes fadeIn {
    from { opacity: 0; }
    to   { opacity: 1; }
}

/* ── Page background — subtle mesh gradient ── */
.stApp {
    background: linear-gradient(135deg, #EDF2F7 0%, #F0F4F8 50%, #E8EFF6 100%) !important;
    background-attachment: fixed !important;
}
section.main > div { background: transparent !important; }

/* ── Main content container ── */
section.main > div.block-container {
    max-width: 1400px !important;
    padding-left: clamp(1rem, 3vw, 3rem) !important;
    padding-right: clamp(1rem, 3vw, 3rem) !important;
    padding-top: 2rem !important;
    animation: fadeIn 0.4s ease;
}

/* ── Sidebar ── */
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div {
    background: linear-gradient(180deg, #FFFFFF 0%, #F8FAFC 100%) !important;
    border-right: 1px solid var(--ppp-border) !important;
    box-shadow: 2px 0 12px rgba(107,155,184,0.08) !important;
}

/* ── Headings — fluid + polished ── */
h1 {
    color: var(--ppp-dark) !important;
    font-weight: 800 !important;
    font-size: clamp(1.55rem, 3vw, 2.1rem) !important;
    letter-spacing: -0.5px;
    line-height: 1.25;
    margin-bottom: 0.5rem !important;
}
h2 {
    color: var(--ppp-dark) !important;
    font-weight: 700 !important;
    font-size: clamp(1.15rem, 2.2vw, 1.4rem) !important;
    border-bottom: 2px solid var(--ppp-blue) !important;
    padding-bottom: 8px !important;
    margin-top: 1.8rem !important;
    margin-bottom: 1.2rem !important;
}
h3 { color: var(--ppp-blue-d) !important; font-weight: 600 !important; font-size: clamp(1rem, 1.8vw, 1.1rem) !important; }
h4 { color: var(--ppp-dark) !important; font-weight: 600 !important; font-size: 1rem !important; }

/* ── Body text — slightly larger ── */
p, li { color: var(--ppp-dark) !important; font-size: 0.97rem !important; line-height: 1.65; }
.stMarkdown p { color: var(--ppp-dark) !important; font-size: 0.97rem !important; }

/* ── Caption / subtext ── */
small,
[data-testid="stCaptionContainer"] p,
.stCaption p {
    color: var(--ppp-light) !important;
    font-size: 0.85rem !important;
}

/* ── File uploader — animated hover ── */
[data-testid="stFileUploadDropzone"] {
    background-color: var(--ppp-white) !important;
    border: 2px dashed var(--ppp-blue) !important;
    border-radius: 12px !important;
    transition: var(--ease-std);
    box-shadow: var(--shadow-xs);
}
[data-testid="stFileUploadDropzone"]:hover {
    background-color: var(--ppp-blue-xl) !important;
    border-color: var(--ppp-blue-d) !important;
    box-shadow: var(--shadow-sm);
    transform: translateY(-1px);
}
[data-testid="stFileUploadDropzone"] p,
[data-testid="stFileUploadDropzone"] small {
    color: var(--ppp-mid) !important;
    font-size: 0.92rem !important;
}

/* ── Text area ── */
textarea {
    background-color: var(--ppp-white) !important;
    color: var(--ppp-dark) !important;
    border-color: var(--ppp-border) !important;
    border-radius: 8px !important;
    font-size: 0.95rem !important;
    line-height: 1.6;
    box-shadow: var(--shadow-xs);
    transition: var(--ease-fast);
}
textarea:focus {
    border-color: var(--ppp-blue) !important;
    box-shadow: 0 0 0 3px rgba(107,155,184,0.18) !important;
}

/* ── Primary button — gradient + shadow + hover lift ── */
button[kind="primary"] {
    background: linear-gradient(135deg, var(--ppp-blue) 0%, var(--ppp-blue-d) 100%) !important;
    border: none !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    border-radius: 8px !important;
    padding: 0.6rem 1.6rem !important;
    letter-spacing: 0.3px;
    box-shadow: var(--shadow-btn);
    transition: var(--ease-fast);
}
button[kind="primary"]:hover {
    background: linear-gradient(135deg, var(--ppp-blue-d) 0%, #3d6880 100%) !important;
    box-shadow: 0 4px 16px rgba(107,155,184,0.45) !important;
    transform: translateY(-2px);
}
button[kind="primary"]:active { transform: translateY(0); }
button[kind="primary"]:disabled { opacity: 0.42 !important; transform: none; box-shadow: none !important; }

/* ── Secondary button ── */
button[kind="secondary"] {
    background-color: var(--ppp-white) !important;
    border: 1.5px solid var(--ppp-blue) !important;
    color: var(--ppp-blue-d) !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    box-shadow: var(--shadow-xs);
    transition: var(--ease-fast);
}
button[kind="secondary"]:hover {
    background-color: var(--ppp-blue-l) !important;
    box-shadow: var(--shadow-sm);
    transform: translateY(-1px);
}

/* ── Progress bar — rounded + animated ── */
[data-testid="stProgressBar"] > div {
    border-radius: 99px !important;
    overflow: hidden;
    background: var(--ppp-blue-l) !important;
}
[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, var(--ppp-blue) 0%, var(--ppp-blue-d) 100%) !important;
    border-radius: 99px !important;
    transition: width 0.5s ease !important;
}

/* ── Metric cards — elevated + hover ── */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #FFFFFF 0%, #F8FBFD 100%) !important;
    border: 1px solid var(--ppp-border) !important;
    border-radius: 12px !important;
    padding: 16px 20px !important;
    box-shadow: var(--shadow-sm);
    transition: var(--ease-std);
}
[data-testid="stMetric"]:hover {
    box-shadow: var(--shadow-md);
    transform: translateY(-2px);
    border-color: var(--ppp-blue) !important;
}
[data-testid="stMetricLabel"] > div { color: var(--ppp-mid) !important; font-size: 0.82rem !important; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
[data-testid="stMetricValue"] > div { color: var(--ppp-blue-d) !important; font-weight: 800 !important; font-size: 1.75rem !important; }

/* ── Expander — card style with shadow ── */
[data-testid="stExpander"] {
    background-color: var(--ppp-white) !important;
    border: 1px solid var(--ppp-border) !important;
    border-radius: 12px !important;
    margin-bottom: 12px !important;
    box-shadow: var(--shadow-sm);
    transition: var(--ease-std);
    animation: fadeSlideUp 0.3s ease;
    overflow: hidden;
}
[data-testid="stExpander"]:hover {
    box-shadow: var(--shadow-md);
    border-color: var(--ppp-blue) !important;
}
[data-testid="stExpander"] summary {
    padding: 14px 18px !important;
    border-radius: 12px !important;
    transition: var(--ease-fast);
}
[data-testid="stExpander"] summary:hover { background-color: var(--ppp-blue-xl) !important; }
[data-testid="stExpander"] summary p { color: var(--ppp-dark) !important; font-weight: 600 !important; font-size: 0.97rem !important; }

/* ── Alerts ── */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    border-width: 1px !important;
    box-shadow: var(--shadow-xs);
    animation: fadeSlideUp 0.25s ease;
}
[data-testid="stAlert"] p { color: inherit !important; font-size: 0.94rem !important; }

/* ── Info box ── */
[data-testid="stAlert"][data-baseweb="notification"] {
    border-left: 4px solid var(--ppp-blue) !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--ppp-border) !important;
    border-radius: 10px !important;
    overflow: hidden !important;
    box-shadow: var(--shadow-sm);
}

/* ── Code / tags ── */
code {
    background-color: var(--ppp-blue-l) !important;
    color: var(--ppp-blue-d) !important;
    border-radius: 5px !important;
    padding: 2px 7px !important;
    font-size: 0.85rem !important;
    font-weight: 600;
}

/* ── Horizontal rule ── */
hr { border: none !important; border-top: 1px solid var(--ppp-border) !important; margin: 2rem 0 !important; }

/* ── Step label badge — pill with gradient ── */
.step-label {
    display: inline-block;
    background: linear-gradient(135deg, var(--ppp-blue) 0%, var(--ppp-blue-d) 100%);
    color: #ffffff;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    padding: 4px 12px;
    border-radius: 99px;
    margin-bottom: 8px;
    box-shadow: 0 2px 6px rgba(107,155,184,0.3);
}

/* ── Section divider heading ── */
.section-heading {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 1rem;
}

/* ── Status / log panel ── */
[data-testid="stStatusWidget"] {
    border-radius: 10px !important;
    box-shadow: var(--shadow-sm);
    border: 1px solid var(--ppp-border) !important;
}

/* ─────────────────────────────────────
   RESPONSIVE BREAKPOINTS
───────────────────────────────────── */

/* ── Tablet (≤ 1024 px) ── */
@media (max-width: 1024px) {
    [data-testid="stMetricValue"] > div { font-size: 1.4rem !important; }
}

/* ── Mobile (≤ 768 px) ── */
@media (max-width: 768px) {
    [data-testid="stHorizontalBlock"] { flex-wrap: wrap !important; }
    [data-testid="stColumn"] { width: 100% !important; flex: 1 1 100% !important; min-width: 0 !important; }
    [data-testid="stMetric"] { padding: 12px 16px !important; }
    [data-testid="stMetricValue"] > div { font-size: 1.3rem !important; }
    [data-testid="stSidebar"] img { max-width: 130px !important; }
    .step-label { font-size: 0.67rem !important; }
    textarea { font-size: 0.9rem !important; }
    [data-testid="stDownloadButton"] button { width: 100% !important; }
}

/* ── Small mobile (≤ 480 px) ── */
@media (max-width: 480px) {
    section.main > div.block-container { padding-left: 0.75rem !important; padding-right: 0.75rem !important; }
    h1 { font-size: 1.3rem !important; }
    h2 { font-size: 1.1rem !important; }
}
</style>
""",
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────
# BRIEFING PREVIEW RENDERER
# (defined before the run block — Python executes top-to-bottom)
# ──────────────────────────────────────────────
def _md(text: str) -> str:
    """Escape dollar signs so Streamlit doesn't render them as LaTeX math."""
    return text.replace("$", r"\$")


def _render_briefing_preview(b: CandidateBriefing):
    """Render a single briefing — two-column on desktop, stacked on mobile."""
    col_left, col_right = st.columns([3, 2], gap="large")

    with col_left:
        st.markdown(f"**{b.current_role.title}** · {b.current_role.employer}")
        st.caption(f"Estimated tenure: {b.current_role.tenure_years} years")
        st.markdown("**Career Summary**")
        st.markdown(_md(b.career_narrative))
        st.markdown(f"**Firm Context** — {_md(b.firm_aum_context)}")
        st.markdown("**Experience Tags**")
        tag_html = " ".join([f"`{t}`" for t in b.experience_tags])
        st.markdown(tag_html)

    with col_right:
        # Metrics sit side-by-side; on mobile the parent column already stacks
        # so these two sub-columns stay side-by-side within the available width.
        mcol1, mcol2 = st.columns(2, gap="small")
        with mcol1:
            st.metric("Role Fit", f"{b.role_fit.score}/10")
        with mcol2:
            st.metric("Mobility", f"{b.mobility_signal.score}/5")

        st.markdown("**Role Fit**")
        st.markdown(_md(b.role_fit.justification))
        st.markdown("**Mobility Rationale**")
        st.markdown(_md(b.mobility_signal.rationale))
        st.markdown("**Outreach Hook**")
        st.info(f'"{_md(b.outreach_hook)}"')


# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
tavily_key = os.getenv("TAVILY_API_KEY", "")

with st.sidebar:
    st.image("PPP-Primary-Logo.png", width=170)
    st.markdown("---")
    # st.markdown(f"**Model:** `{CLAUDE_MODEL}`")
    st.markdown("**Version:** 1.2.0")
    st.caption("Platinum Pacific Partners · Internal Tool")


# ──────────────────────────────────────────────
# PAGE HEADER
# ──────────────────────────────────────────────
st.markdown("# Candidate Briefing Agent")
st.markdown(
    "Upload a candidate list, configure the role, and generate AI-powered briefings "
    "with career narrative, mobility signal, role fit score, and outreach hook."
)
st.markdown("---")


# ──────────────────────────────────────────────
# STEP 1 — CSV UPLOAD
# ──────────────────────────────────────────────
st.markdown('<span class="step-label">Step 1</span>', unsafe_allow_html=True)
st.markdown("## Upload Candidate List")

col_upload, col_format = st.columns([3, 1], gap="medium")

with col_upload:
    uploaded_file = st.file_uploader(
        "Upload candidates.csv",
        type=["csv"],
        label_visibility="collapsed",
    )

with col_format:
    with st.expander("Required columns"):
        st.markdown(
            "```\nfull_name\ncurrent_employer\ncurrent_title\nlinkedin_url\n```"
        )

candidates = []
if uploaded_file is not None:
    candidates, parse_error = parse_csv(uploaded_file)
    uploaded_file.seek(0)

    if parse_error:
        st.error(f"CSV Error: {parse_error}")
    elif candidates:
        st.success(f"{len(candidates)} candidate(s) loaded")
        preview_df = pd.DataFrame(
            [
                {
                    "#": c.candidate_id,
                    "Name": c.full_name,
                    "Title": c.current_title,
                    "Employer": c.current_employer,
                }
                for c in candidates
            ]
        )
        st.dataframe(preview_df, width="stretch", hide_index=True)

st.markdown("---")


# ──────────────────────────────────────────────
# STEP 2 — ROLE SPECIFICATION
# ──────────────────────────────────────────────
st.markdown('<span class="step-label">Step 2</span>', unsafe_allow_html=True)
st.markdown("## Role Specification")
st.caption(
    "Edit for each mandate — the agent scores every candidate against this spec."
)

col_spec, col_reset = st.columns([6, 1], gap="small")
with col_reset:
    st.markdown("<div style='padding-top:8px'>", unsafe_allow_html=True)
    if st.button("Reset", type="secondary"):
        st.session_state["role_spec"] = DEFAULT_ROLE_SPEC
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

role_spec = st.text_area(
    "Role Specification",
    value=st.session_state.get("role_spec", DEFAULT_ROLE_SPEC),
    height=260,
    label_visibility="collapsed",
    key="role_spec_input",
)
st.session_state["role_spec"] = role_spec

st.markdown("---")


# ──────────────────────────────────────────────
# STEP 3 — GENERATE
# ──────────────────────────────────────────────
st.markdown('<span class="step-label">Step 3</span>', unsafe_allow_html=True)
st.markdown("## Generate Briefings")

blocking_reasons = []
if not candidates:
    blocking_reasons.append("No candidates loaded — upload a CSV above")

for reason in blocking_reasons:
    st.warning(reason)

# ── Button state machine ──────────────────────────────────────
# "idle"    → button enabled; click sets state to "running" + reruns
# "running" → button disabled; pipeline executes; state resets to "idle"
# This two-phase approach lets Streamlit re-render the disabled button
# before the blocking pipeline call begins.
if "pipeline_state" not in st.session_state:
    st.session_state["pipeline_state"] = "idle"

is_pipeline_running = st.session_state["pipeline_state"] == "running"
can_run = len(blocking_reasons) == 0 and not is_pipeline_running

btn_label = (
    "Running..." if is_pipeline_running
    else f"Generate Briefings for {len(candidates)} Candidate(s)" if candidates
    else "Generate Briefings"
)

st.markdown("<div style='margin-top: 12px;'>", unsafe_allow_html=True)
if st.button(btn_label, type="primary", disabled=not can_run):
    st.session_state["pipeline_state"] = "running"
    st.rerun()  # re-render with disabled button before pipeline starts
st.markdown("</div>", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# PIPELINE EXECUTION
# ──────────────────────────────────────────────
if st.session_state.get("pipeline_state") == "running" and not blocking_reasons:
    try:
        anthropic_client = anthropic.Anthropic(api_key=anthropic_key)
        tavily_client = TavilyClient(api_key=tavily_key)
    except Exception as e:
        st.session_state["pipeline_state"] = "idle"
        st.error(f"Failed to initialise API clients: {e}")
        st.stop()

    st.markdown("---")

    total_candidates = len(candidates)
    completed: list[CandidateBriefing] = []
    progress_bar = st.progress(0.0, text="Starting...")
    results_container = st.container()

    # st.status gives a live-updating log panel that flushes each write()
    # immediately to the browser — no custom polling needed.
    pipeline_error: str | None = None

    with st.status("Processing candidates...", expanded=False) as run_status:

        def on_status(msg: str) -> None:
            try:
                run_status.write(msg)
            except Exception:
                pass  # never let a UI write error interrupt the research loop

        def on_preview(briefing: CandidateBriefing) -> None:
            completed.append(briefing)
            frac = len(completed) / total_candidates
            progress_bar.progress(
                frac, text=f"{len(completed)}/{total_candidates} complete"
            )

            is_error = "[error]" in briefing.experience_tags
            fit_score = briefing.role_fit.score
            mob_score = briefing.mobility_signal.score
            label = (
                f"{'Error' if is_error else 'Done'}  ·  "
                f"{briefing.full_name}  ·  "
                f"Role Fit {fit_score}/10  ·  "
                f"Mobility {mob_score}/5  ·  "
                f"{briefing.current_role.employer}"
            )
            with results_container:
                with st.expander(label, expanded=False):
                    if is_error:
                        st.error(
                            "This candidate could not be processed — manual review required."
                        )
                    else:
                        _render_briefing_preview(briefing)

        try:
            output = run_pipeline(
                candidates=candidates,
                role_spec=role_spec,
                anthropic_client=anthropic_client,
                tavily_client=tavily_client,
                status_callback=on_status,
                preview_callback=on_preview,
            )
        except Exception as e:
            pipeline_error = str(e)
            logger.error(f"Unexpected pipeline error: {e}", exc_info=True)
            # Preserve any candidates that completed before the crash
            output = OutputSchema(candidates=completed) if completed else None

        st.session_state["pipeline_state"] = "idle"
        if pipeline_error:
            run_status.update(
                label="Pipeline stopped early — see error below", state="error"
            )
        else:
            run_status.update(
                label="All candidates complete", state="complete", expanded=False
            )

    progress_bar.progress(
        1.0, text="Complete" if not pipeline_error else "Stopped early"
    )

    if pipeline_error:
        st.error(
            f"An unexpected error stopped the pipeline: {pipeline_error}. "
            "Any candidates that completed before the error are shown below."
        )
        if not output:
            st.stop()

    # ── Results ──
    st.markdown("---")
    st.markdown("## Results")

    json_str, is_valid, validation_msg = validate_and_serialize(output)
    if is_valid:
        st.success(validation_msg)
    else:
        st.warning(f"Output validation warning: {validation_msg}")

    st.markdown("### Summary")
    summary_df = briefings_to_summary_df(output.candidates)
    st.dataframe(
        summary_df.style.map(
            lambda v: "color: #b94040; font-weight:600" if v == "⚠️ Error" else "",
            subset=["Status"],
        ),
        width="stretch",
        hide_index=True,
    )

    output_path = "output.json"
    disk_saved = False
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(json_str)
        disk_saved = True
    except OSError as e:
        logger.warning(f"Failed to write {output_path}: {e}")
        disk_error = str(e)

    st.markdown("### Download")
    col_dl, col_info = st.columns([1, 2], gap="medium")
    with col_dl:
        st.download_button(
            label="Download output.json",
            data=json_str,
            file_name="output.json",
            mime="application/json",
            type="primary",
        )
    with col_info:
        if disk_saved:
            st.info(f"Also saved to `{output_path}` in the project directory.")
        else:
            st.warning(f"Could not save to disk: {disk_error}. Use the download button above.")
