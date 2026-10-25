# -*- coding: utf-8 -*-
#
# Streamlit frontend for cv_project.py, as one page instead of a chat:
#   search form -> pick a job in the table -> the model extracts the
#   requirements and checks your bank -> fill your answers in the editable
#   table -> Build CV -> download
#
#   streamlit run streamlit_app.py
#

import io
import json

from pathlib import Path

import pandas as pd
import streamlit as st
from xhtml2pdf import pisa

import cv_project as cv
from cv_to_pdf import markdown_to_html
from job_sources import SOURCES, fetch_jobs

st.set_page_config(page_title="CV tailoring", page_icon=":material/description:")


@st.cache_resource(show_spinner=False)
def load_model() -> str:
    cv.load_model()
    return cv.MODEL_ID


@st.cache_data(show_spinner=False)
def cached_job_count(source_names: tuple[str, ...], modified: tuple[float, ...]) -> int:
    # modified (the cache files' mtimes) only makes the count refresh after a download
    return sum(len(json.loads(path.read_text(encoding="utf-8")))
               for path in map(cache_path, source_names) if path.exists())


@st.cache_data(max_entries=50, show_spinner=False)
def extract_requirements(job_text: str) -> list[str]:
    return cv.extract_requirements(job_text)


@st.cache_data(max_entries=20, show_spinner=False)
def cv_pdf_bytes(cv_markdown: str) -> bytes:
    buffer = io.BytesIO()
    result = pisa.CreatePDF(markdown_to_html(cv_markdown), dest=buffer, encoding="utf-8")
    if result.err:
        raise RuntimeError(f"xhtml2pdf reported {result.err} errors while writing the PDF")
    return buffer.getvalue()


st.session_state.setdefault("results", [])      # all jobs matching the last search
st.session_state.setdefault("search_id", 0)     # new key for the jobs table on every search
st.session_state.setdefault("analysis", None)   # {"job", "requirements", "gaps", "round", "cv"}


def analyze(job: dict) -> None:
    with st.status(":shimmer[Analyzing the job]", type="compact") as status:
        with st.status("Loading the model", type="step"):
            load_model()
        with st.status("Extracting requirements", type="step"):
            requirements = extract_requirements(job["description"])
        gaps = []
        if requirements:
            with st.status(f"Checking your bank against {len(requirements)} requirements", type="step"):
                gaps = cv.analyze_gaps(cv.load_bank(cv.BANK_PATH), requirements)
        status.update(label="Analysis complete", state="complete")
    cv.write_log("requirements", "\n".join(requirements))
    st.session_state.analysis = {"job": job, "requirements": requirements, "gaps": gaps, "round": 0, "cv": None}


def cache_path(source_name: str) -> Path:
    return Path(f"{source_name}_cache.json")


def build_cv(analysis: dict, answers: list[str], placements: list[str]) -> None:
    #
    # answers and placements (role ids, "" = free section) line up with the open gaps shown in the editor. Each answered gap
    # is saved to the bank (and marked demonstrated by add_answer), so it leaves
    # the editor on the next run and can't be saved twice.
    #
    bank = cv.load_bank(cv.BANK_PATH)
    open_gaps = [g for g in analysis["gaps"] if g["status"] != "demonstrated"]
    saved = sum(cv.add_answer(gap, answer or "", bank, attach_to)
                for gap, answer, attach_to in zip(open_gaps, answers, placements))
    if saved:
        cv.save_bank(bank, cv.BANK_PATH)
    selected = cv.select_bullets(bank, analysis["gaps"], analysis["requirements"])
    cv_markdown = cv.build_cv(bank, selected, analysis["requirements"], analysis["job"])
    cv.write_log("gaps", json.dumps(analysis["gaps"], indent=2, ensure_ascii=False))
    cv.write_log("rows", json.dumps(cv.build_selection_rows(selected, False), indent=2, ensure_ascii=False))
    cv.write_log("cv", cv_markdown)
    analysis["cv"] = cv_markdown
    analysis["round"] += 1  # fresh editor, without the answers just saved


# ---------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("Job search")
    source_names = st.pills("Sources", list(SOURCES), default=cv.JOB_SOURCES, selection_mode="multi")
    data_source = st.segmented_control("Job data", ["Use cached", "Download"], default="Use cached",
                                       help="Cached data is what the last download saved to disk.")
    sources = tuple(source_names or ())
    cache_files = [cache_path(name) for name in sources]
    cached_total = cached_job_count(sources, tuple(p.stat().st_mtime for p in cache_files if p.exists()))
    listing_total = st.session_state.get("total_listings", cached_total)
    if listing_total > 1:
        max_jobs = st.slider("Jobs shown", min_value=1, max_value=listing_total,
                             value=min(cv.MAX_JOBS_SHOWN, listing_total))
    else:
        max_jobs = 1

st.title("CV tailoring")
st.caption(f"Builds a CV for a job from your own sentences, never rewritten. "
           f"Model: {cv.MODEL_ID}, running locally.")

# ---------------------------------------------------------------------
# 1. SEARCH
# ---------------------------------------------------------------------
with st.form("search", border=False):
    with st.container(horizontal=True, vertical_alignment="bottom"):
        keyword = st.text_input("Job keyword", value=cv.SEARCH_KEYWORD, placeholder="e.g. data analyst")
        searched = st.form_submit_button("Search", icon=":material/search:", type="primary")

if searched:
    if not source_names:
        st.warning("Select at least one job source in the sidebar.")
    else:
        refresh = data_source == "Download"
        with st.spinner(f"{'Downloading' if refresh else 'Reading cached jobs for'} {', '.join(source_names)}..."):
            jobs = fetch_jobs(list(source_names), refresh=refresh)
        results = cv.search_jobs(jobs, keyword.strip())
        cv.write_log("results", json.dumps(results, indent=2, ensure_ascii=False))
        st.session_state.results = results
        st.session_state.total_listings = len(jobs)
        st.session_state.search_keyword = keyword.strip()
        st.session_state.search_id += 1
        st.session_state.analysis = None
        st.rerun()  # redraws the sidebar slider with the new listing count

shown = st.session_state.results[:max_jobs]
if not shown:
    if "search_keyword" in st.session_state:
        st.info(f"No jobs match *{st.session_state.search_keyword}* among "
                f"{st.session_state.total_listings} listings. Try a broader keyword.")
    st.stop()

# ---------------------------------------------------------------------
# 2. PICK A JOB
# ---------------------------------------------------------------------
st.subheader("1. Pick a job")
st.caption(f"Found {len(st.session_state.results)} jobs for *{st.session_state.search_keyword}*. "
           f"Select a row to analyze it.")
event = st.dataframe(
    pd.DataFrame(shown)[["title", "company", "location", "source", "url"]],
    column_config={"url": st.column_config.LinkColumn("Link", display_text="Open")},
    hide_index=True,
    on_select="rerun",
    selection_mode="single-row",
    key=f"jobs_{st.session_state.search_id}",
    alt="Matching jobs; select one row to tailor a CV for it",
)
if not event.selection.rows:
    st.stop()

job = shown[event.selection.rows[0]]
analysis = st.session_state.analysis
if analysis is None or analysis["job"]["url"] != job["url"]:
    with st.expander("Job description"):
        st.text(job["description"])
    analyze(job)
    analysis = st.session_state.analysis
else:
    with st.expander("Job description"):
        st.text(job["description"])

if not analysis["requirements"]:
    st.warning("The model couldn't extract requirements from this job. Pick another one.")
    st.stop()

# ---------------------------------------------------------------------
# 3. REQUIREMENTS AND ANSWERS
# ---------------------------------------------------------------------
st.subheader("2. Requirements")
gaps = analysis["gaps"]
bank = cv.load_bank(cv.BANK_PATH)
selected = cv.select_bullets(bank, gaps, analysis["requirements"])
relevant_rows = cv.build_selection_rows(selected, True)
if relevant_rows:
    with st.expander(f"From your bank, relevant to this job ({len(relevant_rows)})", expanded=True):
        st.caption("Used exactly as written.")
        st.dataframe(pd.DataFrame(relevant_rows), hide_index=True, alt="Bank sentences matching the job")
else:
    st.info("Nothing in the bank matches this job's requirements.")

open_gaps = [g for g in gaps if g["status"] != "demonstrated"]
answers: list[str] = []
placements: list[str] = []
# "" = the free section; a sentence there is kept in the bank and pulled in whenever a job needs it
place_ids = {cv.FREE_SECTION_TITLE: ""}
place_ids.update({f"{r['title']} - {r['company']}" if r["company"] else r["title"]: r["id"] for r in bank["roles"]})
if open_gaps:
    st.markdown("**Not yet shown by your bank.** Fill in the *Your answer* column where it is true for you, "
                "choose where it goes in *Place under*, and leave it empty to skip. Answers are saved to the bank and go into the CV exactly as "
                "you type them: write each as a finished CV sentence, starting with an action verb.")
    gap_table = pd.DataFrame({
        "Missing requirements": [g["requirement"] for g in open_gaps],
        "Place under": pd.Series([cv.FREE_SECTION_TITLE] * len(open_gaps), dtype="string"),
        "Your answer": pd.Series([""] * len(open_gaps), dtype="string"),
    })
    edited = st.data_editor(
        gap_table,
        key=f"answers_{job['url']}_{analysis['round']}",
        hide_index=True,
        disabled=["Missing requirements"],
        column_config={
            "Place under": st.column_config.SelectboxColumn(
                "Place under", options=list(place_ids), required=True,
                help="Where the sentence goes in the CV, and in your bank for later jobs"),
            "Your answer": st.column_config.TextColumn(
                "Your answer", help="A finished CV sentence, or empty to skip", width="large"),
        },
        alt="Requirements not yet shown by the bank, with a column to type your answers",
    )
    answers = [a if isinstance(a, str) else "" for a in edited["Your answer"]]
    placements = [place_ids.get(place, "") for place in edited["Place under"]]
else:
    st.success("Your bank already demonstrates every requirement.")

# ---------------------------------------------------------------------
# 4. BUILD
# ---------------------------------------------------------------------
if st.button("Build CV", icon=":material/description:", type="primary"):
    build_cv(analysis, answers, placements)
    st.rerun()

if analysis["cv"]:
    cv_markdown = analysis["cv"]
    st.subheader("3. Your CV")
    with st.container(border=True):
        st.markdown(cv_markdown.rsplit("\n<!--", 1)[0])  # drop the "Tailored for" comment
    with st.container(horizontal=True):
        st.download_button("Download Markdown", cv_markdown, file_name=cv.CV_MARKDOWN_PATH.name,
                           mime="text/markdown", icon=":material/download:")
        st.download_button("Download PDF", cv_pdf_bytes(cv_markdown), file_name=cv.CV_PDF_PATH.name,
                           mime="application/pdf", icon=":material/picture_as_pdf:", type="primary")
