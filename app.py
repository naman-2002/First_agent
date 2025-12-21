import pandas as pd
from jobspy import scrape_jobs
import google.generativeai as genai
import time
import random
import streamlit as st # New Library
import json
import re
import html

# -------------------------
# CONFIG
# -------------------------
if 'GEMINI_API_KEY' not in st.secrets:
    st.error("Missing Gemini API Key in Streamlit Secrets.")
    st.stop()
GOOGLE_API_KEY = st.secrets["GEMINI_API_KEY"] 

MODEL_NAME = 'gemma-3-12b'
genai.configure(api_key=GOOGLE_API_KEY)
# create a single model handle to reuse
_model = genai.GenerativeModel(MODEL_NAME)


# -------------------------
# UTIL: parsing + cleaning (unchanged)
# -------------------------
def parse_json_with_retry(text, model, prompt_retry=None, max_tries=1):
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"(\{.*\})", text, flags=re.S)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass

    for _ in range(max_tries):
        if not prompt_retry:
            prompt_retry = "The previous output was not valid JSON. Please output only valid JSON for the required fields, and nothing else."
        resp = model.generate_content(prompt_retry, generation_config={"temperature": 0.0})
        try:
            return json.loads(resp.text)
        except Exception:
            m = re.search(r"(\{.*\})", resp.text, flags=re.S)
            if m:
                try:
                    return json.loads(m.group(1))
                except Exception:
                    pass
    return None


cleaning_dict = {"â€™": "'", "â€": " ", "\.": ".", "\+": "+", "\-": "-"}

def clean_description(text: str, preserve_paragraphs: bool = False) -> str:
    if not text:
        return ""
    text = html.unescape(text)
    for k, v in cleaning_dict.items():
        text = text.replace(k, v)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if preserve_paragraphs:
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n[ \t]+", "\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        text = re.sub(r"[^\S\n]+", " ", text)
        text = "\n".join(line.strip() for line in text.split("\n")).strip()
    else:
        text = re.sub(r"\s+", " ", text).strip()
    for k, v in cleaning_dict.items():
        text = text.replace(k, v)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text

def normalize_summary(summary):
    if isinstance(summary, list):
        return " ".join(str(s) for s in summary)
    if summary is None:
        return "Not specified"
    return str(summary)


# -------------------------
# NEW: small LLM helpers (single-purpose prompts)
# -------------------------
def llm_free_text(prompt: str) -> str:
    """Ask Gemini a single free-text question and return the raw text (deterministic)."""
    resp = _model.generate_content(prompt, generation_config={"temperature": 0.0})
    return resp.text.strip()


def ask_experience_raw(description: str, title: str | None = None) -> str:
    # Single focused prompt to get a human-readable answer (don't enforce buckets here)
    prompt = f"""
You are a careful job analyst. Answer in one short sentence (1-2 lines).

Question: Based on the Job Title and Job Description below, what experience level is required for this role?

Guidance:
- If the description explicitly states years (e.g., "3+ years", "minimum 2 years"), quote that.
- If no explicit years are present, infer a reasonable label (e.g., "Senior-level, likely 5+ years", "Mid-level, ~3 years") using job title and responsibility scope.
- If there's no signal, answer "Not specified".

Job Title:
{title or "N/A"}

Job Description:
{description[:8000]}
"""
    return llm_free_text(prompt)


def ask_job_type_raw(description: str, title: str | None = None) -> str:
    prompt = f"""
You are a job analyst. Answer in a single short phrase (e.g., "Full-time", "Part-time", "Contract", "Internship", "Remote internship", "Not specified").

Question: What job type does this posting offer (full-time, part-time, contract, internship, freelance, etc.)? Use any signals in title or description. If unclear, return "Not specified".

Job Title:
{title or "N/A"}

Job Description:
{description[:8000]}
"""
    return llm_free_text(prompt)


def ask_summary_text(description: str) -> str:
    prompt = f"""
You are a clear summarizer. Provide a concise 3-4 sentence summary of the main responsibilities and focus of this job posting. Do not use bullet points. Return only the short paragraph.

Job Description:
{description[:8000]}
"""
    return llm_free_text(prompt)


# -------------------------
# NEW: deterministic mappers (LLM raw -> bucket)
# -------------------------
def map_experience_to_bucket(raw_text: str) -> str:
    """Map an LLM free-text experience answer (or direct JD text) into one of your buckets."""
    if not raw_text:
        return "Not specified"
    text = raw_text.lower()

    # explicit "fresher" or "intern"
    if re.search(r"\b(fresher|intern(ship)?|entry[- ]?level)\b", text):
        return "Fresher"

    # explicit ranges / patterns we like to catch first
    m = re.search(r"(\d+\s*\+\s*years|\d+\s*-\s*\d+\s*years|\d+\s*years)", text)
    if m:
        s = m.group(1)
        # normalize some common forms to your specified buckets
        if "+" in s:
            num = re.search(r"(\d+)", s)
            if num and int(num.group(1)) >= 5:
                return "5+ years"
            elif num and int(num.group(1)) >= 3:
                return "3+ years"
            else:
                return "1+ years"
        # range like "1-3 years"
        r = re.search(r"(\d+)\s*-\s*(\d+)", s)
        if r:
            a, b = int(r.group(1)), int(r.group(2))
            if a == 0 and b <= 1:
                return "0-1 years"
            if a <= 1 and b <= 3:
                return "1-3 years"
            if b <= 5:
                return "3-5 years"
            return "5+ years"
        # single "2 years"
        n = re.search(r"(\d+)\s*years?", s)
        if n:
            nval = int(n.group(1))
            if nval <= 1:
                return "0-1 years"
            if nval <= 3:
                return "1-3 years"
            if nval <= 5:
                return "3-5 years"
            return "5+ years"

    # seniority words
    if re.search(r"\b(senior|lead|principal|manager|director)\b", text):
        # prefer higher buckets for manager/director
        if re.search(r"\b(manager|director|principal)\b", text):
            return "5+ years"
        return "3+ years"

    # junior keywords
    if re.search(r"\b(junior|associate|entry)\b", text):
        return "0-2 year"

    # fallback
    return "Not specified"


def map_job_type_to_bucket(raw_text: str, scraped_job_type) -> str:
    """
    Map free-text job type into one of:
    Full Time | Intership | Contract | Part-Time | Not specified
    """

    # -------- SAFE NORMALIZATION --------
    def normalize(val):
        if val is None:
            return ""
        if isinstance(val, list):
            val = " ".join(str(v) for v in val)
        if isinstance(val, float):  # NaN case
            return ""
        return str(val).strip().lower()

    scraped = normalize(scraped_job_type)
    raw = normalize(raw_text)

    # -------- Prefer scraped job_type if valid --------
    if scraped not in ("", "n/a", "na"):
        if "intern" in scraped:
            return "Intership"
        if "part" in scraped:
            return "Part-Time"
        if "contract" in scraped or "freelance" in scraped:
            return "Contract"
        if "full" in scraped or "permanent" in scraped:
            return "Full Time"

    # -------- Fallback to LLM raw text --------
    if "intern" in raw:
        return "Intership"
    if "part" in raw:
        return "Part-Time"
    if "contract" in raw or "freelance" in raw:
        return "Contract"
    if "full" in raw or "permanent" in raw:
        return "Full Time"

    return "Not specified"



# -------------------------
# NEW: main two-pass extractor that replaces summarize_job
# -------------------------
def extract_job_fields(description: str, title: str = None, scraped_job_type: str | None = None):
    """
    Two-pass extraction:
    1) Ask LLM short free-text questions: experience_raw, job_type_raw, summary_raw
    2) Map those free-text answers into deterministic buckets
    Returns a dict with raw and bucketed fields plus summary text.
    """
    # 1) LLM free-text understanding (single-purpose)
    try:
        experience_raw = ask_experience_raw(description, title)
    except Exception as e:
        experience_raw = "Not specified"

    try:
        job_type_raw = ask_job_type_raw(description, title)
    except Exception as e:
        job_type_raw = "Not specified"

    try:
        summary_raw = ask_summary_text(description)
    except Exception as e:
        summary_raw = "Not specified"

    # 2) deterministic mapping
    experience_bucket = map_experience_to_bucket(experience_raw)
    job_type_bucket = map_job_type_to_bucket(job_type_raw, scraped_job_type)

    # normalize summary to plain text (defensive)
    summary_text = normalize_summary(summary_raw)

    return {
        "experience_raw": experience_raw,
        "experience": experience_bucket,
        "job_type_raw": job_type_raw,
        "job_type_": job_type_bucket,
        "summary": summary_text
    }


# -------------------------
# The rest of your Streamlit app is mostly unchanged - just call extract_job_fields instead of summarize_job
# -------------------------

st.title("💡 Agentic Job Search Dashboard")
st.markdown("Enter your job criteria below and the agent will scrape, clean, and summarize the top results for you.")

with st.sidebar:
    st.header("Search Parameters")
    search_terms_input = st.text_area("Job Search Keywords (One per line)", value="Operations Research Scientist")
    location_input = st.text_area("Locations (One per line)", value="India")
    google_search = st.text_area("Google Search Terms", value="operation research jobs in remote")
    RESULTS_WANTED = st.slider("Max Results Per Search Term/Location", 1, 15, 5)
    HOURS_OLD = st.slider("Maximum Age of Job Post (Hours)", 24, 168, 72)
    JOB_SEARCH_TERM = [t.strip() for t in search_terms_input.split('\n') if t.strip()]
    LOCATION = [l.strip() for l in location_input.split('\n') if l.strip()]
    GOOGLE_SEARCH = google_search.strip()

    if st.button("Run Job Search Agent 🚀"):
        st.session_state['run_search'] = True

if 'run_search' in st.session_state and st.session_state['run_search']:
    all_jobs_dfs = []
    st.subheader(f"Searching for {len(JOB_SEARCH_TERM)} Terms in {len(LOCATION)} Locations...")
    status_box = st.empty()
    for term in JOB_SEARCH_TERM:
        for loc in LOCATION:
            status_box.info(f"🔎 Searching: '{term}' in '{loc}'...")
            try:
                current_jobs = scrape_jobs(
                    site_name=["linkedin", "indeed"], 
                    search_term=term,
                    location=loc,
                    google_search_term=GOOGLE_SEARCH,
                    results_wanted=RESULTS_WANTED,
                    hours_old=HOURS_OLD,
                    linkedin_fetch_description=True
                )
                if not current_jobs.empty:
                    all_jobs_dfs.append(current_jobs)
                time.sleep(random.uniform(3, 6))
            except Exception as e:
                status_box.error(f"❌ Scraping error for {term}/{loc}: {e}")
                time.sleep(5)

    if not all_jobs_dfs:
        status_box.warning("❌ No jobs found matching your criteria.")
        st.session_state['run_search'] = False
        st.stop()

    status_box.info("Combining and cleaning job results...")
    master_df = pd.concat(all_jobs_dfs, ignore_index=True)
    master_df.drop_duplicates(subset=['job_url'], keep='first', inplace=True)
    master_df['description'] = master_df['description'].fillna('') 

    status_box.info(f"Found {len(master_df)} unique jobs. Starting AI summarization...")

    results_list = []
    total_jobs = len(master_df)
    progress_bar = st.progress(0, text=f"Processing 0 of {total_jobs} summaries...")

    for index, job in master_df.iterrows():
        title = job.get('title', 'N/A')
        if total_jobs > 0:
            raw_progress = ((index + 1) / total_jobs) * 100
            progress_percentage = max(0, min(100, round(raw_progress)))
        else:
            progress_percentage = 0

        description = job.get('description', '')
        description1 = clean_description(description, preserve_paragraphs=False)

        # <-- new extractor call -->
        ai_output = extract_job_fields(description1, title=title, scraped_job_type=job.get('job_type', 'N/A'))

        results_list.append({
            "Job Title": title,
            "Description": description1,
            "Company": job.get('company', 'N/A'),
            "Location": job.get('location', 'N/A'),
            "Job Type": job.get('job_type', 'N/A'),
            "Ai_job_type": ai_output["job_type_"],
            "Experience Level": ai_output["experience"],
            "Experience_raw": ai_output["experience_raw"],
            "Apply Link": job.get('job_url', 'N/A'),
            "Summary": ai_output["summary"]
        })

        time.sleep(0.5)

    progress_bar.empty()
    status_box.success(f"🎉 Agent finished! {len(results_list)} unique jobs summarized.")

    final_df = pd.DataFrame(results_list)
    st.subheader("Results Dashboard")
    st.dataframe(final_df, hide_index=True)

    st.download_button(
        label="Download Full Report as CSV",
        data=final_df.to_csv().encode('utf-8'),
        file_name='agent_job_report.csv',
        mime='text/csv',
    )
    st.session_state['run_search'] = False
