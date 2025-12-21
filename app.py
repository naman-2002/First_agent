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

MODEL_NAME = 'gemma-3-12b-it'
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

def extract_all_fields_raw(description: str, title: str | None = None) -> dict:
    """
    Single LLM call to extract experience, job type, and summary together.
    """
    prompt = f"""
You are a job analysis assistant.

Return ONLY valid JSON. No explanations. No markdown.

Rules:
- Extract experience requirement as written or inferred.
- Extract job type (Full-time, Part-time, Contract, Internship).
- If something is unclear, return "Not specified".
- Summary must be 3–4 concise sentences in plain text (NO bullet points).

JSON format:
{{
  "experience_raw": "Short sentence describing experience requirement or Not specified",
  "job_type_raw": "Full-time | Part-time | Contract | Internship | Not specified",
  "summary": "3–4 sentence plain-text summary"
}}

Job Title:
{title or "N/A"}

Job Description:
{description[:9000]}
"""
    response = _model.generate_content(
        prompt,
        generation_config={"temperature": 0.0}
    )
    return parse_json_with_retry(response.text, _model)



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


def map_job_type_to_bucket(raw_text: str) -> str:
    """Map free-text job type into one of: Full Time | Internship | Contract | Part-Time | Not specified"""
    if not raw_text:
        return "Not specified"
    text = raw_text.lower()

    # spelled/informal variations
    if re.search(r"\b(intern(ship)?|internship)\b", text):
        return "Internship"
    if re.search(r"\b(part[- ]?time|parttime)\b", text):
        return "Part-Time"
    if re.search(r"\b(contract|freelance|temporary)\b", text):
        return "Contract"
    if re.search(r"\b(full[- ]?time|fulltime|permanent)\b", text):
        return "Full Time"

    # sometimes job ads say "remote internship", so catch "intern" earlier
    if "intern" in text:
        return "Internship"
    if "part" in text:
        return "Part-Time"
    if "contract" in text or "freelance" in text:
        return "Contract"
    if "full" in text or "permanent" in text:
        return "Full Time"

    return "Not specified"



# -------------------------
# NEW: main two-pass extractor that replaces summarize_job
# -------------------------
def extract_job_fields(description: str, title: str = None):
    """
    One-pass LLM extraction + deterministic mapping
    """
    try:
        raw = extract_all_fields_raw(description, title)
    except Exception:
        raw = None

    if not raw:
        return {
            "experience_raw": "Not specified",
            "experience": "Not specified",
            "job_type_raw": "Not specified",
            "job_type_": "Not specified",
            "summary": "Not specified"
        }

    experience_raw = raw.get("experience_raw", "Not specified")
    job_type_raw = raw.get("job_type_raw", "Not specified")
    summary_raw = raw.get("summary", "Not specified")

    return {
        "experience_raw": experience_raw,
        "experience": map_experience_to_bucket(experience_raw),
        "job_type_raw": job_type_raw,
        "job_type_": map_job_type_to_bucket(job_type_raw),
        "summary": normalize_summary(summary_raw)
    }


# -------------------------
# The rest of your Streamlit app is mostly unchanged - just call extract_job_fields instead of summarize_job
# -------------------------

st.title("💡 Agentic Job Search Dashboard")
st.markdown("Enter your job criteria below and the agent will scrape, clean, and summarize the top results for you.")

with st.sidebar:
    st.header("Search Parameters")
    search_terms_input = st.text_area("Job Search Keywords (One per line)", value="Operations Research\nSupply Chain Management")
    location_input = st.text_area("Locations (One per line)", value="India\nUSA\nRemote")
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
        ai_output = extract_job_fields(description1, title=title)


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
