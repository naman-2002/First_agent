import pandas as pd
from jobspy import scrape_jobs
import google.generativeai as genai
import time
import random
import streamlit as st # New Library
import json
import re
import html



def parse_json_with_retry(text, model, prompt_retry=None, max_tries=1):
    # Try direct parse
    try:
        return json.loads(text)
    except Exception:
        # attempt to find first {...} block
        m = re.search(r"(\{.*\})", text, flags=re.S)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass

    # If we get here, ask the model to return ONLY JSON (one retry)
    for _ in range(max_tries):
        if not prompt_retry:
            prompt_retry = "The previous output was not valid JSON. Please output only valid JSON for the required fields, and nothing else."
        resp = model.generate_content(prompt_retry, generation_config={"temperature": 0.0})
        try:
            return json.loads(resp.text)
        except Exception:
            # try extracting substring
            m = re.search(r"(\{.*\})", resp.text, flags=re.S)
            if m:
                try:
                    return json.loads(m.group(1))
                except Exception:
                    pass
    return None


# --- CONFIGURATION (Move outside function for Streamlit) ---
# Use Streamlit secrets for the API key (safer than hardcoding)
# To use this, create a .streamlit/secrets.toml file with the key.
if 'GEMINI_API_KEY' not in st.secrets:
    st.error("Missing Gemini API Key in Streamlit Secrets.")
    st.stop()
GOOGLE_API_KEY = st.secrets["GEMINI_API_KEY"] 

# Use a stable model
MODEL_NAME = 'gemini-2.5-flash'

cleaning_dict = {
    "â€™": "'",
    "â€": " ",
    "\.": ".",
    "\+": "+",
    "\-": "-"
}


def clean_description(text: str, preserve_paragraphs: bool = False) -> str:
    """
    Clean scraped job descriptions.

    - If preserve_paragraphs=True: keep up to TWO consecutive newlines (paragraph breaks),
      but remove excessive blank lines and normalize in-paragraph spacing.
    - If preserve_paragraphs=False: collapse all whitespace into single spaces (fully flattened).
    """
    if not text:
        return ""

    # 1) HTML entities -> plain text
    text = html.unescape(text)

    # 2) apply small token fixes you discovered
    for k, v in cleaning_dict.items():
        text = text.replace(k, v)

    # 3) normalize line endings (Windows CRLF -> LF)
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    if preserve_paragraphs:
        # A) collapse 3+ newlines into exactly 2 (preserve paragraph breaks)
        text = re.sub(r"\n{3,}", "\n\n", text)

        # B) remove spaces that start or end lines
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n[ \t]+", "\n", text)

        # C) collapse multiple spaces/tabs inside lines to one space
        text = re.sub(r"[ \t]{2,}", " ", text)

        # D) collapse remaining sequences of whitespace longer than 1 char but keep single newlines
        # Replace any run of whitespace that does NOT contain '\n' with a single space
        text = re.sub(r"[^\S\n]+", " ", text)

        # E) trim leading/trailing whitespace on whole text and each line
        text = "\n".join(line.strip() for line in text.split("\n")).strip()

    else:
        # Fully flatten: collapse everything to single-space tokens (no newlines)
        text = re.sub(r"\s+", " ", text).strip()

    # 2) apply small token fixes you discovered
    for k, v in cleaning_dict.items():
        text = text.replace(k, v)

    # Optional: final tiny normalizations
    text = re.sub(r"\n{3,}", "\n\n", text)  # safety
    return text

def normalize_summary(summary):
    if isinstance(summary, list):
        return " ".join(str(s) for s in summary)
    if summary is None:
        return "Not specified"
    return str(summary)


def summarize_job(description):
    if not description or len(description) < 50:
        return {
            "job_type_" : "Not specified",
            "experience": "Not specified",
            "summary": "No description available"
        }

    prompt = f"""
    You are a job analysis assistant.

    Return ONLY valid JSON. No explanations.

    Rules:
    - what is the job type given job description is offering.
    - Is job is a intership, part time, full time or what, plese choose the type mentioned in the json format.
    - First check if years of experience are explicitly mentioned.
    - If yes, map them to the closest bucket.
    - If not mentioned, infer experience using:
      - Job title (e.g., Senior, Lead, Manager)
      - Role scope and responsibility language
    - If still unclear, return "Not specified".
    - Do NOT invent precise years—only choose from the allowed closest buckets.
    - Summary must be 3–4 concise sentences in plain text (NO bullet points).

    JSON format:
    {{
      "job_type_" : Full Time | Intership | Contract | Part-Time,
      "experience": "0-1 years | 1+ years | 0-2 year | 2+ years | 3+ years | 1-3 years | 3-5 years | 5+ years | Fresher | Not specified",
      "summary": "3–4 sentence summary in plain text"
    }}

    Job Description:
    {description[:9000]}

    """

    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel(MODEL_NAME)

        # 1️⃣ Ask the LLM
        response = model.generate_content(
        prompt,
        generation_config={"temperature": 0.0}
        )

        # 2️⃣ Parse safely + retry if needed
        ai_json = parse_json_with_retry(
            response.text,
            model,
            prompt_retry="Please return ONLY valid JSON. No extra text."
        )

        # 3️⃣ Final guard
        if ai_json is None:
            return {
                "job_type_" : "Not specified",
                "experience": "Not specified",
                "summary": "AI parsing failed"
            }

        return ai_json

    except Exception as e:
        return {
            "job_type_" : "Not specified",
            "experience": "Not specified",
            "summary": f"AI error: {str(e)}"
        }



# --- STREAMLIT FRONTEND/AGENT RUNNER ---
st.title("💡 Agentic Job Search Dashboard")
st.markdown("Enter your job criteria below and the agent will scrape, clean, and summarize the top results for you.")

with st.sidebar:
    st.header("Search Parameters")
    
    # User inputs replace the hardcoded lists
    search_terms_input = st.text_area(
        "Job Search Keywords (One per line)",
        value="Operations Research\nSupply Chain Management"
    )
    location_input = st.text_area(
        "Locations (One per line)",
        value="India\nUSA\nRemote"
    )

    google_search = st.text_area(
        "Google Search Terms",
        value= "operation research jobs in remote"
    )
    
    RESULTS_WANTED = st.slider("Max Results Per Search Term/Location", 1, 15, 5)
    HOURS_OLD = st.slider("Maximum Age of Job Post (Hours)", 24, 168, 72)

    # Process inputs into lists
    JOB_SEARCH_TERM = [t.strip() for t in search_terms_input.split('\n') if t.strip()]
    LOCATION = [l.strip() for l in location_input.split('\n') if l.strip()]
    GOOGLE_SEARCH = [l.strip() for l in google_search.split('\n') if l.strip()]

    if st.button("Run Job Search Agent 🚀"):
        st.session_state['run_search'] = True
    
# Main display logic
if 'run_search' in st.session_state and st.session_state['run_search']:
    
    all_jobs_dfs = []
    
    st.subheader(f"Searching for {len(JOB_SEARCH_TERM)} Terms in {len(LOCATION)} Locations...")
    status_box = st.empty() # Placeholder for status updates

    # 1. SCRAPING LOOP
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
    
    # 2. CONSOLIDATE AND CLEAN
    status_box.info("Combining and cleaning job results...")
    master_df = pd.concat(all_jobs_dfs, ignore_index=True)
    master_df.drop_duplicates(subset=['job_url'], keep='first', inplace=True)
    master_df['description'] = master_df['description'].fillna('') 
    
    status_box.info(f"Found {len(master_df)} unique jobs. Starting AI summarization...")

    # 3. AI PROCESSING (Using st.progress for visual appeal)
    results_list = []
    total_jobs = len(master_df)
    progress_bar = st.progress(0, text=f"Processing 0 of {total_jobs} summaries...")
    
    for index, job in master_df.iterrows():
        title = job.get('title', 'N/A')
        
        # Display progress
        if total_jobs > 0:
            raw_progress = ((index + 1) / total_jobs) * 100
            progress_percentage = max(0, min(100, round(raw_progress)))
        else:
            progress_percentage = progress_percentage = 0
            
        description = job.get('description', '')
        description1 = clean_description(description, preserve_paragraphs=False)
        ai_output = summarize_job(description1)
        
        results_list.append({
            "Job Title": title,
            "Description": description1,
            "Company": job.get('company', 'N/A'),
            "Location": job.get('location', 'N/A'),
            "Job Type": job.get('job_type', 'N/A'),
            "Ai_job_type" : ai_output["job_type_"],
            "Experience Level": ai_output["experience"],
            "Apply Link": job.get('job_url', 'N/A'),
            "Summary": normalize_summary(ai_output["summary"])
        })

        time.sleep(0.5) # Shorter sleep here since the AI call is the bottleneck
        
    progress_bar.empty()
    status_box.success(f"🎉 Agent finished! {len(results_list)} unique jobs summarized.")

    # 4. REPORT (Attractive Display)
    final_df = pd.DataFrame(results_list)

    st.subheader("Results Dashboard [Image of Data Visualization Dashboard]")

    st.dataframe(
        final_df,
        column_config={
            "Apply Link": st.column_config.LinkColumn("Apply Link"),
            "AI Summary": st.column_config.TextColumn(
            "AI Summary",
            help="AI-generated key takeaways."
            ),
        },
        hide_index=True
    )

    st.download_button(
        label="Download Full Report as CSV",
        data=final_df.to_csv().encode('utf-8'),
        file_name='agent_job_report.csv',
        mime='text/csv',
    )
    st.session_state['run_search'] = False
