import pandas as pd
from jobspy import scrape_jobs
import google.generativeai as genai
import time
import random
import streamlit as st # New Library

# --- CONFIGURATION (Move outside function for Streamlit) ---
# Use Streamlit secrets for the API key (safer than hardcoding)
# To use this, create a .streamlit/secrets.toml file with the key.
if 'GEMINI_API_KEY' not in st.secrets:
    st.error("Missing Gemini API Key in Streamlit Secrets.")
    st.stop()
GOOGLE_API_KEY = st.secrets["GEMINI_API_KEY"] 

# Use a stable model
MODEL_NAME = 'gemini-2.5-flash'

# --- UTILITY FUNCTION ---
def summarize_job(description):
    """
    Uses the AI Agent to read the job description and create a summary.
    """
    if not description or len(description) < 50:
        return "No description available to summarize."
    
    prompt = f"""
    You are a helpful job assistant. Summarize the following job description in 3 concise bullet points for quick review:
    1. Key Tech Stack/Skills required.
    2. Main Responsibilities (1-2 sentences).
    3. Required Experience Level (choose ONE):
    - Fresher / Entry-level
    - 1–3 years
    - 3–5 years
    - 5+ years
    - Senior / Lead
    - Not specified
    
    Job Description:
    {description[:5000]}
    """
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI Summary Failed: {str(e)}"

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
        "Locations (One per line: e.g., India, Remote)",
        value="India\nUSA\nRemote"
    )
    
    RESULTS_WANTED = st.slider("Max Results Per Search Term/Location", 1, 15, 5)
    HOURS_OLD = st.slider("Maximum Age of Job Post (Hours)", 24, 168, 72)

    # Process inputs into lists
    JOB_SEARCH_TERM = [t.strip() for t in search_terms_input.split('\n') if t.strip()]
    LOCATION = [l.strip() for l in location_input.split('\n') if l.strip()]

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
                    site_name=["linkldin", "indeed"], 
                    search_term=term,
                    location=loc,
                    results_wanted=RESULTS_WANTED,
                    hours_old=HOURS_OLD,
                    linkedin_fetch_description=False 
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
        progress_percentage = (index + 1) / total_jobs
        progress_bar.progress(progress_percentage, text=f"Processing {index + 1} of {total_jobs}: {title[:40]}...")
        
        description = job.get('description', '')
        ai_summary = summarize_job(description)
        
        results_list.append({
            "Job Title": title,
            "Company": job.get('company', 'N/A'),
            "Location": job.get('location', 'N/A'),
            "Job Type": job.get('job_type', 'N/A'),
            "Experience Level": extracted_experience,
            "Key Skills": extracted_skills,
            "Responsibility": extracted_skills,
            "AI Summary": ai_summary,
            "Apply Link": job.get('job_url', 'N/A')
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
            "Apply Link": st.column_config.link_column("Apply Link"),
            "AI Summary": st.column_config.text_column("AI Summary", help="AI-generated key takeaways."),
            # Set other columns to text for better readability
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
