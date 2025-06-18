

import streamlit as st
import requests 
from typing import Optional, Dict, Any, Tuple, List 
from streamlit.runtime.uploaded_file_manager import UploadedFile 
from requests.models import Response 


FASTAPI_URL: str = "https://resume-analyzer-rv58.onrender.com/analyze/"


st.set_page_config(page_title="AI Resume Analyzer", layout="wide", initial_sidebar_state="collapsed")

st.title("📄 AI Resume Analyzer Demo")
st.markdown("""
Upload a candidate's resume and a job description (both in PDF format).
The application will call a backend API to analyze the resume and compare it against the job description using AI.
""")

st.caption("*(Ensure the FastAPI backend server is running)*")


col1, col2 = st.columns(2)

resume_file: Optional[UploadedFile] = None
jd_file: Optional[UploadedFile] = None

with col1:
    
    resume_file = st.file_uploader("1. Upload Resume PDF", type="pdf", key="resume")
with col2:

    jd_file = st.file_uploader("2. Upload Job Description PDF", type="pdf", key="jd")


analyze_button: bool = st.button(
    "Analyze and Compare",
    type="primary",
    disabled=(not resume_file or not jd_file)
)


if analyze_button:
    
    if resume_file is not None and jd_file is not None:
      
        with st.spinner("🚀 Sending files to backend and analyzing... This might take a minute."):
            try:

                files: Dict[str, Tuple[Optional[str], bytes, Optional[str]]] = {
                    "resume_file": (resume_file.name, resume_file.getvalue(), resume_file.type),
                    "jd_file": (jd_file.name, jd_file.getvalue(), jd_file.type),
                }

                
                st.info(f"📞 Contacting API backend at {FASTAPI_URL}...")
               
                response: Response = requests.post(FASTAPI_URL, files=files, timeout=300)

                
                if response.status_code == 200:
                    st.success("✅ Analysis Complete!")
                    
                    results: Dict[str, Any] = response.json()

                   
                    st.subheader("📊 Resume Analysis Results")
                    
                    analysis: Dict[str, Any] = results.get("resume_analysis", {})
                    st.markdown(f"**Summary:**\n {analysis.get('summary', 'Not Available')}")

                    
                    with st.expander("✅ Strengths"):
                        strengths: List[str] = analysis.get('strengths', [])
                        if strengths:
                            for item in strengths: st.markdown(f"- {item}")
                        else:
                            st.markdown("No specific strengths identified by the AI.")

                    with st.expander("⚠️ Weaknesses / Areas for Improvement"):
                        weaknesses: List[str] = analysis.get('weaknesses', [])
                        if weaknesses:
                            for item in weaknesses: st.markdown(f"- {item}")
                        else:
                            st.markdown("No specific weaknesses or missing areas identified by the AI.")

                    with st.expander("🛠️ Identified Skills"):
                         skills: List[str] = analysis.get('skills', [])
                         if skills:
                             
                             st.markdown(", ".join(skills))
                         else:
                             st.markdown("No specific skills identified by the AI.")

                
                    st.subheader("📈 Resume vs. Job Description Similarity")
                    similarity: Dict[str, Any] = results.get("similarity_results", {})
                    
                    overall_score: float = similarity.get('overall_score', float('inf')) 
                    sim_perc: float = similarity.get('similarity_percentage', 0.0) 

                    st.metric(label="Similarity Score (Lower is Better - FAISS L2 Distance)", value=f"{overall_score:.4f}")
                   
                    st.progress(sim_perc / 100.0, text=f"Approx. Match: {sim_perc:.2f}% (Note: Scaling is approximate!)")

                    st.markdown("**Most Relevant Resume Sections (based on JD similarity):**")
                    chunks: List[Dict[str, Any]] = similarity.get('relevant_chunks', [])
                    if chunks:
                       
                        for i, chunk_data in enumerate(chunks):
                            chunk_score = chunk_data.get('score', 0.0)
                            chunk_page = chunk_data.get('page', 'N/A')
                            chunk_content = chunk_data.get('content', 'N/A')
                            with st.expander(f"Match {i+1} (Score: {chunk_score:.4f} | Page: {chunk_page})"):
                               
                                st.caption(chunk_content)
                    else:
                        st.markdown("No relevant sections identified based on similarity search.")

                  
                    proc_time: Optional[float] = results.get("processing_time_seconds")
                    if proc_time is not None:
                        st.caption(f"Backend processing time: {proc_time:.2f} seconds.")

       
                else:
                    st.error(f"❌ API Error: Received Status Code {response.status_code}")
                    try:
                     
                        error_detail: str = response.json().get("detail", response.text)
                        st.error(f"Details: {error_detail}")
                    except requests.exceptions.JSONDecodeError:
                   
                        st.error(f"Could not decode error response body: {response.text}")

    
            except requests.exceptions.RequestException as e:
                st.error(f"🔗 Network Error: Could not connect to the API backend at {FASTAPI_URL}.")
                st.error(f"Details: {e}")
                st.info("Troubleshooting Tips: \n1. Is the FastAPI server running? \n2. Is the URL correct? \n3. Check for network issues.")
            
            except Exception as e:
                st.error("🔥 An unexpected error occurred in the Streamlit application.")
               
                st.exception(e)
    else:

        st.warning(" P lease upload both Resume and Job Description PDF files before clicking Analyze.")