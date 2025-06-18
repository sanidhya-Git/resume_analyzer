

import time
from typing import Dict, List 


from fastapi import (
    FastAPI,
    File,       
    UploadFile,  
    HTTPException, 
    status       
)

from fastapi.middleware.cors import CORSMiddleware

from .schemas import AnalysisAndSimilarityResponse, ResumeAnalysis, SimilarityResult


from .services import (
    process_pdf_bytes,
    perform_resume_analysis,
    calculate_resume_jd_similarity,
    initialization_error 
)


from langchain_core.documents import Document


app = FastAPI(
    title="AI Resume Analyzer API",
    description="API for analyzing resumes against job descriptions using LangChain and Gemini.",
    version="0.1.0",
    
)

origins = [
    "http://localhost",
    "http://localhost:8501", 
    "https://resumeanalyzer-bxgrwmhuwashldzjyugvoj.streamlit.app/",

]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  

    allow_credentials=True, 
    allow_methods=["*"],    
    allow_headers=["*"],    
)



@app.on_event("startup")
async def startup_event() -> None:
    """
    Asynchronous function executed once when the FastAPI application starts.
    Used here to check and log the initialization status of critical components.
    """
    print("FastAPI application startup commencing...")
    if initialization_error:
       
        print(f"--- STARTUP WARNING ---")
        print(initialization_error)
        print("-----------------------")
    else:
        
        print("Google AI components appear to be initialized successfully.")
    print("Startup complete. API is ready.")



@app.get("/", tags=["General"])
async def read_root() -> Dict[str, str]:
    """
    Root endpoint (GET /).

    Provides a simple status message indicating the API is running.
    Useful for basic health checks. Added to the 'General' tag group in API docs.
    """
    return {"status": "AI Resume Analyzer API is running!"}


@app.post(
    "/analyze/",
    response_model=AnalysisAndSimilarityResponse,
    tags=["Analysis"],
    summary="Analyze Resume and Job Description PDFs." 
)
async def analyze_resume_and_jd(
    resume_file: UploadFile = File(..., description="The candidate's resume PDF file."),
    jd_file: UploadFile = File(..., description="The job description PDF file.")
) -> AnalysisAndSimilarityResponse:
    """
    Handles the upload of resume and job description PDF files, orchestrates
    the analysis and similarity comparison using backend services, and returns
    the combined results.

    Args:
        resume_file: The uploaded resume PDF file. Sent as part of a multipart/form-data request.
        jd_file: The uploaded job description PDF file. Sent as part of a multipart/form-data request.

    Returns:
        An AnalysisAndSimilarityResponse object containing the structured results
        of the resume analysis and similarity comparison.

    Raises:
        HTTPException(400): If files are not PDFs or cannot be read.
        HTTPException(422): If processing fails due to invalid data or service errors (ValueError from services).
        HTTPException(500): For unexpected internal server errors during processing.
        HTTPException(503): If critical AI components failed to initialize on startup.
    """
    print("\nReceived request for /analyze/.")
    start_time: float = time.time() 

    if initialization_error:
         print(f"ERROR: Attempted access but initialization failed: {initialization_error}.")
      
         raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"API service unavailable due to initialization failure: {initialization_error}."
         )

 
    if not resume_file.filename or not resume_file.filename.lower().endswith(".pdf") or \
       not jd_file.filename or not jd_file.filename.lower().endswith(".pdf"):
        print(f"ERROR: Invalid file type. Resume='{resume_file.filename}', JD='{jd_file.filename}'")

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file type. Please upload PDF files only for both resume and job description."
        )

    resume_bytes: bytes
    jd_bytes: bytes
    try:
        
        print(f"Reading resume file: {resume_file.filename}.")
        resume_bytes = await resume_file.read()
        print(f"Reading job description file: {jd_file.filename}.")
        jd_bytes = await jd_file.read()
    except Exception as e:
  
         print(f"ERROR reading files: {e}.")
         raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded file content: {e}."
         )
    finally:
    
        await resume_file.close()
        await jd_file.close()

    print(f"Resume size: {len(resume_bytes)} bytes, JD size: {len(jd_bytes)} bytes.")


    try:
        # Step 1: Process PDFs into LangChain Document chunks.
        print("Processing Resume PDF...")
        resume_docs: List[Document] = process_pdf_bytes(resume_bytes, resume_file.filename)
        print("Processing Job Description PDF...")
        jd_docs: List[Document] = process_pdf_bytes(jd_bytes, jd_file.filename)



     
        print("Performing resume analysis...")
        analysis_result: ResumeAnalysis = perform_resume_analysis(resume_docs)

     
        print("Calculating similarity...")
        similarity_result: SimilarityResult = calculate_resume_jd_similarity(resume_docs, jd_docs)

    except ValueError as ve:
        
         print(f"ERROR during processing: {ve}")
       
         raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(ve)
         )
    except Exception as e:
      
         print(f"UNEXPECTED INTERNAL ERROR during processing: {e}")
         # TODO: Log the full traceback here for detailed debugging in production.

         raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected internal error occurred during processing: {e}"
         )


    end_time: float = time.time()
    processing_time: float = end_time - start_time
    print(f"Request processing finished successfully in {processing_time:.2f} seconds.")


    return AnalysisAndSimilarityResponse(
        resume_analysis=analysis_result,
        similarity_results=similarity_result,
        processing_time_seconds=processing_time
    )