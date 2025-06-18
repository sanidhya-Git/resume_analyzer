
import os
import tempfile
from typing import List, Optional, Tuple 
import time

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document


from .config import settings
from .schemas import ResumeAnalysis, SimilarityResult, RelevantChunk

llm: Optional[ChatGoogleGenerativeAI] = None
embeddings: Optional[GoogleGenerativeAIEmbeddings] = None
initialization_error: Optional[str] = None 


try:
    print("Initializing Google AI components...")
    
    llm = ChatGoogleGenerativeAI(
        model=settings.llm_model_name,
        google_api_key=settings.google_api_key,
        temperature=0.3 
    
    )

    embeddings = GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model_name,
        google_api_key=settings.google_api_key
    )
    print("Google AI components initialized successfully.")
except Exception as e:
  
    initialization_error = f"Failed to initialize Google AI components: {e}."
    print(f"ERROR: {initialization_error}") 


def _check_components() -> None:
    """
    Internal helper to verify that AI components were initialized successfully.

    Raises:
        ValueError: If components failed to initialize or are unavailable.
    """
    if initialization_error:

        raise ValueError(initialization_error)
    if not llm or not embeddings:

        raise ValueError("LLM or Embeddings components are not available.")

def process_pdf_bytes(file_bytes: bytes, file_name: str = "document") -> List[Document]:
    """
    Loads PDF content from raw bytes, saves it temporarily, splits it into documents.

    Args:
        file_bytes: The raw byte content of the PDF file.
        file_name: An identifier for the file being processed (for logging). Defaults to "document".

    Returns:
        A list of LangChain Document objects, each representing a chunk of the PDF text.
        Returns an empty list if no text could be extracted.

    Raises:
        ValueError: If PDF processing fails due to an error (e.g., corrupted file, library issue, etc.).
    """
    print(f"Processing PDF: {file_name}.")
    _check_components() 
    docs: List[Document] = []
    temp_file_path: Optional[str] = None 

    try:

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(file_bytes)
            temp_file_path = temp_file.name 

       
        loader = PyPDFLoader(temp_file_path)
        
        docs = loader.load_and_split(
            text_splitter=RecursiveCharacterTextSplitter(
              
                chunk_size=1200,
                
                chunk_overlap=200
            )
        )
        print(f"Loaded and split {file_name} into {len(docs)} chunks.")
    except Exception as e:

        print(f"Error processing {file_name} PDF bytes: {e}")

        raise ValueError(f"Failed to process PDF '{file_name}': {e}")
    finally:
  
        if temp_file_path and os.path.exists(temp_file_path):
            print(f"Cleaning up temporary file: {temp_file_path}.")
            os.remove(temp_file_path)


    if not docs:
         print(f"Warning: No documents (text chunks) extracted from {file_name}.")
      
    return docs

def perform_resume_analysis(resume_docs: List[Document]) -> ResumeAnalysis:
    """
    Analyzes the content of resume documents using an LLM with structured output.

    Args:
        resume_docs: A list of Document objects representing the resume content chunks.

    Returns:
        A ResumeAnalysis object containing the structured analysis results (summary,
        strengths, weaknesses, and skills). Returns a default structure if input is empty.

    Raises:
        ValueError: If the LLM component is not initialized or if the AI analysis fails.
    """
    print("Performing resume analysis...")
    _check_components() 

    
    if not resume_docs:
        print("Warning: No resume documents provided for analysis, returning default structure.")
        
        return ResumeAnalysis(summary="No resume text provided or extracted.", strengths=[], weaknesses=["Resume content appears empty."], skills=[])

   
    analysis_prompt_template = ChatPromptTemplate.from_template(
        """Analyze the following resume text and provide the requested information strictly based ONLY on the text provided.
        If information for a field isn't present or couldn't be deduced/inferred, indicate that clearly (e.g., in weaknesses).

        Resume Text:
        ---
        {resume_text}
        ---

        Provide:
        1. A concise summary (2-3 sentences) of the candidate's overall profile and experience level.
        2. 3-5 bullet points highlighting key strengths relevant to a professional context (achievements, key skills demonstrated, strong experience, etc.).
        3. 2-4 bullet points highlighting potential weaknesses or areas clearly missing important information from the resume (e.g., lack of metrics, unclear career progression, missing contact info - if applicable). If none are obvious, state that.
        4. A list of key technical or professional skills explicitly mentioned in the resume.
        """
    )


    resume_full_text: str = "\n".join([doc.page_content for doc in resume_docs])

 
    MAX_LEN: int = 15000 
    if len(resume_full_text) > MAX_LEN:
        print(f"Warning: Resume text length ({len(resume_full_text)}) exceeds limit ({MAX_LEN}). Truncating.")
        resume_full_text = resume_full_text[:MAX_LEN]

    
    if not resume_full_text.strip():
         print("Warning: Resume text is empty after processing, returning default structure.")
         return ResumeAnalysis(summary="Resume text was empty or could not be processed.", strengths=[], weaknesses=["Resume content appears empty."], skills=[])

  
    structured_llm = llm.with_structured_output(ResumeAnalysis)


    analysis_chain = analysis_prompt_template | structured_llm

    try:
        # Time the LLM call for performance monitoring.
        start_time = time.time()
        # Invoke the chain with the combined resume text.
        analysis_result: ResumeAnalysis = analysis_chain.invoke({"resume_text": resume_full_text})
        end_time = time.time()
        print(f"LLM analysis completed in {end_time - start_time:.2f} seconds.")
        # Return the Pydantic object directly (FastAPI will serialize it).
        return analysis_result
    except Exception as e:
        # Catch errors during the LLM interaction (e.g., API errors, parsing errors, etc.).
        print(f"Error during resume analysis LLM call: {e}.")
        raise ValueError(f"AI analysis failed: {e}.")


def calculate_resume_jd_similarity(
    resume_docs: List[Document], jd_docs: List[Document]
) -> SimilarityResult:
    """
    Calculates the similarity between resume and job description documents
    using vector embeddings and FAISS vector store.

    Args:
        resume_docs: List of Document objects for the resume.
        jd_docs: List of Document objects for the job description.

    Returns:
        A SimilarityResult object containing scores and relevant resume chunks.
        Returns a default structure if input documents are missing.

    Raises:
        ValueError: If Embedding component is not initialized or if similarity calculation fails.
    """
    print("Calculating similarity...")
    _check_components() 

    if not resume_docs or not jd_docs:
         print("Warning: Missing resume or JD documents for similarity calculation.")
      
         return SimilarityResult(overall_score=float('inf'), similarity_percentage=0.0, relevant_chunks=[])

    try:
      
        start_time = time.time()
        print(f"Creating FAISS vector store from {len(resume_docs)} resume chunks...")
       
        vector_store: FAISS = FAISS.from_documents(resume_docs, embeddings)
        vs_creation_time = time.time()
        print(f"Vector store created in {vs_creation_time - start_time:.2f}s.")

      
        jd_full_text: str = "\n".join([doc.page_content for doc in jd_docs])
        if not jd_full_text.strip():
          
             raise ValueError("Job description text content is empty.")

        print(f"Performing similarity search with JD text ({len(jd_full_text)} chars)...")
       
        results_with_scores: List[Tuple[Document, float]] = vector_store.similarity_search_with_score(jd_full_text, k=5)
        search_time = time.time()
        print(f"Similarity search completed in {search_time - vs_creation_time:.2f}s. Found {len(results_with_scores)} results.")

        relevant_chunks: List[RelevantChunk] = []
        total_score: float = 0.0
        for doc, score in results_with_scores:
           
            relevant_chunks.append(RelevantChunk(
                content=doc.page_content,
                
                page=doc.metadata.get('page'),
               
                score=float(score)
            ))
            total_score += score

      
        if results_with_scores:
            
            avg_score: float = float(total_score / len(results_with_scores))
           
            similarity_percentage: float = max(0.0, 100.0 * (1.0 - avg_score / 3.0)) 
        else:
  
            avg_score = float('inf') 
            similarity_percentage = 0.0

        print(f"Average Score: {avg_score:.4f}, Approx Similarity: {similarity_percentage:.2f}%")
       
        return SimilarityResult(
            overall_score=avg_score,
            similarity_percentage=similarity_percentage,
            relevant_chunks=relevant_chunks
        )

    except Exception as e:
       
        print(f"Error during similarity calculation: {e}.")
        raise ValueError(f"Similarity calculation failed: {e}.")