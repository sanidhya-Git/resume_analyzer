

from pydantic import BaseModel, Field
from typing import List, Optional



class ResumeAnalysis(BaseModel):
    """
    Defines the structured output expected from the AI resume analysis.
    Each field corresponds to a specific piece of information extracted or generated.
    """

    summary: str = Field(description="A concise summary of the candidate's profile.")


    strengths: List[str] = Field(description="Bullet points highlighting key strengths of the candidate and their uploaded resume.")


    weaknesses: List[str] = Field(description="Bullet points highlighting potential weaknesses or areas missing info. in the candidate's resume. Also consider differences and gaps in requirements like location match if strictly required in the job description.")

    skills: List[str] = Field(description="Key skills extracted from the resume.")


class RelevantChunk(BaseModel):
    """
    Represents a single chunk of text from the resume deemed relevant
    to the job description during similarity search, along with its metadata.
    """

    content: str

    page: Optional[int] = None


    score: float

class SimilarityResult(BaseModel):
    """
    Encapsulates the results of the resume vs. job description similarity analysis.
    """

    overall_score: float


    similarity_percentage: float


    relevant_chunks: List[RelevantChunk] 



class AnalysisAndSimilarityResponse(BaseModel):
    """
    Defines the structure of the final JSON response returned by the /analyze/ API endpoint.
    It combines the results of the resume analysis and the similarity comparison.
    """

    resume_analysis: ResumeAnalysis


    similarity_results: SimilarityResult


    processing_time_seconds: Optional[float] = None