

import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings


load_dotenv()

class Settings(BaseSettings):
    """
    Defines application settings, loading values from environment variables or .env file.

    Attributes:
        google_api_key (str): The API key for Google Generative AI services.
                               Loaded from the 'GOOGLE_API_KEY' environment variable or .env.
        llm_model_name (str): The specific Gemini model to use for chat/language generation.
        embedding_model_name (str): The specific Google model to use for text embeddings.
    """

    google_api_key: str = os.getenv("GOOGLE_API_KEY", "MISSING_API_KEY")


    llm_model_name: str = "gemini-1.5-flash-latest"
    embedding_model_name: str = "models/embedding-001"

    # --- Pydantic BaseSettings Configuration ---
    class Config:
        """
        Inner class to configure the behavior of the BaseSettings model.
        """
   
        env_file: str = '.env'

        extra: str = 'ignore'

settings: Settings = Settings()



if settings.google_api_key == "MISSING_API_KEY" or not settings.google_api_key:

    print("\n--- WARNING ---")
    print("Configuration Issue: GOOGLE_API_KEY not found in environment variables or the .env file.")
    print("The application requires this key to interact with Google AI services and may not function correctly.")
    print("Please ensure the key is set in your .env file or environment.")
    print("---------------\n")