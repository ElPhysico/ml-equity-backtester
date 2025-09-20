# src/mlbt/api_key.py
import os
from dotenv import load_dotenv


def get_api_key() -> str:
    """Load Alpha Vantage API key from .env file."""
    load_dotenv()
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        raise ValueError("Alpha Vantage API key not set. Please set ALPHA_VANTAGE_API_KEY in your .env file.")
    return api_key