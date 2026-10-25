"""Shared settings for the NASA mission RAG prototype.

Secrets and endpoint settings are read from environment variables, optionally
loaded from a .env file next to this module (see .env.example).
The OpenAI endpoint is configurable so the same code runs against api.openai.com
or any OpenAI-compatible proxy (set OPENAI_BASE_URL).
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

# Variables already set in the environment take precedence over .env
load_dotenv(Path(__file__).with_name(".env"))

# Unset -> api.openai.com
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL") or None

CHAT_MODELS = ["gpt-4o"]
EVAL_MODEL = os.getenv("RAG_EVAL_MODEL", "gpt-4o")
EMBEDDING_MODEL = "text-embedding-3-small"

DATA_DIR = "data_text"
CHROMA_DIR = "./chroma_db_openai"
COLLECTION_NAME = "nasa_space_missions_text"


def get_openai_client(api_key: Optional[str] = None) -> OpenAI:
    """Create an OpenAI client for the configured endpoint."""
    return OpenAI(api_key=api_key, base_url=OPENAI_BASE_URL)
