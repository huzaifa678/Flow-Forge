import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    HF_TOKEN = os.getenv("HF_TOKEN")
    DATABASE_URL = os.getenv("DATABASE_URL")

class LLMConfig:
    PLAN_PROVIDER = "together"