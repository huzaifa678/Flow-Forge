import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    HF_TOKEN = os.getenv("HF_TOKEN")
