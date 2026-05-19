import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    BACKEND_URL = os.getenv("BACKEND_URL")

    # SMTP email delivery (optional — required only for the email feature)
    SMTP_HOST = os.getenv("SMTP_HOST", "")
    SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
    SMTP_USER = os.getenv("SMTP_USER", "")
    SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
    SMTP_USE_TLS = os.getenv("SMTP_USE_TLS", "true").lower() == "true"

    @classmethod
    def smtp_configured(cls) -> bool:
        return bool(cls.SMTP_HOST and cls.SMTP_USER and cls.SMTP_PASSWORD)
