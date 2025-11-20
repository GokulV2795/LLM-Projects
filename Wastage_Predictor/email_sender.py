import os
import smtplib
import tempfile
from email.message import EmailMessage
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()

def send_email_smtp(subject: str, body: str, to: Optional[str] = None, attachments: Optional[List[str]] = None):
    """
    Send email using SMTP. Uses env vars:
      FROM_EMAIL, WAREHOUSE_MANAGER_EMAIL, SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD
    attachments: list of filepaths to attach
    """
    FROM = os.getenv("FROM_EMAIL")
    DEFAULT_TO = os.getenv("WAREHOUSE_MANAGER_EMAIL")
    TO = to if to else DEFAULT_TO
    if not TO:
        raise ValueError("No recipient configured — set WAREHOUSE_MANAGER_EMAIL in env or pass 'to'.")

    SMTP_HOST = os.getenv("SMTP_HOST")
    SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
    SMTP_USER = os.getenv("SMTP_USER")
    SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")

    if not SMTP_HOST or not SMTP_USER or not SMTP_PASSWORD:
        raise ValueError("SMTP settings incomplete. Set SMTP_HOST, SMTP_USER, SMTP_PASSWORD in env.")

    msg = EmailMessage()
    msg["From"] = FROM or SMTP_USER
    msg["To"] = TO
    msg["Subject"] = subject
    msg.set_content(body)

    # Attach files if provided
    if attachments:
        for fp in attachments:
            try:
                with open(fp, "rb") as f:
                    data = f.read()
                maintype = "application"
                subtype = "octet-stream"
                filename = fp.split("/")[-1]
                msg.add_attachment(data, maintype=maintype, subtype=subtype, filename=filename)
            except Exception:
                # ignore attach errors but log
                pass

    # send
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as s:
        s.starttls()
        s.login(SMTP_USER, SMTP_PASSWORD)
        s.send_message(msg)