import os
import base64
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from config import SCOPES, CREDENTIALS_FILE, START_DATE, TOKEN_FILE, MAX_EMAILS_PER_RUN


def get_gmail_service():
    creds = None
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_FILE, "w") as token:
            token.write(creds.to_json())
    return build("gmail", "v1", credentials=creds)

def get_unread_emails():
    service = get_gmail_service()
    results = service.users().messages().list(
        userId="me",
        labelIds=["INBOX"],
        q=f"after:{START_DATE}",
        maxResults=MAX_EMAILS_PER_RUN
    ).execute()

    messages = results.get("messages", [])
    emails = []

    for msg in messages:
        msg_data = service.users().messages().get(
            userId="me",
            id=msg["id"],
            format="full"
        ).execute()

        email = parse_email(msg_data)
        emails.append(email)

    return emails

def parse_email(msg_data):
    headers = msg_data["payload"]["headers"]
    subject = next((h["value"] for h in headers if h["name"] == "Subject"), "No Subject")
    sender = next((h["value"] for h in headers if h["name"] == "From"), "Unknown")
    date = next((h["value"] for h in headers if h["name"] == "Date"), "Unknown")

    body = extract_body(msg_data["payload"])
    attachments = extract_attachments(msg_data["payload"])

    return {
        "id": msg_data["id"],
        "subject": subject,
        "sender": sender,
        "date": date,
        "body": body,
        "attachments": attachments
    }

def extract_body(payload):
    body = ""
    if "parts" in payload:
        for part in payload["parts"]:
            if part["mimeType"] == "text/plain":
                data = part["body"].get("data", "")
                if data:
                    body += base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")
            elif part["mimeType"] == "text/html" and not body:
                data = part["body"].get("data", "")
                if data:
                    body += base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")
    else:
        data = payload["body"].get("data", "")
        if data:
            body = base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")
    return body.strip()

def extract_attachments(payload):
    attachments = []
    if "parts" in payload:
        for part in payload["parts"]:
            if part.get("filename") and part["filename"] != "":
                attachments.append({
                    "filename": part["filename"],
                    "mimeType": part["mimeType"],
                    "attachmentId": part["body"].get("attachmentId", ""),
                    "size": part["body"].get("size", 0)
                })
    return attachments

def mark_as_read(service, email_id):
    service.users().messages().modify(
        userId="me",
        id=email_id,
        body={"removeLabelIds": ["UNREAD"]}
    ).execute()