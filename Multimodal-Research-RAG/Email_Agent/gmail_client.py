import os
import base64
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from config import SCOPES, CREDENTIALS_FILE, START_DATE, TOKEN_FILE, MAX_EMAILS_PER_RUN
from db import get_last_processed_timestamp


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
    
    # Dynamic timestamp — only fetch emails since last run
    query = get_last_processed_timestamp()
    print(f"Fetching emails with query: {query}")
    
    results = service.users().threads().list(
        userId="me",
        q=query,
        maxResults=MAX_EMAILS_PER_RUN
    ).execute()

    threads = results.get("threads", [])
    emails = []

    for thread in threads:
        thread_data = service.users().threads().get(
            userId="me",
            id=thread["id"],
            format="full"
        ).execute()

        messages = thread_data.get("messages", [])
        
        if not messages:
            continue

        latest_message = messages[-1]
        email = parse_email(latest_message)
        
        first_message = messages[0]
        first_headers = first_message["payload"]["headers"]
        first_subject = next(
            (h["value"] for h in first_headers if h["name"] == "Subject"),
            ""
        )
        
        email["thread_id"] = thread["id"]
        email["thread_length"] = len(messages)
        email["first_subject"] = first_subject
        
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
    import base64
    
    body = ""
    
    def get_text_from_part(part):
        data = part.get("body", {}).get("data", "")
        if data:
            return base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")
        return ""
    
    def walk_parts(part):
        mime_type = part.get("mimeType", "")
        
        # If this part has direct body data
        if part.get("body", {}).get("data"):
            if mime_type == "text/plain":
                return get_text_from_part(part), "plain"
            elif mime_type == "text/html":
                return get_text_from_part(part), "html"
        
        # If this part has sub-parts, walk them recursively
        parts = part.get("parts", [])
        plain_text = ""
        html_text = ""
        
        for subpart in parts:
            result, type_ = walk_parts(subpart)
            if type_ == "plain" and result:
                plain_text = result
            elif type_ == "html" and result:
                html_text = result
        
        if plain_text:
            return plain_text, "plain"
        if html_text:
            return html_text, "html"
            
        return "", ""
    
    text, type_ = walk_parts(payload)
    return text

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
