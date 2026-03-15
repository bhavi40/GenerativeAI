import re

def clean_html(text):
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'&nbsp;', ' ', text)
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'&lt;', '<', text)
    text = re.sub(r'&gt;', '>', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_email_content(email):
    body = email["body"]
    if "<html" in body.lower() or "<div" in body.lower():
        body = clean_html(body)

    full_text = f"SUBJECT: {email['subject']}\n"
    full_text += f"FROM: {email['sender']}\n"
    full_text += f"DATE: {email['date']}\n\n"
    full_text += f"BODY:\n{body}\n"

    return full_text