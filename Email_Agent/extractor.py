import re

def clean_html(text):
    # Remove style tags and content
    text = re.sub(r'<style[^>]*>.*?</style>', ' ', text, flags=re.DOTALL)
    # Remove script tags and content
    text = re.sub(r'<script[^>]*>.*?</script>', ' ', text, flags=re.DOTALL)
    # Remove HTML comments
    text = re.sub(r'<!--.*?-->', ' ', text, flags=re.DOTALL)
    # Remove all HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Fix HTML entities
    text = re.sub(r'&nbsp;', ' ', text)
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'&lt;', '<', text)
    text = re.sub(r'&gt;', '>', text)
    text = re.sub(r'&quot;', '"', text)
    text = re.sub(r'&#39;', "'", text)
    text = re.sub(r'&[a-zA-Z]+;', ' ', text)
    # Remove multiple spaces and newlines
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_email_content(email):
    body = email["body"]

    # Always clean HTML regardless of content
    if body:
        body = clean_html(body)
    else:
        body = ""

    full_text = f"SUBJECT: {email['subject']}\n"
    full_text += f"FROM: {email['sender']}\n"
    full_text += f"DATE: {email['date']}\n\n"
    full_text += f"BODY:\n{body}\n"

    return full_text
