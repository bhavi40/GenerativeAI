from gmail_client import get_unread_emails
from extractor import extract_email_content
from agent import analyze_email

emails = get_unread_emails()

print(f"Testing agent on first 5 emails...\n")

for email in emails[:5]:
    content = extract_email_content(email)
    result = analyze_email(content)

    print(f"Subject: {email['subject']}")
    print(f"Result:  {result}")
    print("-" * 60)