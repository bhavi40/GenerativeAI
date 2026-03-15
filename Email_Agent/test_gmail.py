from gmail_client import get_unread_emails

emails = get_unread_emails()

print(f"Found {len(emails)} emails after 2026/01/01\n")
for email in emails[:5]:  # show first 5 only
    print(f"From:    {email['sender']}")
    print(f"Subject: {email['subject']}")
    print(f"Date:    {email['date']}")
    print("-" * 50)