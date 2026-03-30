import time
from gmail_client import get_unread_emails
from agent import analyze_email
from db import create_tables, is_email_processed, save_processed_email, save_job_application

def run():
    print("Starting Job Tracker Agent...")
    print("=" * 60)

    create_tables()

    print("Fetching emails from Gmail...")
    emails = get_unread_emails()
    print(f"Found {len(emails)} emails since 2026/03/16\n")

    total = 0
    skipped = 0
    job_found = 0
    not_job = 0
    failed = 0

    for i, email in enumerate(emails):
        print(f"Processing {i+1}/{len(emails)}: {email['subject'][:50]}")

        if is_email_processed(email["id"]):
            skipped += 1
            print(f"  Already processed — skipping")
            continue

        result = analyze_email(email)

        # If rate limit failed all retries — don't save, retry next run
        if result.get("rate_limit_failed", False):
            failed += 1
            print(f"  Gemini failed — will retry next run")
            continue

        is_job = result.get("is_job_related", False)

        save_processed_email(email, is_job)

        if is_job:
            save_job_application(email["id"], result)
            job_found += 1
            print(f"  JOB FOUND — {result.get('company')} | {result.get('role')} | {result.get('status')}")
        else:
            not_job += 1
            print(f"  Not job related — skipped")

        total += 1

        if result.get("llm_called", False):
            time.sleep(15)

    print("\n" + "=" * 60)
    print(f"Run complete!")
    print(f"  Processed  : {total}")
    print(f"  Skipped    : {skipped} (already in DB)")
    print(f"  Jobs found : {job_found}")
    print(f"  Not job    : {not_job}")
    print(f"  Failed     : {failed} (will retry next run)")

if __name__ == "__main__":
    run()
