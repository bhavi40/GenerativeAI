import sys
import time
import logging
from gmail_client import get_unread_emails
from agent import analyze_email
from db import (create_tables, is_thread_processed,
                save_processed_email, save_job_application)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("agent.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
sys.stdout.reconfigure(encoding="utf-8")
logger = logging.getLogger(__name__)

def run():
    logger.info("Starting Job Tracker Agent...")
    logger.info("=" * 60)

    create_tables()

    logger.info("Fetching emails from Gmail...")
    emails = get_unread_emails()
    logger.info(f"Found {len(emails)} threads since 2026/03/19")

    total = 0
    skipped = 0
    job_found = 0
    not_job = 0
    failed = 0

    for i, email in enumerate(emails):
        logger.info(f"Processing {i+1}/{len(emails)}: {email['subject'][:50]}")

        # Skip if thread already processed
        thread_id = email.get("thread_id")
        if thread_id and is_thread_processed(thread_id):
            skipped += 1
            logger.info(f"  Thread already processed — skipping")
            continue


        result = analyze_email(email)

        if result.get("rate_limit_failed", False):
            failed += 1
            logger.warning(f"  Claude failed — will retry next run")
            continue

        is_job = result.get("is_job_related", False)

        save_processed_email(email, is_job)

        if is_job:
            save_job_application(email["id"], result, thread_id)
            job_found += 1
            logger.info(f"  JOB FOUND — {result.get('company')} | {result.get('role')} | {result.get('status')}")
        else:
            not_job += 1
            logger.info(f"  Not job related — skipped")

        total += 1

        if result.get("llm_called", False):
            time.sleep(15)

    logger.info("=" * 60)
    logger.info(f"Run complete!")
    logger.info(f"  Processed  : {total}")
    logger.info(f"  Skipped    : {skipped} (already in DB)")
    logger.info(f"  Jobs found : {job_found}")
    logger.info(f"  Not job    : {not_job}")
    logger.info(f"  Failed     : {failed} (will retry next run)")

if __name__ == "__main__":
    run()
