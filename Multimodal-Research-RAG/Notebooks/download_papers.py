import os
import json
import time
import csv
import requests
from pathlib import Path
from tqdm import tqdm


# ── Config ────────────────────────────────────────────────
PAPERS_CSV   = Path("papers_list.csv")
PDF_DIR      = Path("data/pdfs")
LOG_FILE     = Path("data/download_log.jsonl")
DELAY        = 3       # seconds between requests (arxiv rate limit = polite)
TIMEOUT      = 30      # seconds before giving up on a download
# ──────────────────────────────────────────────────────────


def setup():
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[+] PDF directory  : {PDF_DIR}")
    print(f"[+] Log file       : {LOG_FILE}")


def load_papers():
    """Read the CSV you reviewed and return list of paper dicts."""
    papers = []
    with open(PAPERS_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            papers.append({
                "arxiv_id": row["arxiv_id"].strip(),
                "title"   : row["title"].strip(),
                "category": row["category"].strip(),
                "year"    : row["year"].strip(),
            })

    print(f"[+] Papers to download : {len(papers)}")

    # show category breakdown
    from collections import Counter
    cats = Counter(p["category"] for p in papers)
    print("\n    Category breakdown:")
    for cat, count in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"      {cat:<35} {count}")

    return papers


def already_downloaded(arxiv_id: str) -> bool:
    """Check if PDF already exists — skip re-downloading."""
    pdf_path = PDF_DIR / f"{arxiv_id.replace('/', '_')}.pdf"
    return pdf_path.exists()


def download_paper(paper: dict) -> dict:
    """
    Download one paper from arxiv.

    arxiv PDF URL format:
        https://arxiv.org/pdf/{arxiv_id}.pdf

    Returns a log record with status.
    """
    arxiv_id = paper["arxiv_id"]
    pdf_name = f"{arxiv_id.replace('/', '_')}.pdf"
    pdf_path = PDF_DIR / pdf_name
    url      = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

    log = {
        "arxiv_id": arxiv_id,
        "title"   : paper["title"],
        "category": paper["category"],
        "year"    : paper["year"],
        "pdf_path": str(pdf_path),
        "url"     : url,
        "status"  : None,
        "size_kb" : None,
        "error"   : None,
    }

    try:
        response = requests.get(
            url,
            timeout=TIMEOUT,
            headers={"User-Agent": "Mozilla/5.0 (research project)"},
            stream=True
        )

        if response.status_code == 200:
            # check it's actually a PDF
            content_type = response.headers.get("Content-Type", "")
            if "pdf" not in content_type.lower() and "octet-stream" not in content_type.lower():
                log["status"] = "failed"
                log["error"]  = f"unexpected content type: {content_type}"
                return log

            # save to disk
            with open(pdf_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            size_kb = pdf_path.stat().st_size / 1024
            log["status"]  = "success"
            log["size_kb"] = round(size_kb, 1)

        elif response.status_code == 403:
            log["status"] = "failed"
            log["error"]  = "403 forbidden — paper may be restricted"

        elif response.status_code == 404:
            log["status"] = "failed"
            log["error"]  = "404 not found — check arxiv ID"

        else:
            log["status"] = "failed"
            log["error"]  = f"HTTP {response.status_code}"

    except requests.Timeout:
        log["status"] = "failed"
        log["error"]  = "timeout"

    except Exception as e:
        log["status"] = "failed"
        log["error"]  = str(e)

    return log


def download_all(papers: list):
    """Download all papers with rate limiting and progress tracking."""

    # load existing log to skip already-done
    existing = set()
    if LOG_FILE.exists():
        with open(LOG_FILE) as f:
            for line in f:
                rec = json.loads(line)
                if rec["status"] == "success":
                    existing.add(rec["arxiv_id"])

    to_download = [p for p in papers if p["arxiv_id"] not in existing]
    already     = len(papers) - len(to_download)

    print(f"\n[+] Already downloaded : {already}")
    print(f"[+] To download now    : {len(to_download)}")

    if not to_download:
        print("    Nothing to do!")
        return

    success = 0
    failed  = 0

    log_handle = open(LOG_FILE, "a")

    for paper in tqdm(to_download, desc="  Downloading"):
        log = download_paper(paper)

        # write log line
        log_handle.write(json.dumps(log) + "\n")
        log_handle.flush()

        if log["status"] == "success":
            success += 1
        else:
            failed += 1
            tqdm.write(f"  FAILED {paper['arxiv_id']}: {log['error']}")

        # polite delay — arxiv requests this
        time.sleep(DELAY)

    log_handle.close()

    print(f"\n[+] Download complete!")
    print(f"    Success : {success + already}")
    print(f"    Failed  : {failed}")
    print(f"    Log     : {LOG_FILE}")


def print_summary():
    """Print final summary from log file."""
    if not LOG_FILE.exists():
        return

    records = []
    with open(LOG_FILE) as f:
        for line in f:
            records.append(json.loads(line))

    success = [r for r in records if r["status"] == "success"]
    failed  = [r for r in records if r["status"] == "failed"]

    total_mb = sum(r["size_kb"] or 0 for r in success) / 1024

    print(f"\n── Final Summary ───────────────────────────────")
    print(f"  Downloaded   : {len(success)} PDFs")
    print(f"  Failed       : {len(failed)}")
    print(f"  Total size   : {total_mb:.1f} MB")
    print(f"  Location     : {PDF_DIR}/")

    if failed:
        print(f"\n  Failed papers (check arxiv IDs):")
        for r in failed[:10]:
            print(f"    {r['arxiv_id']} — {r['error']}")
        if len(failed) > 10:
            print(f"    ... and {len(failed)-10} more (see {LOG_FILE})")

    print(f"""
╔══════════════════════════════════════════════════════════╗
║  Download complete!                                      ║
║                                                          ║
║  Next step:                                              ║
║  Run extract_pdf_content.py to extract text +            ║
║  images + captions from each PDF                        ║
╚══════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    setup()
    papers = load_papers()
    download_all(papers)
    print_summary()
