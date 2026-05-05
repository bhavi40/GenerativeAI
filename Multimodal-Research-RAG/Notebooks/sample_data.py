

import json
import shutil
import random
from pathlib import Path
from collections import defaultdict

# ── CONFIGURE THESE TO MATCH YOUR LOCAL PATHS ────────────────
JSONL_PATH  = Path(r"C:\Users\rajes\OneDrive\Desktop\Bhavishya\Agent\pathology_rag_pipeline\Notebooks\Extraction\download_log.jsonl")
PDF_SRC_DIR = Path(r"C:\Users\rajes\OneDrive\Desktop\Bhavishya\Agent\pathology_rag_pipeline\Source_Data\Data\pdfs")
OUTPUT_DIR  = Path(r"C:\Users\rajes\OneDrive\Desktop\Bhavishya\Agent\pathology_rag_pipeline\upload_to_kaggle")

PDFS_PER_CATEGORY = 3
RANDOM_SEED       = 42

# ── Setup ─────────────────────────────────────────────────────
sample_pdf_dir = OUTPUT_DIR / "pdfs"
sample_pdf_dir.mkdir(parents=True, exist_ok=True)
random.seed(RANDOM_SEED)

# ── Load JSONL ────────────────────────────────────────────────
records = []
with open(JSONL_PATH, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if rec.get("status") == "success":
            records.append(rec)

print(f"Total successful papers: {len(records)}\n")

# ── Group by category ─────────────────────────────────────────
by_category = defaultdict(list)
for rec in records:
    by_category[rec["category"]].append(rec)

print(f"Categories found: {len(by_category)}")
for cat, papers in sorted(by_category.items()):
    print(f"  {cat:<45} → {len(papers)} papers")

# ── Pick 3 per category ───────────────────────────────────────
sample_records = []
for cat, papers in sorted(by_category.items()):
    picked = random.sample(papers, min(PDFS_PER_CATEGORY, len(papers)))
    sample_records.extend(picked)

print(f"\nSelected {len(sample_records)} PDFs "
      f"({PDFS_PER_CATEGORY} per category)\n")

# ── Copy PDFs + write sample JSONL ────────────────────────────
copied, missing = 0, 0
clean_records = []

for rec in sample_records:
    arxiv_id = rec["arxiv_id"]

    # try both dot and underscore filename formats
    src = PDF_SRC_DIR / f"{arxiv_id}.pdf"
    if not src.exists():
        src = PDF_SRC_DIR / f"{arxiv_id.replace('.', '_')}.pdf"

    if src.exists():
        dst = sample_pdf_dir / src.name
        shutil.copy2(src, dst)

        # write clean record with updated path
        clean_rec = {
            "arxiv_id" : arxiv_id,
            "title"    : rec["title"],
            "category" : rec["category"],
            "year"     : rec.get("year", ""),
            "status"   : "success",
        }
        clean_records.append(clean_rec)
        copied += 1
        print(f"  ✓  {arxiv_id:<15}  [{rec['category']}]")
    else:
        missing += 1
        print(f"  ✗  MISSING: {arxiv_id}")

# ── Write download_log.jsonl into output folder ───────────────
log_out = OUTPUT_DIR / "download_log.jsonl"
with open(log_out, "w", encoding="utf-8") as f:
    for rec in clean_records:
        f.write(json.dumps(rec) + "\n")

print(f"""
─────────────────────────────────────────
Done.
  PDFs copied : {copied}
  Missing     : {missing}
  Output dir  : {OUTPUT_DIR}

Upload this entire folder to Kaggle:
  upload_to_kaggle/
    pdfs/               ← {copied} sample PDFs
    download_log.jsonl  ← metadata for extraction notebook
─────────────────────────────────────────
""")