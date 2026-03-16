
# Job Application Tracker Agent

An AI-powered email agent that automatically reads your Gmail inbox, classifies job-related emails using Google Gemini LLM, extracts structured data, and stores everything in PostgreSQL — so you always know where you stand in your job search.

---

## What it does

- Connects to Gmail via OAuth 2.0 and fetches emails from a configured start date
- Applies a two-stage filter — rule-based subject filtering first, then Gemini LLM for uncertain emails — reducing API calls by ~80%
- Classifies emails into 7 statuses: `applied`, `interviewing`, `rejected`, `offer`, `recruiter_outreach`, `interested`, `ghosted`
- Detects rejection emails using semantic pattern matching (e.g. "we are not moving forward", "after careful consideration")
- Extracts company name, role, location, applied date, and a one-sentence summary
- Saves everything to PostgreSQL with deduplication — never processes the same email twice
- Runs automatically every 2 hours via APScheduler
- Includes a Streamlit dashboard with stats, application table, bar charts, and an AI chat interface
- Deployed on AWS (Lambda + RDS + S3 + EventBridge + CloudWatch)

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| AI / LLM | Google Gemini 2.5 Flash Lite |
| Email | Gmail API (OAuth 2.0) |
| Database (local) | PostgreSQL (Docker) |
| Database (cloud) | AWS RDS PostgreSQL |
| Storage | AWS S3 |
| Compute | AWS Lambda |
| Scheduling | AWS EventBridge / APScheduler |
| Monitoring | AWS CloudWatch |
| Dashboard | Streamlit |
| Containerization | Docker, Docker Compose |

---

## Project Structure

```
email-agent/
├── agent.py            # Two-stage email classification using Gemini LLM
├── gmail_client.py     # Gmail API connection and email fetching
├── extractor.py        # Email body HTML cleaning and text extraction
├── db.py               # PostgreSQL connection, table creation, queries
├── main.py             # Main pipeline — fetch, classify, store
├── scheduler.py        # APScheduler — runs every 2 hours automatically
├── dashboard.py        # Streamlit dashboard with AI chat interface
├── chat.py             # Natural language query interface using Gemini
├── config.py           # Configuration and environment variables
├── Dockerfile          # Container definition
├── docker-compose.yml  # Multi-container setup (agent + db + dashboard)
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variable template
└── README.md
```

---

## How It Works

```
Gmail Inbox (after 2026/01/01)
        │
        ▼
Stage 1 — Subject Filter (no API call)
"50% off sale" → skip instantly
        │
        ▼ (uncertain or job-likely subjects only)
Stage 2 — Gemini LLM Classification
Extracts: company, role, status, location, date
        │
        ▼
PostgreSQL (processed_emails + job_applications)
        │
        ▼
Streamlit Dashboard + AI Chat
"Which companies rejected me?" → Gemini answers
```

---

## Database Schema

**processed_emails**
| Column | Type | Description |
|---|---|---|
| id | serial | Primary key |
| email_id | varchar | Gmail message ID (unique) |
| subject | text | Email subject |
| sender | varchar | Sender email address |
| email_date | varchar | Date of email |
| is_job_related | boolean | True if job related |
| processed_at | timestamp | When processed |

**job_applications**
| Column | Type | Description |
|---|---|---|
| id | serial | Primary key |
| email_id | varchar | Foreign key to processed_emails |
| company | varchar | Company name |
| role | varchar | Job title |
| status | varchar | applied / interviewing / rejected / offer / etc |
| location | varchar | City, State or Remote |
| applied_date | varchar | Date from email |
| notes | text | One sentence summary |
| confidence | float | Gemini confidence score |
| created_at | timestamp | When inserted |

---

## Setup

### Prerequisites
- Python 3.11+
- Docker Desktop
- Google Cloud Project with Gmail API enabled
- Google AI Studio API key (Gemini)
- AWS account (for production deployment)

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/email-agent.git
cd email-agent
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
```bash
cp .env.example .env
```

Edit `.env` with your credentials:
```
GEMINI_API_KEY=your-gemini-api-key
DB_HOST=localhost
DB_PORT=5432
DB_NAME=jobtracker
DB_USER=postgres
DB_PASSWORD=your-password
AWS_ACCESS_KEY_ID=your-aws-key
AWS_SECRET_ACCESS_KEY=your-aws-secret
AWS_REGION=us-east-1
```

### 5. Set up Gmail OAuth
- Go to [Google Cloud Console](https://console.cloud.google.com)
- Create a project → Enable Gmail API
- Create OAuth 2.0 credentials → Desktop app
- Download `credentials.json` → place in project root

### 6. Run locally
```bash
python main.py
```

On first run, a browser window opens for Gmail authentication. After login, `token.json` is saved automatically — no login needed on subsequent runs.

### 7. Run with Docker Compose
```bash
docker-compose up --build
```

This starts:
- `agent` container — runs scheduler every 2 hours
- `db` container — PostgreSQL database
- `dashboard` container — Streamlit at `http://localhost:8501`

---

## Streamlit Dashboard

```bash
streamlit run dashboard.py
```

Open `http://localhost:8501` in your browser.

Features:
- Summary metrics — total, applied, interviewing, rejected, offers
- Applications table with status color coding and filtering
- Bar chart by application status
- AI chat interface — ask questions in plain English

Example questions:
```
"Which companies rejected me?"
"Did I apply to Google?"
"Which companies showed interest?"
"How many applications do I have this month?"
"Which roles am I interviewing for?"
```

---

## AWS Production Deployment

### Architecture
```
EventBridge (every 2 hours)
    → AWS Lambda (Python agent)
        → AWS RDS PostgreSQL (data storage)
        → AWS S3 (raw email backups)
        → CloudWatch (logs + monitoring)
Streamlit on EC2 (dashboard — always on)
```

### Steps
1. Create RDS PostgreSQL instance (db.t3.micro — free tier)
2. Migrate local DB to RDS: `pg_dump` → upload → `psql` restore
3. Package Lambda function with dependencies
4. Create EventBridge rule — schedule every 2 hours
5. Create S3 bucket for email backups
6. Set up CloudWatch log group for monitoring
7. Launch EC2 t2.micro → deploy Streamlit dashboard

---

## Email Classification Logic

The agent uses a two-stage approach:

**Stage 1 — Subject keyword filter (instant, no API call)**
- Blocklist: sale, discount, job alert, job digest, newsletter, shipping, etc.
- Allowlist: application, interview, offer, recruiter, rejected, etc.
- Unknown subjects proceed to Stage 2

**Stage 2 — Gemini LLM classification**
- Sends subject + first 3000 chars of body to Gemini
- Returns structured JSON with company, role, status, confidence
- Handles rate limits with exponential backoff (30s, 60s, 90s retries)

**Rejection detection phrases:**
- "we are not moving forward"
- "after careful consideration"
- "we have decided to pursue other candidates"
- "we regret to inform you"
- "we will keep your resume on file"
- and more...

---

## Environment Variables

| Variable | Description |
|---|---|
| `GEMINI_API_KEY` | Google AI Studio API key |
| `DB_HOST` | PostgreSQL host |
| `DB_PORT` | PostgreSQL port (default 5432) |
| `DB_NAME` | Database name |
| `DB_USER` | Database user |
| `DB_PASSWORD` | Database password |
| `AWS_ACCESS_KEY_ID` | AWS access key |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key |
| `AWS_REGION` | AWS region (e.g. us-east-1) |

---

## Future Improvements

- Add email reply automation — auto-send acknowledgment to recruiters
- Integrate with LinkedIn API to cross-reference applications
- Add weekly summary email report
- Migrate to Databricks Delta Lake for large-scale analytics
- Add Power BI dashboard for advanced visualizations
- Support Outlook / IMAP in addition to Gmail

---

## License

MIT License — free to use and modify.

---

## Author

Built as a personal productivity tool to automate job search tracking using AI agents, cloud infrastructure, and modern data engineering practices.
