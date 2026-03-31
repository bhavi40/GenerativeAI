# AI-Powered Job Application Tracker Agent

An intelligent email agent that automatically reads your Gmail inbox, classifies job-related emails using Claude LLM, extracts structured data, and stores everything in PostgreSQL. Ask questions about your applications in plain English using a Text-to-SQL conversational interface exposed as a REST API.

---

## Demo

```
You: which companies rejected me?
Agent: You have 3 rejections:
       • Caterpillar Inc — Software Engineer 1
       • Google — Data Engineer
       • Amazon — Backend Engineer

You: any interviews scheduled?
Agent: Yes! You have one active interview:
       • IHS Consultants — Fullstack Developer (screening via Teams)
```

---

## Architecture

```
Gmail Inbox (OAuth 2.0)
        ↓
gmail_client.py — thread-based email fetching
        ↓
extractor.py — recursive HTML/MIME parsing
        ↓
agent.py — Claude LLM classification + extraction
        ↓
db.py — PostgreSQL with deduplication
        ↓
chat.py — Text-to-SQL natural language interface
        ↓
api.py — FastAPI REST API microservice
```


---

## Project Structure

```
email-agent/
├── agent.py            # Claude LLM email classification and data extraction
├── gmail_client.py     # Gmail API connection, thread-based email fetching
├── extractor.py        # Recursive HTML/MIME parsing, body text extraction
├── db.py               # PostgreSQL operations, deduplication, status priority
├── main.py             # Full pipeline orchestration with structured logging
├── chat.py             # Text-to-SQL chatbot powered by Claude LLM
├── api.py              # FastAPI REST API microservice
├── config.py           # Configuration and environment variables
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variable template
└── README.md
```

---

## Key Engineering Decisions

### 1. Thread-based deduplication
Instead of processing individual emails, emails are grouped by Gmail thread ID. Each conversation thread = one record. Prevents duplicate entries from email reply chains.

### 2. Dynamic start date
Instead of a hardcoded date, the agent uses MAX(processed_at) from PostgreSQL as the query start time. Each run fetches only emails since the last run — efficient and cost effective.

### 3. Status priority system
Once a higher-priority status is recorded it cannot be overwritten by a lower one. Rejection has the highest priority and can never be downgraded.

```python
STATUS_PRIORITY = {
    "offer": 6,
    "interviewing": 5,
    "interested": 4,
    "recruiter_outreach": 3,
    "applied": 2,
    "ghosted": 1,
    "rejected": 7  # highest — never overwritten
}
```

### 4. ATS platform recognition
Companies use Workday, Greenhouse, iCIMS, Lever to send emails. The agent ignores sender domains and extracts the real company name from email body and signature.

### 5. Recursive MIME parsing
Modern HTML emails use deeply nested multipart structures. Built a recursive walk_parts() function that extracts body content regardless of nesting depth.

### 6. Rate limit handling
Exponential backoff on API rate limits — waits 60s, 120s, 180s on consecutive failures. Emails that fail all retries are NOT saved to DB and are retried on the next run.

### 7. Text-to-SQL chatbot
Instead of context stuffing, uses the Text-to-SQL pattern:
```
Question → Claude generates SQL → PostgreSQL executes → Claude formats answer
```
Same pattern used by Retool AI, Metabase AI, and AWS Q for internal tools.

---

## Setup

### Prerequisites
- Python 3.11+
- Docker Desktop
- Google Cloud Project with Gmail API enabled
- Anthropic API key (console.anthropic.com)

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

Edit .env:
```
ANTHROPIC_API_KEY=sk-ant-your-key-here
DB_HOST=localhost
DB_PORT=5432
DB_NAME=jobtracker
DB_USER=postgres
DB_PASSWORD=your-password
```

### 5. Set up Gmail OAuth
- Go to console.cloud.google.com
- Create project and enable Gmail API
- OAuth consent screen → External → Add your Gmail as test user
- Credentials → Create OAuth 2.0 Client ID → Desktop app
- Download JSON → rename to credentials.json → place in project root

### 6. Start PostgreSQL with Docker
```bash
docker run --name job-tracker-db \
  -e POSTGRES_PASSWORD=your-password \
  -e POSTGRES_DB=jobtracker \
  -e POSTGRES_USER=postgres \
  -p 5432:5432 \
  -d postgres:15
```

### 7. Run the agent
```bash
python main.py
```

On first run, a browser window opens for Gmail authentication. After login, token.json is saved automatically for subsequent runs.

### 8. Run the API
```bash
uvicorn api:app --reload --port 8000
```

Open http://localhost:8000/docs for interactive API documentation.

---

## Future Improvements

- Deploy agent to AWS Lambda with EventBridge trigger every 2 hours
- Migrate PostgreSQL to AWS RDS for managed cloud database
- Add CI/CD pipeline with GitHub Actions
- Build React chat UI consuming the FastAPI endpoints
- Add LLM-as-judge evaluation framework for agent accuracy testing
- Add email reply automation to auto-send acknowledgments to recruiters
- Integrate LinkedIn API to cross-reference applications
- Migrate to Databricks Delta Lake for analytics at scale

---

## License

MIT License — free to use and modify.

---

## Author

Built as a personal productivity tool to automate job search tracking using AI agents, modern data engineering practices, and REST API design.
