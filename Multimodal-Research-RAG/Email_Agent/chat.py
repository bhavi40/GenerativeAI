import anthropic
import psycopg2
from psycopg2.extras import RealDictCursor
from db import get_connection, get_stats
from config import ANTHROPIC_API_KEY, CLAUDE_MODEL

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

DB_SCHEMA = """
Table: processed_emails
- id (serial) — primary key
- email_id (varchar) — unique Gmail message ID
- subject (text) — email subject
- sender (varchar) — sender email address
- email_date (varchar) — date email was received
- is_job_related (boolean) — true if job related
- processed_at (timestamp) — when agent processed it

Table: job_applications
- id (serial) — primary key
- email_id (varchar) — foreign key to processed_emails
- company (varchar) — company name
- role (varchar) — job title
- status (varchar) — applied/interviewing/rejected/offer/recruiter_outreach/interested/ghosted
- location (varchar) — city state or remote
- applied_date (varchar) — date from email
- notes (text) — one sentence summary
- confidence (float) — LLM confidence score
- created_at (timestamp) — when inserted
"""

def generate_sql(question):
    prompt = f"""You are a SQL expert. Generate a PostgreSQL query to answer the user's question.

DATABASE SCHEMA:
{DB_SCHEMA}

RULES:
- Generate ONLY the SQL query, nothing else
- No explanation, no markdown, no backticks
- Use proper PostgreSQL syntax
- Always use lowercase for column and table names
- For case-insensitive search use ILIKE
- Limit results to 50 rows maximum unless asked for all
- Never use DROP, DELETE, UPDATE, INSERT — read only queries only

USER QUESTION: {question}

SQL QUERY:"""

    message = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text.strip()

def execute_sql(sql):
    conn = get_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    try:
        cur.execute(sql)
        results = cur.fetchall()
        return results, None
    except Exception as e:
        return None, str(e)
    finally:
        cur.close()
        conn.close()

def format_answer(question, sql, results, error):
    if error:
        prompt = f"""The user asked: "{question}"
I generated this SQL: {sql}
But got this error: {error}

Apologize briefly and suggest they rephrase the question."""
    else:
        results_text = ""
        if results:
            for row in results:
                results_text += str(dict(row)) + "\n"
        else:
            results_text = "No results found"

        prompt = f"""You are a helpful job search assistant.

The user asked: "{question}"

I ran this SQL query: {sql}

Results from database:
{results_text}

Answer the user's question in plain English based on these results.
Be concise and friendly. Use bullet points if listing multiple items.
Do not mention SQL or database in your answer."""

    message = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text.strip()

def ask(question):
    # Step 1 — Claude generates SQL
    sql = generate_sql(question)
    print(f"\n  Generated SQL: {sql}")

    # Step 2 — Execute SQL on PostgreSQL
    results, error = execute_sql(sql)

    # Step 3 — Claude formats the answer
    answer = format_answer(question, sql, results, error)
    return answer

def chat():
    print("\nJob Tracker Chat (Text-to-SQL)")
    print("=" * 40)
    print("Ask anything about your job applications!")
    print("Type 'quit' to exit\n")

    stats = get_stats()
    print(f"Current stats: {stats['total']} total | "
          f"{stats['applied']} applied | "
          f"{stats['interviewing']} interviewing | "
          f"{stats['rejected']} rejected\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        if not question:
            continue
        answer = ask(question)
        print(f"\nAssistant: {answer}\n")

if __name__ == "__main__":
    chat()
