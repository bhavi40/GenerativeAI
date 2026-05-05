import psycopg2
from psycopg2.extras import RealDictCursor
from config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD, START_DATE

STATUS_PRIORITY = {
    "offer": 6,
    "interviewing": 5,
    "interested": 4,
    "recruiter_outreach": 3,
    "applied": 2,
    "ghosted": 1,
    "rejected": 7
}

def get_connection():
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )

def create_tables():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS processed_emails (
            id SERIAL PRIMARY KEY,
            email_id VARCHAR(255) UNIQUE NOT NULL,
            thread_id VARCHAR(255),
            subject TEXT,
            sender VARCHAR(500),
            email_date VARCHAR(255),
            is_job_related BOOLEAN DEFAULT FALSE,
            processed_at TIMESTAMP DEFAULT NOW()
        );
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS job_applications (
            id SERIAL PRIMARY KEY,
            email_id VARCHAR(255) REFERENCES processed_emails(email_id),
            thread_id VARCHAR(255),
            company VARCHAR(255),
            role VARCHAR(255),
            status VARCHAR(100),
            location VARCHAR(255),
            applied_date VARCHAR(255),
            notes TEXT,
            confidence FLOAT,
            created_at TIMESTAMP DEFAULT NOW()
        );
    """)
    conn.commit()
    cur.close()
    conn.close()
    print("Tables ready.")

def is_thread_processed(thread_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT id FROM processed_emails WHERE thread_id = %s",
        (thread_id,)
    )
    result = cur.fetchone()
    cur.close()
    conn.close()
    return result is not None

def save_processed_email(email, is_job_related):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO processed_emails
        (email_id, thread_id, subject, sender, email_date, is_job_related)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (email_id) DO NOTHING
    """, (
        email["id"],
        email.get("thread_id"),
        email["subject"],
        email["sender"],
        email["date"],
        is_job_related is True
    ))
    conn.commit()
    cur.close()
    conn.close()

def get_existing_application(company, role):
    conn = get_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("""
        SELECT id, status FROM job_applications
        WHERE LOWER(company) = LOWER(%s)
        AND LOWER(role) = LOWER(%s)
    """, (company, role))
    result = cur.fetchone()
    cur.close()
    conn.close()
    return result

def save_job_application(email_id, job_data, thread_id=None):
    company = job_data.get("company")
    role = job_data.get("role")
    new_status = job_data.get("status")

    existing = get_existing_application(company, role)

    conn = get_connection()
    cur = conn.cursor()

    if existing:
        existing_priority = STATUS_PRIORITY.get(existing["status"], 0)
        new_priority = STATUS_PRIORITY.get(new_status, 0)

        if new_priority > existing_priority:
            cur.execute("""
                UPDATE job_applications
                SET status = %s,
                    email_id = %s,
                    thread_id = %s,
                    notes = %s,
                    confidence = %s
                WHERE id = %s
            """, (
                new_status,
                email_id,
                thread_id,
                job_data.get("notes"),
                job_data.get("confidence"),
                existing["id"]
            ))
            print(f"  Status updated: {existing['status']} → {new_status}")
        else:
            print(f"  Kept status: {existing['status']} (priority higher than {new_status})")
    else:
        cur.execute("""
            INSERT INTO job_applications
            (email_id, thread_id, company, role, status, location,
             applied_date, notes, confidence)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            email_id,
            thread_id,
            company,
            role,
            new_status,
            job_data.get("location"),
            job_data.get("applied_date"),
            job_data.get("notes"),
            job_data.get("confidence")
        ))

    conn.commit()
    cur.close()
    conn.close()

def get_all_applications():
    conn = get_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("""
        SELECT
            ja.company,
            ja.role,
            ja.status,
            ja.location,
            ja.applied_date,
            ja.notes,
            ja.confidence,
            pe.subject,
            pe.email_date
        FROM job_applications ja
        JOIN processed_emails pe ON ja.email_id = pe.email_id
        ORDER BY ja.created_at DESC
    """)
    results = cur.fetchall()
    cur.close()
    conn.close()
    return results

def get_stats():
    conn = get_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("""
        SELECT
            COUNT(*) as total,
            COUNT(CASE WHEN status = 'applied' THEN 1 END) as applied,
            COUNT(CASE WHEN status = 'interviewing' THEN 1 END) as interviewing,
            COUNT(CASE WHEN status = 'rejected' THEN 1 END) as rejected,
            COUNT(CASE WHEN status = 'offer' THEN 1 END) as offers,
            COUNT(CASE WHEN status = 'ghosted' THEN 1 END) as ghosted,
            COUNT(CASE WHEN status = 'interested' THEN 1 END) as interested,
            COUNT(CASE WHEN status = 'recruiter_outreach' THEN 1 END) as recruiter_outreach
        FROM job_applications
    """)
    result = cur.fetchone()
    cur.close()
    conn.close()
    return result
def get_last_processed_timestamp():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT MAX(processed_at) 
        FROM processed_emails
    """)
    result = cur.fetchone()
    cur.close()
    conn.close()
    
    if result and result[0]:
        # Format for Gmail API query — epoch timestamp
        import time
        from datetime import timezone
        ts = int(result[0].replace(tzinfo=timezone.utc).timestamp())
        return f"after:{ts}"
    else:
        # First ever run — use START_DATE
        return f"after:{START_DATE}"
