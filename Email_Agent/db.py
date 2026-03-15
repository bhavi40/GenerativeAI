import psycopg2
from psycopg2.extras import RealDictCursor
from config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD

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
    print("Tables created successfully.")

def is_email_processed(email_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT id FROM processed_emails WHERE email_id = %s",
        (email_id,)
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
        (email_id, subject, sender, email_date, is_job_related)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (email_id) DO NOTHING
    """, (
        email["id"],
        email["subject"],
        email["sender"],
        email["date"],
        bool(is_job_related)
    ))
    conn.commit()
    cur.close()
    conn.close()

def save_job_application(email_id, job_data):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO job_applications
        (email_id, company, role, status, location, applied_date, notes, confidence)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        email_id,
        job_data.get("company"),
        job_data.get("role"),
        job_data.get("status"),
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
            COUNT(CASE WHEN status = 'ghosted' THEN 1 END) as ghosted
        FROM job_applications
    """)
    result = cur.fetchone()
    cur.close()
    conn.close()
    return result