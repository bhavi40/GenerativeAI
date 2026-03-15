import json
import re
import time
from google import genai
from google.genai import errors
from config import GEMINI_API_KEY, GEMINI_MODEL

client = genai.Client(api_key=GEMINI_API_KEY)

NOT_JOB_SUBJECTS = [
    "sale", "off", "discount", "deal", "offer", "save",
    "shop", "order", "shipped", "delivery", "tracking",
    "receipt", "payment", "bill", "statement",
    "newsletter", "unsubscribe", "subscription",
    "points", "reward", "cashback", "coupon",
    "job alert", "job digest", "jobs for you",
    "top jobs", "recommended jobs", "job matches",
    "new jobs", "hiring near", "jobs in",
    "linkedin news", "linkedin digest",
    "glassdoor", "indeed digest", "ziprecruiter",
    "welcome to", "verify your", "confirm your email",
    "password", "security alert", "sign in",
    "follow request", "liked your", "commented on",
    "get 5x", "earn points", "free shipping",
    "more jobs", "more job", "and more",
    "jobs in", "hiring for", "is hiring",
    "stand out", "automated systems",
    "volunteer", "forecast", "rain gear",
    "kohl", "mango", "shein", "clinique",
    "hdfc", "irctc", "axis bank", "tax",
    "costco", "krispy", "pink", "ulta",
]

JOB_SUBJECTS = [
    "application", "applied", "applicant",
    "interview", "interviewing", "schedule",
    "offer letter", "position", "role",
    "recruiting", "recruiter", "opportunity",
    "thank you for applying", "we received your",
    "next steps", "moving forward", "not moving forward",
    "unfortunately", "regret", "decision",
    "hiring", "candidate", "assessment",
    "your resume", "your profile",
    "background check", "onboarding",
    "your application to",
    "your application at",
]

def is_job_related_subject(subject):
    subject_lower = subject.lower()

    for keyword in NOT_JOB_SUBJECTS:
        if keyword in subject_lower:
            return False

    for keyword in JOB_SUBJECTS:
        if keyword in subject_lower:
            return True

    return None

def analyze_email(email):
    subject = email.get("subject", "")
    body = email.get("body", "")

    subject_check = is_job_related_subject(subject)

    if subject_check is False:
        print(f"  Filtered by subject — not job related")
        return {"is_job_related": False, "gemini_called": False}

    truncated_body = body[:3000]

    prompt = f"""
You are a job application tracker assistant. Analyze this email carefully.

SUBJECT: {subject}
BODY:
{truncated_body}

ONLY mark as job-related if the email is a DIRECT interaction between you and a specific company:
- Application confirmation from a specific company
- Recruiter reaching out directly about a specific role
- Interview invitation from a specific company
- Rejection email from a specific company
- Job offer from a specific company
- Company showed interest but put you on hold or waitlist

NOT job-related:
- LinkedIn/Glassdoor/Indeed job digest or alert emails
- Any email listing multiple jobs
- Shopping, promotions, bills, newsletters

REJECTION EMAIL DETECTION — mark as rejected if email contains:
- "we are not moving forward"
- "we have decided to move forward with other candidates"
- "after careful consideration"
- "we have decided not to proceed"
- "your application was not selected"
- "unfortunately we are unable to offer you"
- "we regret to inform you"
- "we will keep your resume on file"
- "thank you for your interest but"
- Any polite language meaning NO or not selected

Respond ONLY with a JSON object, no explanation, no markdown, no backticks.

If NOT job related:
{{"is_job_related": false}}

If IS job related:
{{
    "is_job_related": true,
    "company": "Exact company name",
    "role": "Exact job title",
    "status": "one of: applied / interviewing / rejected / offer / recruiter_outreach / interested / ghosted",
    "location": "City State or Remote or null",
    "applied_date": "date from email or null",
    "notes": "one sentence summary",
    "confidence": 0.95
}}

Status guide:
- applied = company confirmed they received your application
- interviewing = interview invite or scheduling email
- rejected = company said no in any way direct or polite
- offer = company made you an offer
- recruiter_outreach = recruiter contacted you about a specific role
- interested = company showed interest but put you on hold or waitlist
- ghosted = you followed up but got no response
"""

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt
            )
            raw = response.text.strip()
            raw = re.sub(r"```json|```", "", raw).strip()
            result = json.loads(raw)
            result["gemini_called"] = True
            return result

        except errors.ClientError as e:
            if "429" in str(e):
                wait = 30 * (attempt + 1)
                print(f"  Rate limit hit — waiting {wait} seconds...")
                time.sleep(wait)
            else:
                print(f"  Gemini error: {e}")
                return {"is_job_related": False, "gemini_called": True}

        except json.JSONDecodeError:
            print(f"  Could not parse response — skipping")
            return {"is_job_related": False, "gemini_called": True}

    return {"is_job_related": False, "gemini_called": True}