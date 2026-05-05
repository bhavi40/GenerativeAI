import json
import re
import time
import anthropic
from config import ANTHROPIC_API_KEY, CLAUDE_MODEL

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

def analyze_email(email):
    subject = email.get("subject", "")
    body = email.get("body", "")
    sender = email.get("sender", "")

    truncated_body = body[:3000]

    prompt = f"""You are a job application tracker assistant. Analyze this email carefully.

SUBJECT: {subject}
FROM: {sender}
BODY:
{truncated_body}

IMPORTANT: To find the company name:
- FIRST look at the subject line — the company you applied TO is usually here
- If subject says "volunteer with X" or "application at X" or "joining X" → X is the company
- The signature company is the RECRUITER or SENDER company — may be different from where you applied
- Always pick the company the ROLE is FOR, not the recruiter's company
- IGNORE the email domain completely (@myworkday.com, @greenhouse.io etc are NOT company names)

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
    "company": "Exact company name — find from email body and signature only, ignore the email domain",
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
- ghosted = you followed up but got no response"""

    max_retries = 3
    for attempt in range(max_retries):
        try:
            message = client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=1024,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            raw = message.content[0].text.strip()
            raw = re.sub(r"```json|```", "", raw).strip()
            result = json.loads(raw)
            result["llm_called"] = True
            return result

        except anthropic.RateLimitError:
            wait = 30 * (attempt + 1)
            print(f"  Rate limit hit — waiting {wait} seconds...")
            time.sleep(wait)

        except anthropic.APIError as e:
            print(f"  Claude error: {e}")
            return {"is_job_related": False, "llm_called": True}

        except json.JSONDecodeError:
            print(f"  Could not parse response — skipping")
            return {"is_job_related": False, "llm_called": True}

    return {"is_job_related": False, "llm_called": True, "rate_limit_failed": True}
