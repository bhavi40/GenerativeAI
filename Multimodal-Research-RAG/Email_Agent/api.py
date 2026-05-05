from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from chat import ask
from db import get_stats, get_all_applications

app = FastAPI(
    title="Job Tracker API",
    description="AI-powered job application tracking chatbot API",
    version="1.0.0"
)

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    question: str
    answer: str
    sql_generated: str = None

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "job-tracker-api"}

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    try:
        answer = ask(request.question)
        return ChatResponse(
            question=request.question,
            answer=answer
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
def stats_endpoint():
    try:
        stats = get_stats()
        return {
            "total": stats["total"],
            "applied": stats["applied"],
            "interviewing": stats["interviewing"],
            "rejected": stats["rejected"],
            "offers": stats["offers"],
            "interested": stats["interested"],
            "ghosted": stats["ghosted"],
            "recruiter_outreach": stats["recruiter_outreach"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/applications")
def applications_endpoint(status: str = None):
    try:
        apps = get_all_applications()
        if status:
            apps = [a for a in apps if a["status"] == status]
        return {
            "total": len(apps),
            "applications": [dict(a) for a in apps]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/applications/{company}")
def get_company_applications(company: str):
    try:
        apps = get_all_applications()
        company_apps = [
            a for a in apps 
            if company.lower() in a["company"].lower()
        ]
        return {
            "company": company,
            "total": len(company_apps),
            "applications": [dict(a) for a in company_apps]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))