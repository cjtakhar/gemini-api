from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import json
import os

# ✅ Try to get the API key from environment
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ✅ Fallback to secrets.json for local dev only
if not GEMINI_API_KEY and os.path.exists("secrets.json"):
    with open("secrets.json") as f:
        secrets = json.load(f)
        GEMINI_API_KEY = secrets.get("GEMINI_API_KEY")

# ✅ Fail gracefully if still no key
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set and secrets.json is missing")

GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
)

app = FastAPI(title="Gemini Question Answering API")

# ✅ CORS for localhost + GitHub Pages
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://cjtakhar.github.io"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str

@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    print(f"🧠 Received question: {request.question}")
    payload = {
        "contents": [
            {"parts": [{"text": request.question}]}
        ]
    }

    headers = {"Content-Type": "application/json"}
    response = requests.post(GEMINI_URL, headers=headers, data=json.dumps(payload))

    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.text)

    try:
        result = response.json()
        answer = result["candidates"][0]["content"]["parts"][0]["text"]
        return AnswerResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse response: {str(e)}")

# ✅ Respond to preflight CORS checks
@app.options("/ask")
def options_ask():
    print("🔥 Received OPTIONS /ask")
    return {}

# ✅ Friendly root message
@app.get("/")
def root():
    return {"message": "Gemini API is live and ready."}
