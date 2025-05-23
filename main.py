from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import json
import os

# Try to get the API key from environment variable
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Fallback to secrets.json if not found (local dev only)
if not GEMINI_API_KEY:
    with open("secrets.json") as f:
        secrets = json.load(f)
        GEMINI_API_KEY = secrets["GEMINI_API_KEY"]

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set in environment or secrets.json")

GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

app = FastAPI(title="Gemini Question Answering API", description="Ask a question and get a Gemini answer.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "https://cjtakhar.github.io"],  # Local + GitHub Pages
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
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": request.question
                    }
                ]
            }
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
