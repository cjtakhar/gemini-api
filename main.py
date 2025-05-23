from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import json
import os

# Try to get the API key from environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

#  Fallback to secrets.json for local development
if not GEMINI_API_KEY:
    try:
        with open("secrets.json") as f:
            secrets = json.load(f)
            GEMINI_API_KEY = secrets.get("GEMINI_API_KEY")
    except FileNotFoundError:
        raise RuntimeError("GEMINI_API_KEY is not set and secrets.json is missing")

#  Final safety check
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY could not be loaded from environment or secrets.json")

#  Construct Gemini API endpoint
GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
)

# Create FastAPI app
app = FastAPI(
    title="Gemini Question Answering API",
    description="Ask a question and get a Gemini answer."
)

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",       # Vite dev server
        "https://cjtakhar.github.io"  # GitHub Pages
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request and response models
class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str

# POST endpoint to ask a question
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

# Optional root route to show API status
@app.get("/")
def read_root():
    return {"message": "Gemini API is live and ready."}