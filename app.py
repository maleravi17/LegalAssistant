from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import google.generativeai as genai
import json
import os
from dotenv import load_dotenv

load_dotenv()
API_KEYS = [
    os.getenv("GEMINI_API_KEY_1"),
    os.getenv("GEMINI_API_KEY_2"),
    os.getenv("GEMINI_API_KEY_3")
]

current_key_index = 0
SESSION_FOLDER = "sessions"
os.makedirs(SESSION_FOLDER, exist_ok=True)

def load_session(session_id):
    session_file = os.path.join(SESSION_FOLDER, f"{session_id}.json")
    if os.path.exists(session_file):
        with open(session_file, "r") as f:
            return json.load(f)
    return []

def save_session(session_id, session_data):
    session_file = os.path.join(SESSION_FOLDER, f"{session_id}.json")
    with open(session_file, "w") as f:
        json.dump(session_data, f)

def initialize_gemini():
    global current_key_index
    try:
        genai.configure(api_key=API_KEYS[current_key_index])
        return genai.GenerativeModel('gemini-1.5-pro')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error initializing Gemini: {e}")

def rotate_key():
    global current_key_index
    if current_key_index < len(API_KEYS) - 1:
        current_key_index += 1
        return initialize_gemini()
    else:
        raise HTTPException(status_code=500, detail="All API keys exhausted.")

model = initialize_gemini()
app = FastAPI()

# Add CORS middleware to fix 405 error
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Serve static files (e.g., index.html)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Root endpoint to serve HTML
@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("static/index.html", "r") as f:
        return f.read()

class ChatRequest(BaseModel):
    session_id: str
    prompt: str

class ChatResponse(BaseModel):
    response: str

@app.post("/chat", response_model=ChatResponse)
async def chat_with_law_assistant(request: ChatRequest):
    global model
    session_data = load_session(request.session_id)
    session_data.append({"role": "user", "text": request.prompt})

    examples = """
    Example 1:
    User: What is the difference between civil law and criminal law?
    Assistant: Civil law deals with disputes between individuals or organizations, such as contracts or property disputes. Criminal law involves actions harmful to society, prosecuted by the state, like theft or assault.
    """

    prompt = f"""
    You are a legal assistant specializing in Indian law, IPC sections, justice, advocates, lawyers, passports, and judgments.
    Provide accurate answers with IPC sections and Indian Acts where relevant.
    Use plain language and decline non-legal questions politely.

    {examples}

    History: {" ".join([f"{msg['role']}: {msg['text']}" for msg in session_data])}

    User: {request.prompt}
    Assistant:
    """

    try:
        response = model.generate_content(prompt)
        assistant_response = response.text
        session_data.append({"role": "assistant", "text": assistant_response})
        save_session(request.session_id, session_data)
        return ChatResponse(response=assistant_response)
    except Exception:
        new_model = rotate_key()
        if new_model:
            model = new_model
            return await chat_with_law_assistant(request)
        raise HTTPException(status_code=500, detail="Unable to process request.")