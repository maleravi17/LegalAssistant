from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import google.generativeai as genai
import json
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
import logging
from pypdf import PdfReader  # Replace PyPDF2 with pypdf
from werkzeug.utils import secure_filename

# Load environment variables
load_dotenv()
API_KEYS = [
    os.getenv("GEMINI_API_KEY_1"),
    os.getenv("GEMINI_API_KEY_2"),
    os.getenv("GEMINI_API_KEY_3")
]

if not API_KEYS or all(key is None for key in API_KEYS):
    raise ValueError("No valid API keys found in environment variables.")

SESSION_FOLDER = "sessions"
os.makedirs(SESSION_FOLDER, exist_ok=True)
UPLOADS_FOLDER = "uploads"
os.makedirs(UPLOADS_FOLDER, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class APIKeyManager:
    def __init__(self, keys):
        self.keys = [k for k in keys if k]  # Filter out None keys
        if not self.keys:
            raise ValueError("No valid API keys provided.")
        self.index = 0
        self.last_rotation = datetime.now()
        self.usage = {key: 0 for key in self.keys}
        self.max_requests_per_key = 100
        self.rotation_interval = timedelta(minutes=10)

    def get_model(self):
        if self.usage[self.keys[self.index]] >= self.max_requests_per_key or \
           (datetime.now() - self.last_rotation) > self.rotation_interval:
            self.rotate()
        try:
            genai.configure(api_key=self.keys[self.index])
            self.usage[self.keys[self.index]] += 1
            logger.info(f"Using API key index {self.index}")
            return genai.GenerativeModel('gemini-1.5-pro')
        except Exception as e:
            logger.error(f"API key {self.keys[self.index]} failed: {e}")
            self.rotate()
            if self.index >= len(self.keys):
                raise HTTPException(status_code=500, detail="All API keys have been exhausted. Please add more keys.")
            genai.configure(api_key=self.keys[self.index])
            self.usage[self.keys[self.index]] += 1
            return genai.GenerativeModel('gemini-1.5-pro')

    def rotate(self):
        self.usage[self.keys[self.index]] = 0
        self.index = (self.index + 1) % len(self.keys)
        self.last_rotation = datetime.now()

try:
    key_manager = APIKeyManager(API_KEYS)
    model = key_manager.get_model()
except ValueError as e:
    logger.error(f"Initialization failed: {e}")
    raise

def load_session(session_id):
    logger.info(f"Loading session: {session_id}")
    session_file = os.path.join(SESSION_FOLDER, f"{session_id}.json")
    if os.path.exists(session_file):
        logger.info(f"Session file found: {session_file}")
        try:
            with open(session_file, "r") as f:
                data = json.load(f)
                return {"history": data.get("history", []), "last_interaction": data.get("last_interaction", {})}
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in session file {session_file}: {e}")
            return {"history": [], "last_interaction": {}}
    logger.warning(f"Session file not found: {session_file}")
    return {"history": [], "last_interaction": {}}

def save_session(session_id, session_data):
    session_file = os.path.join(SESSION_FOLDER, f"{session_id}.json")
    try:
        with open(session_file, "w") as f:
            json.dump({"history": session_data["history"], "last_interaction": session_data["last_interaction"]}, f)
        logger.info(f"Session saved: {session_file}")
    except Exception as e:
        logger.error(f"Failed to save session {session_file}: {e}")

def format_response(text):
    paragraphs = text.split('\n\n') if '\n\n' in text else text.split('\n')
    formatted = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if para.startswith('* ') or para.startswith('- ') or para.startswith('**'):
            lines = para.split('\n')
            formatted_para = []
            for line in lines:
                line = line.strip()
                if line.startswith('* ') or line.startswith('- '):
                    formatted_para.append(f"• {line[2:]}")
                elif line.startswith('**') and line.endswith('**'):
                    formatted_para.append(f"\n**{line[2:-2]}**\n")
                else:
                    formatted_para.append(line)
            formatted.append('\n'.join(formatted_para))
        else:
            formatted.append(para)
    return '\n\n'.join(formatted)

# --- FastAPI App ---
app = FastAPI()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

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
    logger.info(f"New chat request: session_id={request.session_id}, prompt={request.prompt}")
    session_data = load_session(request.session_id)
    session_data["history"].append({"role": "user", "text": request.prompt})
    session_data["last_interaction"] = {"prompt": request.prompt, "response": None}

    # Special handling for session load
    if request.prompt == "Load this session":
        logger.info(f"Loading session history for {request.session_id}")
        return ChatResponse(response="Session loaded successfully.")

    examples = """
    Example 1:
    User: What is the difference between civil law and criminal law?
    Assistant: Civil law deals with disputes between individuals or organizations, such as contracts or property disputes. Criminal law, on the other hand, involves actions that are harmful to society and are prosecuted by the state, such as theft or assault.
    ...
    """

    prompt = f"""
    You are a legal assistant specializing in Indian law, IPC section, justice, advocates, lawyers, official Passports related, and judgment-related topics.
    ...
    """

    try:
        response = model.generate_content(prompt)
        assistant_response = format_response(response.text)
        session_data["last_interaction"]["response"] = assistant_response
        session_data["history"].append({"role": "assistant", "text": assistant_response})
        save_session(request.session_id, session_data)
        logger.info(f"Chat response generated for session_id={request.session_id}")
        return ChatResponse(response=assistant_response)
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        new_model = key_manager.get_model()
        if new_model:
            model = new_model
            return await chat_with_law_assistant(request)
        raise HTTPException(status_code=500, detail="Sorry, I am unable to process your request at the moment.")

# ... (regenerate and upload endpoints remain the same) ...

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
