from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import google.generativeai as genai
import json
import os
from dotenv import load_dotenv
import time
import logging
import re

# Load environment variables
load_dotenv()
API_KEYS = [
    os.getenv("GEMINI_API_KEY_1"),
    os.getenv("GEMINI_API_KEY_2"),
    os.getenv("GEMINI_API_KEY_3")
]

current_key_index = 0
SESSION_FOLDER = "sessions"
os.makedirs(SESSION_FOLDER, exist_ok=True)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_session(session_id):
    """Load session data from a file."""
    session_file = os.path.join(SESSION_FOLDER, f"{session_id}.json")
    if os.path.exists(session_file):
        try:
            with open(session_file, "r") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in session file {session_file}: {str(e)}")
            return []
    logger.warning(f"Session file not found: {session_file}")
    return []

def save_session(session_id, session_data):
    """Save session data to a file."""
    session_file = os.path.join(SESSION_FOLDER, f"{session_id}.json")
    try:
        with open(session_file, "w") as f:
            json.dump(session_data, f)
        logger.info(f"Successfully saved session: {session_file}")
    except Exception as e:
        logger.error(f"Failed to save session {session_file}: {str(e)}")

def initialize_gemini():
    """Initialize Gemini with the current API key."""
    global current_key_index
    if not API_KEYS or all(key is None for key in API_KEYS):
        raise HTTPException(status_code=500, detail="No valid API keys found in environment variables.")
    try:
        genai.configure(api_key=API_KEYS[current_key_index])
        return genai.GenerativeModel('gemini-1.5-pro')
    except Exception as e:
        logger.error(f"Failed to initialize Gemini with key index {current_key_index}: {str(e)}")
        if current_key_index < len(API_KEYS) - 1:
            current_key_index += 1
            return initialize_gemini()
        raise HTTPException(status_code=500, detail=f"Error initializing Gemini: {str(e)}")

def rotate_key():
    """Rotate to the next API key."""
    global current_key_index
    if current_key_index < len(API_KEYS) - 1:
        current_key_index += 1
        return initialize_gemini()
    else:
        raise HTTPException(status_code=500, detail="All API keys have been used. Please add more keys.")

def is_greeting(prompt: str) -> bool:
    """Check if the input is a simple greeting."""
    greetings = ["hello", "hi", "hey", "hola", "namaste"]
    prompt_lower = prompt.lower().strip()
    return any(greeting in prompt_lower for greeting in greetings) and len(prompt_lower.split()) <= 2

def format_response(text):
    """Format the response with paragraphs and bullet points, removing duplicate endings."""
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
    # Remove duplicate "Would you like more information?" endings
    final_text = '\n\n'.join(formatted)
    final_text = re.sub(r'Would you like more information\?\s*Would you like more information\?', 'Would you like more information?', final_text, flags=re.IGNORECASE)
    return final_text

def retry_request(func, max_retries=3, delay=5):
    """Retry the API call if a quota error occurs."""
    for attempt in range(max_retries):
        try:
            return func()
        except genai.QuotaExceededError as e:
            logger.warning(f"Quota exceeded on attempt {attempt + 1}: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(delay)
                continue
            raise HTTPException(status_code=429, detail="Quota exceeded. Please check your API plan at https://ai.google.dev/gemini-api/docs/rate-limits.")
        except Exception as e:
            logger.error(f"Unexpected error during API call: {str(e)}")
            raise HTTPException(status_code=500, detail="Internal server error.")

# Initialize Gemini model
model = initialize_gemini()

# --- FastAPI App ---
app = FastAPI()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS", "HEAD"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("static/index.html", "r") as f:
        return f.read()

@app.head("/")
async def head_root():
    return HTMLResponse(status_code=200)

class ChatRequest(BaseModel):
    session_id: str
