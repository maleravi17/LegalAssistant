import os
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import google.generativeai as genai
from dotenv import load_dotenv
import PyPDF2
import json
import asyncio
import re
import time
import shutil
import logging
import tempfile
import random
from datetime import datetime
from typing import List, Dict, Optional

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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Message(BaseModel):
    message_id: str
    role: str
    text: str
    timestamp: str
    intent: Optional[str] = None

class SessionData(BaseModel):
    messages: List[Message]
    memory_enabled: bool = True

def load_session(session_id: str) -> SessionData:
    session_file = os.path.join(SESSION_FOLDER, f"{session_id}.json")
    if os.path.exists(session_file):
        try:
            with open(session_file, "r") as f:
                data = json.load(f)
                messages = [Message(**msg) for msg in data.get("messages", [])]
                return SessionData(messages=messages, memory_enabled=data.get("memory_enabled", True))
        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"Error loading session file {session_file}: {str(e)}")
            backup_dir = "backup_sessions"
            os.makedirs(backup_dir, exist_ok=True)
            backup_file = os.path.join(backup_dir, f"{session_id}_{time.strftime('%Y%m%d-%H%M%S')}.json")
            try:
                shutil.move(session_file, backup_file)
                logger.info(f"Moved corrupted session file to backup: {backup_file}")
            except Exception as move_error:
                logger.error(f"Failed to move corrupted session file: {move_error}")
            return SessionData(messages=[])
    logger.warning(f"Session file not found: {session_file}")
    return SessionData(messages=[])

def save_session(session_id: str, session_data: SessionData):
    session_file = os.path.join(SESSION_FOLDER, f"{session_id}.json")
    try:
        with open(session_file, "w") as f:
            json.dump(session_data.dict(), f)
        logger.info(f"Successfully saved session: {session_id}")
    except Exception as e:
        logger.error(f"Failed to save session {session_file}: {str(e)}")

def initialize_gemini():
    global model
    try:
        genai.configure(api_key=API_KEYS[current_key_index])
        best_model = 'models/gemini-2.0-flash-001'
        model = genai.GenerativeModel(best_model)
        logger.info(f"Initialized Gemini model with API key index {current_key_index}")
        return model
    except Exception as e:
        logger.error(f"Failed to initialize Gemini model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to initialize Gemini model: {str(e)}")

def rotate_key():
    global current_key_index
    if current_key_index < len(API_KEYS) - 1:
        current_key_index += 1
        logger.info(f"Rotated to API key index {current_key_index}")
        return initialize_gemini()
    else:
        logger.error("All API keys have been used")
        raise HTTPException(status_code=500, detail="All API keys have been used. Please add more keys.")

async def retry_request(func, retries=3, delay=5):
    for attempt in range(retries):
        try:
            return await func()
        except Exception as e:
            if attempt == retries - 1:
                logger.error(f"Failed after {retries} attempts: {str(e)}")
                raise
            logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {delay} seconds...")
            await asyncio.sleep(delay)
            delay *= 2

async def process_uploaded_file(file: UploadFile):
    if file.content_type == 'application/pdf':
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(await file.read())
                temp_file_path = temp_file.name
            with open(temp_file_path, 'rb') as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                text = ""
                for page in pdf_reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"
            os.unlink(temp_file_path)
            logger.info(f"Successfully processed PDF file: {file.filename}")
            return text.strip() if text.strip() else "No text could be extracted from the PDF."
        except Exception as e:
            logger.error(f"Error processing PDF file {file.filename}: {str(e)}")
            return f"Error processing PDF file: {str(e)}"
    elif file.content_type.startswith('image/'):
        logger.info(f"Image file uploaded: {file.filename}. Image processing not implemented.")
        return f"Uploaded image: {file.filename}. Image processing is not currently supported."
    else:
        logger.error(f"Unsupported file type: {file.content_type}")
        return f"Unsupported file type: {file.content_type}"

def is_greeting(prompt: str) -> bool:
    greetings = ["hello", "hi", "hey", "hola", "namaste", "good morning", "good evening"]
    prompt_lower = prompt.lower().strip()
    return any(greeting in prompt_lower for greeting in greetings) and len(prompt_lower.split()) <= 2

def summarize_history(messages: List[Message], max_length: int = 500) -> str:
    if not messages:
        return ""
    summary = []
    for msg in messages[-10:]:
        if msg.role == "user":
            summary.append(f"User asked: {msg.text}")
        elif msg.role == "assistant":
            summary.append(f"Assistant responded: {msg.text}")
    summary_text = " ".join(summary)
    return summary_text[:max_length] + "..." if len(summary_text) > max_length else summary_text

def format_response(text, prompt: str):
    paragraphs = text.split('\n\n') if '\n\n' in text else [text]
    formatted = []
    url_pattern = r'(?<![\w-])(https?://[^\s<>\[\]]+)(?![\w-])'  # Improved regex to avoid capturing brackets

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # Handle URLs first to ensure they are clickable
        def replace_url(match):
            url = match.group(1)
            return f'<a href="{url}" target="_blank" rel="noopener noreferrer">{url}</a>'

        para = re.sub(url_pattern, replace_url, para)

        # Handle bullet points (preserve existing behavior)
        if para.startswith('* ') or para.startswith('- '):
            lines = para.split('\n')
            list_items = []
            in_list = False
            for line in lines:
                line = line.strip()
                if line.startswith('* ') or line.startswith('- '):
                    if not in_list:
                        list_items.append('<ul>')
                        in_list = True
                    list_items.append(f'<li>{line[2:]}</li>')
                else:
                    if in_list:
                        list_items.append('</ul>')
                        in_list = False
                    list_items.append(line)
            if in_list:
                list_items.append('</ul>')
            para = '\n'.join(list_items)
        elif para.startswith('**') and para.endswith('**'):
            para = f"<p><strong>{para[2:-2]}</strong></p>"
        else:
            para = f"<p>{para}</p>"

        formatted.append(para)

    final_text = '\n'.join(formatted)
    return final_text

# Initialize Gemini model
model = initialize_gemini()

# FastAPI App
app = FastAPI()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS", "HEAD"],
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
    prompt: str
    regenerate: bool = False

class ChatResponse(BaseModel):
    response: str
    message_id: str

class MemoryResponse(BaseModel):
    messages: List[Message]

@app.post("/chat", response_model=ChatResponse)
async def chat_with_law_assistant(session_id: str = Form(...), prompt: str = Form(...), file: UploadFile = File(None)):
    global model
    try:
        file_content = ""
        if file:
            file_content = await process_uploaded_file(file)

        if not session_id:
            raise HTTPException(status_code=422, detail="session_id is required")

        session_data = load_session(session_id)

        session_file = os.path.join(SESSION_FOLDER, f"{session_id}.json")
        if not os.path.exists(session_file) and not prompt.strip():
            assistant_response = "Okay, I'm ready to assist you with your legal questions related to Indian law, including IPC sections, Indian Acts, judgments, passport-related issues, and other relevant topics. I can also help determine legal rights and official government procedures within my area of expertise. Please ask your question."
            return ChatResponse(response=assistant_response, message_id="initial")

        if is_greeting(prompt):
            assistant_response = "Hello! I'm Lexi, your legal assistant for Indian law. How can I help you today?"
            message_id = f"msg-{int(time.time())}"
            session_data.messages.append(Message(
                message_id=message_id,
                role="user",
                text=prompt,
                timestamp=datetime.utcnow().isoformat()
            ))
            session_data.messages.append(Message(
                message_id=f"msg-{int(time.time())}",
                role="assistant",
                text=assistant_response,
                timestamp=datetime.utcnow().isoformat()
            ))
            save_session(session_id, session_data)
            return ChatResponse(response=assistant_response, message_id=message_id)

        message_id = f"msg-{int(time.time())}"
        if session_data.memory_enabled:
            session_data.messages.append(Message(
                message_id=message_id,
                role="user",
                text=prompt,
                timestamp=datetime.utcnow().isoformat()
            ))
        else:
            session_data.messages = [Message(
                message_id=message_id,
                role="user",
                text=prompt,
                timestamp=datetime.utcnow().isoformat()
            )]

        with open("prompts/base_prompt.txt", "r") as f:
            base_prompt = f.read()

        history_summary = summarize_history(session_data.messages)
        history = "\n".join([f"{msg.role}: {msg.text}" for msg in session_data.messages])

        formatting_instruction = "Format the response with clear paragraphs separated by double newlines and use bullet points (e.g., '* ') for lists or key points. For follow-up questions, reference the conversation history to provide context-aware responses. Include URLs as clickable links."
        prompt = f"{base_prompt}\n\n{formatting_instruction}\n\nConversation Summary:\n{history_summary}\n\nFull Conversation History:\n{history}\n\nUser: {prompt}\nAssistant:"

        if file_content:
            prompt = f"File content:\n{file_content}\n\n{prompt}"

        async def generate_content():
            response = model.generate_content(prompt)
            return response

        try:
            response = await retry_request(generate_content)
            assistant_response = format_response(response.text, prompt)
            assistant_message_id = f"msg-{int(time.time())}"
            if session_data.memory_enabled:
                session_data.messages.append(Message(
                    message_id=assistant_message_id,
                    role="assistant",
                    text=assistant_response,
                    timestamp=datetime.utcnow().isoformat()
                ))
            save_session(session_id, session_data)
            return ChatResponse(response=assistant_response, message_id=assistant_message_id)
        except genai.QuotaExceededError:
            try:
                model = rotate_key()
                response = await retry_request(generate_content)
                assistant_response = format_response(response.text, prompt)
                assistant_message_id = f"msg-{int(time.time())}"
                if session_data.memory_enabled:
                    session_data.messages.append(Message(
                        message_id=assistant_message_id,
                        role="assistant",
                        text=assistant_response,
                        timestamp=datetime.utcnow().isoformat()
                    ))
                save_session(session_id, session_data)
                return ChatResponse(response=assistant_response, message_id=assistant_message_id)
            except genai.QuotaExceededError:
                raise HTTPException(status_code=429, detail="Quota exceeded for all API keys. Please check your API plan at https://ai.google.dev/gemini-api/docs/rate-limits.")
        except Exception as e:
            logger.error(f"Error generating content: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error generating content: {str(e)}")

    except Exception as e:
        logger.error(f"Unexpected error in /chat: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.post("/regenerate", response_model=ChatResponse)
async def regenerate_response(request: ChatRequest):
    if not request.prompt:
        raise HTTPException(status_code=400, detail="Prompt is required for regeneration.")
    return await chat_with_law_assistant(session_id=request.session_id, prompt=request.prompt)

@app.get("/memory/{session_id}", response_model=MemoryResponse)
async def get_memory(session_id: str):
    session_data = load_session(session_id)
    return MemoryResponse(messages=session_data.messages)

@app.delete("/memory/{session_id}/{message_id}")
async def forget_message(session_id: str, message_id: str):
    session_data = load_session(session_id)
    session_data.messages = [msg for msg in session_data.messages if msg.message_id != message_id]
    save_session(session_id, session_data)
    return {"status": "success", "message": f"Message {message_id} forgotten"}

@app.post("/memory/{session_id}/toggle")
async def toggle_memory(session_id: str, enable: bool = Form(...)):
    session_data = load_session(session_id)
    session_data.memory_enabled = enable
    if not enable:
        session_data.messages = []
    save_session(session_id, session_data)
    return {"status": "success", "memory_enabled": enable"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
