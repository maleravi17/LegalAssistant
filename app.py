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
import urllib.parse

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

def load_session(session_id):
    """Load session data from a file."""
    session_file = os.path.join(SESSION_FOLDER, f"{session_id}.json")
    if os.path.exists(session_file):
        try:
            with open(session_file, "r") as f:
                return json.load(f)
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
            return {"history": [], "recent_follow_ups": []}
    logger.warning(f"Session file not found: {session_file}")
    return {"history": [], "recent_follow_ups": []}

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
    """Initialize the Gemini model and return the model instance."""
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
    """Rotate to the next API key."""
    global current_key_index
    if current_key_index < len(API_KEYS) - 1:
        current_key_index += 1
        logger.info(f"Rotated to API key index {current_key_index}")
        return initialize_gemini()
    else:
        current_key_index = 0  # Cycle back to first key
        logger.info(f"Cycled back to API key index {current_key_index}")
        return initialize_gemini()

async def retry_request(func, retries=3, delay=5):
    """Retry a function with exponential backoff."""
    for attempt in range(retries):
        try:
            return await func()
        except Exception as e:
            if attempt == retries - 1:
                logger.error(f"Failed after {retries} attempts: {str(e)}")
                raise
            logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {delay} seconds...")
            await asyncio.sleep(delay)
            delay *= 2  # Exponential backoff

async def process_uploaded_file(file: UploadFile):
    """Process uploaded PDF or image files."""
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
    """Check if the input is a simple greeting."""
    greetings = ["hello", "hi", "hey", "hola", "namaste", "good morning", "good evening"]
    prompt_lower = prompt.lower().strip()
    return any(greeting in prompt_lower for greeting in greetings) and len(prompt_lower.split()) <= 2

def generate_follow_up_question(prompt: str, response: str, recent_follow_ups: list) -> str:
    """Generate a contextually relevant follow-up question based on the prompt and response."""
    follow_up_questions = [
        "Would you like to explore specific case laws related to this topic?",
        "Do you need details on relevant IPC sections or Indian Acts?",
        "Would you like assistance with drafting a legal document?",
        "Are you seeking guidance on procedural steps for this issue?",
        "Would you like information on recent amendments to this law?",
        "Do you need help applying this to a specific scenario?",
        "Would you like references to official government resources?",
        "Are you interested in defenses or remedies under this law?"
    ]

    # Filter out recently used questions (last 3 responses)
    available_questions = [q for q in follow_up_questions if q not in recent_follow_ups[-3:]]

    # If all questions have been used recently, reset to full list
    if not available_questions:
        available_questions = follow_up_questions

    # Prioritize questions based on context
    if "ipc section" in prompt.lower() or "indian penal code" in prompt.lower():
        prioritized = [q for q in available_questions if "IPC sections" in q or "case laws" in q]
        return random.choice(prioritized) if prioritized else random.choice(available_questions)
    elif "act" in prompt.lower() or "law" in prompt.lower():
        prioritized = [q for q in available_questions if "Indian Acts" in q or "amendments" in q]
        return random.choice(prioritized) if prioritized else random.choice(available_questions)
    elif "procedure" in prompt.lower() or "process" in prompt.lower():
        prioritized = [q for q in available_questions if "procedural steps" in q]
        return random.choice(prioritized) if prioritized else random.choice(available_questions)
    elif "case" in prompt.lower() or "judgment" in prompt.lower():
        prioritized = [q for q in available_questions if "case laws" in q]
        return random.choice(prioritized) if prioritized else random.choice(available_questions)
    else:
        return random.choice(available_questions)

def format_response(text, prompt: str, session_data: dict):
    """Format the response with paragraphs, bullet points, proper hyperlinks, and a contextually relevant follow-up."""
    # Split text into paragraphs
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

    final_text = '\n\n'.join(formatted)

    # Remove any existing follow-up questions to prevent duplication
    follow_up_pattern = r'(Would you like|Do you need|Are you seeking|Are you interested in).+\?\s*$'
    final_text = re.sub(follow_up_pattern, '', final_text, flags=re.IGNORECASE | re.MULTILINE).strip()

    # Format hyperlinks with descriptive text
    def format_url(match):
        url = match.group(0)
        parsed_url = urllib.parse.urlparse(url)
        # Simplify URL for display (remove query params)
        display_text = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}"
        if "indiacode.nic.in" in url:
            display_text = "India Code"
        elif "legislative.gov.in" in url:
            display_text = "Legislative Department"
        elif "dopt.gov.in" in url:
            display_text = "Department of Personnel and Training"
        return f'<a href="{url}" target="_blank">{display_text}</a>'

    final_text = re.sub(r'(https?://[^\s<>]+|www\.[^\s<>]+)', format_url, final_text)

    # Append a contextually relevant follow-up question
    recent_follow_ups = session_data.get("recent_follow_ups", [])
    follow_up = generate_follow_up_question(prompt, final_text, recent_follow_ups)
    recent_follow_ups.append(follow_up)
    session_data["recent_follow_ups"] = recent_follow_ups[-5:]  # Keep last 5 for context
    final_text = f"{final_text}\n\n**Follow-Up Question:** {follow_up}"

    return final_text, session_data

# Initialize Gemini model
model = initialize_gemini()

# FastAPI App
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
    prompt: str
    regenerate: bool = False

class ChatResponse(BaseModel):
    response: str

@app.post("/chat", response_model=ChatResponse)
async def chat_with_law_assistant(session_id: str = Form(...), prompt: str = Form(...), file: UploadFile = File(None)):
    global model
    try:
        file_content = ""
        if file:
            file_content = await process_uploaded_file(file)

        # Validate session_id
        if not session_id:
            raise HTTPException(status_code=422, detail="session_id is required")

        # Load session data
        session_data = load_session(session_id)
        history = session_data.get("history", [])

        # Check for initial welcome message
        session_file = os.path.join(SESSION_FOLDER, f"{session_id}.json")
        if not os.path.exists(session_file) and not prompt.strip():
            assistant_response = "Okay, I'm ready to assist you with your legal questions related to Indian law, including IPC sections, Indian Acts, judgments, passport-related issues, and other relevant topics. I can also help determine legal rights and official government procedures within my area of expertise. Please ask your question."
            return ChatResponse(response=assistant_response)

        # Handle greetings
        if is_greeting(prompt):
            assistant_response = "Hello! I'm Lexi, your legal assistant for Indian law. How can I help you today?"
            history.append({"role": "user", "text": prompt})
            history.append({"role": "assistant", "text": assistant_response})
            session_data["history"] = history
            save_session(session_id, session_data)
            return ChatResponse(response=assistant_response)

        # Append user input to session history
        history.append({"role": "user", "text": prompt})

        # Check for expanded response
        expanded_response = False
        last_user_prompt = prompt
        if history and len(history) >= 2:
            last_assistant_msg = history[-2] if history[-2]["role"] == "assistant" else None
            if last_assistant_msg and last_assistant_msg["text"].strip().endswith(("Would you like to explore specific case laws related to this topic?",
                                                                                 "Do you need details on relevant IPC sections or Indian Acts?",
                                                                                 "Would you like assistance with drafting a legal document?",
                                                                                 "Are you seeking guidance on procedural steps for this issue?",
                                                                                 "Would you like information on recent amendments to this law?",
                                                                                 "Do you need help applying this to a specific scenario?",
                                                                                 "Would you like references to official government resources?",
                                                                                 "Are you interested in defenses or remedies under this law?")) and prompt.lower() == "yes":
                expanded_response = True
                last_user_prompt = history[-3]["text"] if len(history) >= 3 and history[-3]["role"] == "user" else prompt

        # Load base prompt
        with open("prompts/base_prompt.txt", "r") as f:
            base_prompt = f.read()

        # Construct conversation history
        history_text = " ".join([f"{msg['role']}: {msg['text']}" for msg in history])

        # Construct prompt with formatting instructions for all responses
        formatting_instruction = "Format the response with clear paragraphs separated by double newlines and use bullet points (e.g., '* ') for lists or key points."
        if expanded_response:
            prompt = f"{base_prompt}\n\nThe user previously asked: \"{last_user_prompt}\". They have responded \"yes\" to request more information.\nProvide a detailed response with specific IPC sections, relevant Indian Acts, and case law examples (e.g., case names, court, year) related to the topic. Include source websites or URLs. {formatting_instruction}\n\nConversation History:\n{history_text}\n\nUser: yes\nAssistant:"
        else:
            prompt = f"{base_prompt}\n\n{formatting_instruction}\n\nConversation History:\n{history_text}\n\nUser: {prompt}\nAssistant:"

        if file_content:
            prompt = f"File content:\n{file_content}\n\n{prompt}"

        # Generate response
        async def generate_content():
            response = model.generate_content(prompt)
            return response

        try:
            response = await retry_request(generate_content)
            assistant_response, session_data = format_response(response.text, prompt, session_data)
            history.append({"role": "assistant", "text": assistant_response})
            session_data["history"] = history
            save_session(session_id, session_data)
            return ChatResponse(response=assistant_response)
        except genai.QuotaExceededError:
            try:
                model = rotate_key()
                response = await retry_request(generate_content)
                assistant_response, session_data = format_response(response.text, prompt, session_data)
                history.append({"role": "assistant", "text": assistant_response})
                session_data["history"] = history
                save_session(session_id, session_data)
                return ChatResponse(response=assistant_response)
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
    """Regenerate a response for the given prompt."""
    if not request.prompt:
        raise HTTPException(status_code=400, detail="Prompt is required for regeneration.")
    return await chat_with_law_assistant(session_id=request.session_id, prompt=request.prompt)

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
