
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
        logger.error("All API keys have been used")
        raise HTTPException(status_code=500, detail="All API keys have been used. Please add more keys.")

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

def is_follow_up(prompt: str) -> bool:
    """Check if the input is a follow-up request."""
    follow_up_phrases = ["yes", "more", "tell me more", "continue", "go on", "expand", "details", "elaborate", "further"]
    prompt_lower = prompt.lower().strip()
    return any(phrase == prompt_lower or phrase in prompt_lower for phrase in follow_up_phrases) and len(prompt_lower.split()) <= 4

def format_response(text, prompt: str, is_follow_up: bool = False):
    """Format the response with paragraphs, bullet points, and proper hyperlinks."""
    # Remove unwanted prompts or disclaimers from the raw response
    text = re.sub(r"Would you like more information\?|\b[Pp]lease\s+let\s+me\s+know\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"Disclaimer:.*?(?=\n|$)", "", text, flags=re.IGNORECASE)
    
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
    # Format hyperlinks (ensure no trailing punctuation)
    final_text = re.sub(r'(https?://[^\s<>]+)(?<![\.,;])', r'<a href="\1" target="_blank">\1</a>', final_text)
    
    # Append disclaimer only for non-follow-up responses
    if not is_follow_up:
        final_text += (
            "\n\nDisclaimer: This information is for educational purposes only and should not be considered legal advice. "
            "It is essential to consult with a legal professional for specific guidance regarding your situation."
        )
    
    return final_text.strip()

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

        # Check for initial welcome message
        session_file = os.path.join(SESSION_FOLDER, f"{session_id}.json")
        if not os.path.exists(session_file) and not prompt.strip():
            assistant_response = (
                "Hello! I'm Lexi, your legal assistant specializing in Indian law, IPC sections, and related topics. "
                "Ask me anything about legal rights, government procedures, or specific laws, and I'll provide detailed answers with sources."
            )
            return ChatResponse(response=assistant_response)

        # Handle greetings
        if is_greeting(prompt):
            assistant_response = "Hi there! I'm Lexi, ready to assist with your legal questions about Indian law. What's on your mind?"
            session_data.append({"role": "user", "text": prompt})
            session_data.append({"role": "assistant", "text": assistant_response})
            save_session(session_id, session_data)
            return ChatResponse(response=assistant_response)

        # Append user input to session history
        session_data.append({"role": "user", "text": prompt})

        # Load base prompt
        with open("prompts/base_prompt.txt", "r") as f:
            base_prompt = f.read()

        # Construct conversation history (last 6 messages for context, like Grok)
        history = "\n".join([f"{msg['role'].capitalize()}: {msg['text']}" for msg in session_data[-6:]])  # Increased to 6 for better context

        # Handle follow-up questions
        if is_follow_up(prompt):
            # Get the last user query and assistant response
            last_user_query = next((msg['text'] for msg in reversed(session_data[:-1]) if msg['role'] == 'user'), "")
            last_assistant_response = next((msg['text'] for msg in reversed(session_data[:-1]) if msg['role'] == 'assistant'), "")
            prompt_instruction = (
                f"The user has requested further information with the input '{prompt}'. "
                f"Refer to the last user query: '{last_user_query}' and the assistant's response: '{last_assistant_response}'. "
                "Provide a detailed, context-aware follow-up response that expands on the previous legal topic. "
                "Include specific IPC sections, Indian Acts, or case law with source URLs (e.g., https://www.indiankanoon.org, https://lddashboard.legislative.gov.in). "
                "Use a conversational tone, avoid asking 'Would you like more information?' or similar prompts, and do not include a disclaimer."
            )
        else:
            prompt_instruction = (
                "Provide a detailed, accurate response to the user's legal query, focusing on Indian law. "
                "Include specific IPC sections, Indian Acts, or case law with source URLs (e.g., https://www.indiankanoon.org, https://lddashboard.legislative.gov.in). "
                "Use a conversational tone and avoid asking 'Would you like more information?' or similar prompts."
            )

        # Construct prompt with formatting instructions
        formatting_instruction = (
            "Format the response with clear paragraphs separated by double newlines. "
            "Use bullet points (e.g., '* ') for lists or key points. "
            "Ensure hyperlinks are plain URLs (e.g., https://www.example.com) without trailing punctuation."
        )
        final_prompt = (
            f"{base_prompt}\n\n"
            f"{formatting_instruction}\n\n"
            f"{prompt_instruction}\n\n"
            f"Conversation History:\n{history}\n\n"
            f"User: {prompt}\nAssistant:"
        )

        if file_content:
            final_prompt = f"File content:\n{file_content}\n\n{final_prompt}"

        # Generate response
        async def generate_content():
            response = model.generate_content(final_prompt)
            return response

        try:
            response = await retry_request(generate_content)
            assistant_response = format_response(response.text, prompt, is_follow_up=is_follow_up(prompt))
            session_data.append({"role": "assistant", "text": assistant_response})
            save_session(session_id, session_data)
            return ChatResponse(response=assistant_response)
        except genai.QuotaExceededError:
            try:
                model = rotate_key()
                response = await retry_request(generate_content)
                assistant_response = format_response(response.text, prompt, is_follow_up=is_follow_up(prompt))
                session_data.append({"role": "assistant", "text": assistant_response})
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
