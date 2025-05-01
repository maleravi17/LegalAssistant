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

def format_response(text, prompt: str):
    """Format the response with paragraphs, bullet points, and properly formatted hyperlinks."""
    # First, split text into paragraphs
    paragraphs = text.split('\n\n') if '\n\n' in text else text.split('\n')
    formatted = []

    # Regex to match URLs, excluding square brackets
    url_pattern = r'(?<![\w-])(https?:\/\/[^\s<>\]\)]+)(?![\w-])'

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # Handle paragraphs with bullet points or bold text
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
            para = '\n'.join(formatted_para)

        # Process hyperlinks: detect URLs and strip surrounding square brackets
        def replace_url(match):
            url = match.group(1)
            # Strip square brackets if they exist
            if url.startswith('[') and url.endswith(']'):
                url = url[1:-1]
            return f'<a href="{url}" target="_blank">{url}</a>'

        # Apply hyperlink formatting
        para = re.sub(url_pattern, replace_url, para)

        formatted.append(para)

    # Join paragraphs with double newlines
    final_text = '\n\n'.join(formatted)
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
            assistant_response = "Okay, I'm ready to assist you with your legal questions related to Indian law, including IPC sections, Indian Acts, judgments, passport-related issues, and other relevant topics. I can also help determine legal rights and official government procedures within my area of expertise. Please ask your question."
            return ChatResponse(response=assistant_response)

        # Handle greetings
        if is_greeting(prompt):
            assistant_response = "Hello! I'm Lexi, your legal assistant for Indian law. How can I help you today?"
            session_data.append({"role": "user", "text": prompt})
            session_data.append({"role": "assistant", "text": assistant_response})
            save_session(session_id, session_data)
            return ChatResponse(response=assistant_response)

        # Append user input to session history
        session_data.append({"role": "user", "text": prompt})

        # Load base prompt
        with open("prompts/base_prompt.txt", "r") as f:
            base_prompt = f.read()

        # Construct conversation history
        history = " ".join([f"{msg['role']}: {msg['text']}" for msg in session_data])

        # Construct prompt with formatting instructions
        formatting_instruction = "Format the response with clear paragraphs separated by double newlines and use bullet points (e.g., '* ') for lists or key points."
        prompt = f"{base_prompt}\n\n{formatting_instruction}\n\nConversation History:\n{history}\n\nUser: {prompt}\nAssistant:"

        if file_content:
            prompt = f"File content:\n{file_content}\n\n{prompt}"

        # Generate response
        async def generate_content():
            response = model.generate_content(prompt)
            return response

        try:
            response = await retry_request(generate_content)
            assistant_response = format_response(response.text, prompt)
            session_data.append({"role": "assistant", "text": assistant_response})
            save_session(session_id, session_data)
            return ChatResponse(response=assistant_response)
        except genai.QuotaExceededError:
            try:
                model = rotate_key()
                response = await retry_request(generate_content)
                assistant_response = format_response(response.text, prompt)
                session_data.append({"role": "assistant", "text": assistant_response})
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
    # Call the chat endpoint with regenerate=True
    return await chat_with_law_assistant(session_id=request.session_id, prompt=request.prompt)

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
