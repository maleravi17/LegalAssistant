from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from fastapi import HTTPException
import PyPDF2
import json
import os
import asyncio
import re
import time
import shutil, logging
import tempfile


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
        best_model = 'models/gemini-2.0-flash-001'
        model = genai.GenerativeModel(best_model)  
    except Exception as e:
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
    greetings = ["hello", "hi", "hey", "hola", "namaste", "good morning", "good evening"]
    prompt_lower = prompt.lower().strip()
    return any(greeting in prompt_lower for greeting in greetings) and len(prompt_lower.split()) <= 2

def format_response(text):
    """Format the response with paragraphs, bullet points, and proper hyperlinks."""
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
    # Enhanced hyperlink matching with full URLs
    final_text = re.sub(r'(https?://[^\s<>]+|www\.[^\s<>]+)', r'<a href="\1" target="_blank">\1</a>', final_text)
    return final_text

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
    prompt: str
    regenerate: bool = False

class ChatResponse(BaseModel):
    response: str

def get_prompt(prompt_type, user_input, session_data, ask_for_more_info=True):
    """Get the prompt from file, and add the specific content depending on the prompt_type"""
    with open("prompts/base_prompt.txt", "r") as f:
        prompt = f.read()
    if prompt_type == 'greeting':
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(await file.read())
                temp_file_path = temp_file.name

            with open(temp_file_path, 'rb') as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
            os.unlink(temp_file_path)
        except Exception as e:
            logger.error(f"Error processing PDF file: {str(e)}. File name: {file.filename}")
            return "Error processing PDF file."
    elif file.content_type.startswith('image/'):
        logger.info(f"File is an image. File name: {file.filename}")
        return "file is an image"
    else:
    return text

@app.post("/chat", response_model=ChatResponse)  # Add response_model here
async def chat_with_law_assistant(request: ChatRequest, file: UploadFile = File(None)):
    # If it's a greeting, customize the prompt for a friendly response.
    # Add the base guidelines to the prompt to maintain consistency.
    global model
    try:
        file_content = ""
        if file:
            try:
                file_content = await process_uploaded_file(file)
            except Exception as e:
                logger.error(f"Error processing uploaded file: {e}")
                file_content = ""  # Ensure file_content is empty in case of an error

        # Load session data
        session_data = load_session(request.session_id)

        # Check for initial welcome message before appending user input
    session_file = os.path.join(SESSION_FOLDER, f"{request.session_id}.json")
    if not os.path.exists(session_file) and not request.prompt.strip() and not request.regenerate:
      assistant_response = "Okay, I'm ready to assist you with your legal questions related to Indian law, including IPC sections, Indian Acts, judgments, passport-related issues, and other relevant topics. I can also help determine legal rights and official government procedures within my area of expertise. Please ask your question."
    else:
        # Conditionally add the user input to the session history
        session_data.append({"role": "user", "text": request.prompt})
        # Check if the last assistant message asked for more information and user said "yes"
        expanded_response = False
        if session_data and len(session_data) >= 2:
            last_assistant_msg = session_data[-2] if session_data[-2]["role"] == "assistant" else None
            if last_assistant_msg and last_assistant_msg["text"].strip().endswith("Would you like more information?") and request.prompt.lower() == "yes":
                expanded_response = True

        base_prompt = """You are a legal assistant named Lexi specializing in Indian law, IPC sections, and related legal topics. You are an attorney and/or criminal lawyer to determine legal rights with full knowledge of IPC section, Indian Acts and government-related official work. Your task is to provide accurate, related IPC section numbers and Indian Acts, judgements, and professional answers to legal questions. If the question is not related to law or related to all above options, politely decline to answer.

Guidelines:
- Provide answers in plain language that is easy to understand.
- Include the disclaimer: "Disclaimer: This information is for educational purposes only and should not be considered legal advice. It is essential to consult with a legal professional for specific guidance regarding your situation."
- If user asks question in local language, assist user in same language.
- Provide source websites or URLs to the user.
- If required for specific legal precedents or case law, provide relevant citations (e.g., case names, court, and year) along with a brief summary of the judgment.
- Format your response with clear paragraphs separated by double newlines and use bullet points (e.g., '* ') for lists or key points.
- You must decide if more information is needed from the user, if so ask politely. 
"""
        # Create a context-specific prompt for expanded responses
        if expanded_response:
            prompt = f'{base_prompt}\n\nThe user previously asked: "{last_user_prompt}". They have responded "yes" to request more information.\nProvide a detailed response with specific IPC sections, relevant Indian Acts, and case law examples (e.g., case names, court, year) related to the topic. Include source websites or URLs.\n\nConversation History:\n{" ".join([f"{msg["role"]}: {msg["text"]}" for msg in session_data])}\n\nUser: yes\nAssistant:'
        else:    
            prompt = f'{base_prompt}\n\nConversation History:\n{" ".join([f"{msg["role"]}: {msg["text"]}" for msg in session_data])}\n\nUser: {request.prompt}\nAssistant:'

        if file_content:
          prompt = f"File content:\n{file_content}\n\n{prompt}"

        try:
            def generate_content():
                return model.generate_content(prompt)
                assistant_response = await retry_request(generate_content)
                assistant_response = format_response(assistant_response.text)
                session_data.append({"role": "assistant", "text": assistant_response})
                save_session(request.session_id, session_data)
                return ChatResponse(response=assistant_response)
        except genai.QuotaExceededError as e:
            raise HTTPException(status_code=429, detail="Quota exceeded. Please check your API plan at https://ai.google.dev/gemini-api/docs/rate-limits.")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error generating content: {str(e)}")
    except genai.QuotaExceededError as e:
        raise HTTPException(status_code=429, detail="Quota exceeded. Please check your API plan at https://ai.google.dev/gemini-api/docs/rate-limits.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating content: {str(e)}")











if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
