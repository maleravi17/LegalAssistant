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
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
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

def should_offer_more_info(response: str, prompt: str) -> bool:
    """Determine if 'Would you like more information?' should be added."""
    # Skip for greetings
    if is_greeting(prompt):
        logger.debug("Skipping more info prompt for greeting")
        return False
    # Skip for file upload responses
    if "Uploaded image" in response or "Error processing PDF file" in response or "No text could be extracted" in response:
        logger.debug("Skipping more info prompt for file upload response")
        return False
    # Count words, excluding disclaimer
    disclaimer = "Disclaimer: This information is for educational purposes only and should not be considered legal advice. It is essential to consult with a legal professional for specific guidance regarding your situation."
    response_clean = response.replace(disclaimer, '').strip()
    word_count = len(response_clean.split())
    # Add if response is short (less than 80 words)
    if word_count < 80:
        logger.debug(f"Adding more info prompt due to short response: {word_count} words")
        return True
    # Add if response lacks specific legal details
    has_details = any(keyword in response_clean.lower() for keyword in [
        "section ", "act ", "case ", "judgment ", "court ", "url", "http", 
        "ipc ", "article ", "law ", "legal ", "ruling ", "precedent ", "citation "
    ])
    if not has_details:
        logger.debug("Adding more info prompt due to lack of specific legal details")
        return True
    logger.debug("Skipping more info prompt due to sufficient detail")
    return False

def format_response(text, prompt):
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
    final_text = '\n\n'.join(formatted)
    
    # Remove any follow-up questions added by Gemini
    follow_up_patterns = [
        r'Would you like more information\?\s*',
        r'Do you want more details\?\s*',
        r'Would you like additional information\?\s*',
        r'Need more info\?\s*',
        r'Would you like to know more\?\s*'
    ]
    for pattern in follow_up_patterns:
        if re.search(pattern, final_text, re.IGNORECASE):
            logger.debug(f"Removing Gemini-added follow-up question matching: {pattern}")
            final_text = re.sub(pattern, '', final_text, flags=re.IGNORECASE).strip()
    
    # Handle disclaimers
    disclaimer = "Disclaimer: This information is for educational purposes only and should not be considered legal advice. It is essential to consult with a legal professional for specific guidance regarding your situation."
    disclaimer_pattern = re.escape(disclaimer)
    if re.search(disclaimer_pattern, final_text, re.IGNORECASE):
        logger.debug("Existing disclaimer found in response, removing it")
        final_text = re.sub(rf'{disclaimer_pattern}\s*', '', final_text, flags=re.IGNORECASE).strip()
    else:
        logger.debug("No existing disclaimer found in response")
    
    # Add hyperlinks
    final_text = re.sub(r'(https?://[^\s<>]+|www\.[^\s<>]+)', r'<a href="\1" target="_blank">\1</a>', final_text)
    
    # Add single disclaimer
    final_text += f"\n\n{disclaimer}"
    logger.debug("Appended single disclaimer to response")
    
    # Add "Would you like more information?" if appropriate
    if should_offer_more_info(final_text, prompt):
        final_text += "\n\nWould you like more information?"
        logger.debug("Appended 'Would you like more information?' to response")
    
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

class ChatResponse(BaseModel):
    response: str

@app.post("/chat", response_model=ChatResponse)
async def chat_with_law_assistant(
    session_id: str = Form(...),
    prompt: str = Form(...),
    regenerate: bool = Form(False),
    file: UploadFile = File(None)
):
    global model
    logger.debug(f"Received /chat request: session_id={session_id}, prompt={prompt}, regenerate={regenerate}, file={file.filename if file else None}")
    try:
        file_content = ""
        if file:
            file_content = await process_uploaded_file(file)

        # Load session data
        session_data = load_session(session_id)

        # Check for initial welcome message
        session_file = os.path.join(SESSION_FOLDER, f"{session_id}.json")
        if not os.path.exists(session_file) and not prompt.strip() and not regenerate:
            assistant_response = "Okay, I'm ready to assist you with your legal questions related to Indian law, including IPC sections, Indian Acts, judgments, passport-related issues, and other relevant topics. I can also help determine legal rights and official government procedures within my area of expertise. Please ask your question."
            logger.debug("Returning initial welcome message")
            return ChatResponse(response=assistant_response)

        # Handle greetings
        if is_greeting(prompt):
            assistant_response = "Hello! I'm Lexi, your legal assistant for Indian law. How can I help you today?"
            session_data.append({"role": "user", "text": prompt})
            session_data.append({"role": "assistant", "text": assistant_response})
            save_session(session_id, session_data)
            logger.debug("Handled greeting prompt")
            return ChatResponse(response=assistant_response)

        # Append user input to session history
        session_data.append({"role": "user", "text": prompt})

        # Check for expanded response
        expanded_response = False
        last_user_prompt = prompt
        if session_data and len(session_data) >= 2:
            last_assistant_msg = session_data[-2] if session_data[-2]["role"] == "assistant" else None
            if last_assistant_msg and last_assistant_msg["text"].strip().endswith("Would you like more information?") and prompt.lower() == "yes":
                expanded_response = True
                last_user_prompt = session_data[-3]["text"] if len(session_data) >= 3 and session_data[-3]["role"] == "user" else prompt

        # Load base prompt
        with open("prompts/base_prompt.txt", "r") as f:
            base_prompt = f.read()

        # Construct conversation history
        history = " ".join([f"{msg['role']}: {msg['text']}" for msg in session_data])

        # Construct prompt
        if expanded_response:
            prompt_text = f"{base_prompt}\n\nThe user previously asked: \"{last_user_prompt}\". They have responded \"yes\" to request more information.\nProvide a detailed response with specific IPC sections, relevant Indian Acts, and case law examples (e.g., case names, court, year) related to the topic. Include source websites or URLs.\n\nConversation History:\n{history}\n\nUser: yes\nAssistant:"
        else:
            prompt_text = f"{base_prompt}\n\nConversation History:\n{history}\n\nUser: {prompt}\nAssistant:"

        if file_content:
            prompt_text = f"File content:\n{file_content}\n\n{prompt_text}"

        # Generate response
        async def generate_content():
            logger.debug(f"Generating content for prompt: {prompt_text[:50]}...")
            response = model.generate_content(prompt_text)
            logger.debug("Content generated successfully")
            return response

        try:
            response = await retry_request(generate_content)
            assistant_response = format_response(response.text, prompt)
            session_data.append({"role": "assistant", "text": assistant_response})
            save_session(session_id, session_data)
            logger.debug("Response generated and session saved")
            return ChatResponse(response=assistant_response)
        except genai.QuotaExceededError:
            try:
                model = rotate_key()
                response = await retry_request(generate_content)
                assistant_response = format_response(response.text, prompt)
                session_data.append({"role": "assistant", "text": assistant_response})
                save_session(session_id, session_data)
                logger.debug("Response generated after key rotation")
                return ChatResponse(response=assistant_response)
            except genai.QuotaExceededError:
                logger.error("Quota exceeded for all API keys")
                raise HTTPException(status_code=429, detail="Quota exceeded for all API keys. Please check your API plan at https://ai.google.dev/gemini-api/docs/rate-limits.")
        except Exception as e:
            logger.error(f"Error generating content: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error generating content: {str(e)}")

    except Exception as e:
        logger.error(f"Unexpected error in /chat: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.post("/regenerate", response_model=ChatResponse)
async def regenerate_response(
    session_id: str = Form(...),
    prompt: str = Form(...),
    regenerate: bool = Form(True)
):
    """Regenerate a response for the given prompt."""
    logger.debug(f"Received /regenerate request: session_id={session_id}, prompt={prompt}, regenerate={regenerate}")
    if not prompt:
        logger.error("Prompt is required for regeneration")
        raise HTTPException(status_code=400, detail="Prompt is required for regeneration.")
    # Call the chat endpoint with regenerate=True
    return await chat_with_law_assistant(
        session_id=session_id,
        prompt=prompt,
        regenerate=regenerate,
        file=None
    )

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
