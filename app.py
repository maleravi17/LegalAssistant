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
    """Format the response with paragraphs and bullet points, removing duplicate endings and fixing hyperlinks."""
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
    # Improved hyperlink matching with proper HTML formatting
    final_text = re.sub(r'(https?://[^\s<>]+)', r'<a href="\1" target="_blank">\1</a>', final_text)
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
    prompt: str

class ChatResponse(BaseModel):
    response: str

@app.post("/chat", response_model=ChatResponse)
async def chat_with_law_assistant(request: ChatRequest):
    global model

    # Load session data
    session_data = load_session(request.session_id)

    # Check for initial welcome message before appending user input
    session_file = os.path.join(SESSION_FOLDER, f"{request.session_id}.json")
    if not os.path.exists(session_file) and not request.prompt.strip():
        assistant_response = "Okay, I'm ready to assist you with your legal questions related to Indian law, including IPC sections, Indian Acts, judgments, passport-related issues, and other relevant topics. I can also help determine legal rights and official government procedures within my area of expertise. Please ask your question."
    else:
        # Add the user input to the session history
        session_data.append({"role": "user", "text": request.prompt})

        # Two-shot prompting examples
        examples = """
        Example 1:
        User: What is the difference between civil law and criminal law?
        Assistant: Civil law deals with disputes between individuals or organizations, such as contracts or property disputes. Criminal law, on the other hand, involves actions that are harmful to society and are prosecuted by the state, such as theft or assault.

        Example 2:
        User: Can a lawyer represent both parties in a case?
        Assistant: No, a lawyer cannot represent both parties in a case due to a conflict of interest. It is unethical and prohibited by legal professional standards.

        Example 3:
        User: Explain the process of filing a lawsuit in civil court.
        Assistant: Sure! Here's a step-by-step explanation:
        1. **Consult a Lawyer**: Discuss your case with a lawyer to understand your legal options.
        2. **Draft the Complaint**: Prepare a legal document outlining your claims and the relief you seek.
        3. **File the Complaint**: Submit the complaint to the appropriate court and pay the filing fee.
        4. **Serve the Defendant**: Notify the defendant about the lawsuit by serving them the complaint.
        5. **Await Response**: The defendant has a specified time to respond to the complaint.
        6. **Discovery Phase**: Both parties exchange information and evidence related to the case.
        7. **Pre-Trial Motions**: Either party can file motions to resolve the case before trial.
        8. **Trial**: If the case proceeds to trial, both parties present their arguments and evidence.
        9. **Judgment**: The judge or jury delivers a verdict.
        10. **Appeal**: If either party is dissatisfied, they can appeal the decision.
        """

        # Check if the last assistant message asked for more information and user said "yes"
        expanded_response = False
        if session_data and len(session_data) >= 2:
            last_assistant_msg = session_data[-2] if session_data[-2]["role"] == "assistant" else None
            if last_assistant_msg and last_assistant_msg["text"].strip().endswith("Would you like more information?") and request.prompt.lower() == "yes":
                expanded_response = True

        # Create a context-specific prompt
        if expanded_response:
            last_user_prompt = session_data[-3]["text"] if len(session_data) >= 3 and session_data[-3]["role"] == "user" else "general legal query"
            prompt = f"""
            You are a legal assistant specializing in Indian law, IPC section, justice, advocates, lawyers, official Passports related, and judgment-related topics.
            You are an attorney and/or criminal lawyer to determine legal rights with full knowledge of IPC section, Indian Acts and government-related official work.
            The user previously asked: "{last_user_prompt}". They have responded "yes" to request more information.
            Provide a detailed response with specific IPC sections, relevant Indian Acts, and case law examples (e.g., case names, court, year) related to the topic. Include source websites or URLs.

            Guidelines:
            - Provide answers in plain language that is easy to understand.
            - Include the disclaimer: "Disclaimer: This information is for educational purposes only and should not be considered legal advice. It is essential to consult with a legal professional for specific guidance regarding your situation."
            - Format your response with clear paragraphs separated by double newlines and use bullet points (e.g., '* ') for lists or key points.
            - Do not end with "Would you like more information?" since the user has already indicated interest. Ensure this instruction is strictly followed.

            {examples}

            Conversation History:
            {" ".join([f"{msg['role']}: {msg['text']}" for msg in session_data])}

            User: yes
            Assistant:
            """
            def generate_content():
                return model.generate_content(prompt)
            response = retry_request(generate_content)
            assistant_response = format_response(response.text)
        else:
            if is_greeting(request.prompt):
                prompt = f"""
                You are a legal assistant named Lexi specializing in Indian law, IPC sections, and related legal topics.
                The user has greeted you with: "{request.prompt}". Provide a friendly response without legal details, disclaimer, or any ending question.

                Guidelines:
                - Keep the response concise and friendly.
                - Do not include the disclaimer or "Would you like more information?".

                Conversation History:
                {" ".join([f"{msg['role']}: {msg['text']}" for msg in session_data])}

                User: {request.prompt}
                Assistant:
                """
            else:
                prompt = f"""
                You are a legal assistant specializing in Indian law, IPC section, justice, advocates, lawyers, official Passports related, and judgment-related topics.
                You are an attorney and/or criminal lawyer to determine legal rights with full knowledge of IPC section, Indian Acts and government-related official work.
                Your task is to provide accurate, related IPC section numbers and Indian Acts, judgements, and professional answers to legal questions.
                If the question is not related to law or related to all above options, politely decline to answer.

                Guidelines:
                - Provide answers in plain language that is easy to understand.
                - Include the disclaimer: "Disclaimer: This information is for educational purposes only and should not be considered legal advice. It is essential to consult with a legal professional for specific guidance regarding your situation."
                - If user asks question in local language, assist user in same language.
                - Provide source websites or URLs to the user.
                - If required for specific legal precedents or case law, provide relevant citations (e.g., case names, court, and year) along with a brief summary of the judgment.
                - Format your response with clear paragraphs separated by double newlines and use bullet points (e.g., '* ') for lists or key points.
                - End your response with "Would you like more information?".

                {examples}

                Conversation History:
                {" ".join([f"{msg['role']}: {msg['text']}" for msg in session_data])}

                User: {request.prompt}
                Assistant:
                """
            def generate_content():
                return model.generate_content(prompt)
            response = retry_request(generate_content)
            assistant_response = format_response(response.text)

    # Add the assistant's response to the session history (only if generated)
    if 'assistant_response' in locals() and assistant_response:
        session_data.append({"role": "assistant", "text": assistant_response})

    # Save the updated session data
    save_session(request.session_id, session_data)

    return ChatResponse(response=assistant_response if 'assistant_response' in locals() else "")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
