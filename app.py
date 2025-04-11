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
import PyPDF2
from werkzeug.utils import secure_filename

# Load environment variables
load_dotenv()
API_KEYS = [
    os.getenv("GEMINI_API_KEY_1"),
    os.getenv("GEMINI_API_KEY_2"),
    os.getenv("GEMINI_API_KEY_3")
]

SESSION_FOLDER = "sessions"
os.makedirs(SESSION_FOLDER, exist_ok=True)
UPLOADS_FOLDER = "uploads"
os.makedirs(UPLOADS_FOLDER, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class APIKeyManager:
    def __init__(self, keys):
        self.keys = keys
        self.index = 0
        self.last_rotation = datetime.now()
        self.usage = {key: 0 for key in keys}
        self.max_requests_per_key = 100
        self.rotation_interval = timedelta(minutes=10)

    def get_model(self):
        if self.usage[self.keys[self.index]] >= self.max_requests_per_key or \
           (datetime.now() - self.last_rotation) > self.rotation_interval:
            self.rotate()
        try:
            genai.configure(api_key=self.keys[self.index])
            self.usage[self.keys[self.index]] += 1
            return genai.GenerativeModel('gemini-1.5-pro')
        except Exception as e:
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

key_manager = APIKeyManager(API_KEYS)
model = key_manager.get_model()

def load_session(session_id):
    session_file = os.path.join(SESSION_FOLDER, f"{session_id}.json")
    if os.path.exists(session_file):
        with open(session_file, "r") as f:
            data = json.load(f)
            return {"history": data.get("history", []), "last_interaction": data.get("last_interaction", {})}
    return {"history": [], "last_interaction": {}}

def save_session(session_id, session_data):
    session_file = os.path.join(SESSION_FOLDER, f"{session_id}.json")
    with open(session_file, "w") as f:
        json.dump({"history": session_data["history"], "last_interaction": session_data["last_interaction"]}, f)

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

    prompt = f"""
    You are a legal assistant specializing in Indian law, IPC section, justice, advocates, lawyers, official Passports related, and judgment-related topics.
    You are an attorney and/or criminal lawyer to determine legal rights with full knowledge of IPC section, Indian Acts and government-related official work.
    Your task is to provide accurate, related IPC section numbers and Indian Acts, judgements, and professional answers to legal questions.
    If the question is not related to law or related to all above options, politely decline to answer.

    Guidelines:
    - Provide answers in plain language that is easy to understand.
    - If user asks question in local language, assist user in same language.
    - Provide source websites or URLs to the user. 
    - If required for specific legal precedents or case law, provide relevant citations (e.g., case names, court, and year) along with a brief summary of the judgment.
    - Format your response with clear paragraphs separated by double newlines and use bullet points (e.g., '* ') for lists or key points.

    {examples}

    Conversation History:
    {" ".join([f"{msg['role']}: {msg['text']}" for msg in session_data['history']])}

    User: {request.prompt}
    Assistant:
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
        new_model = key_manager.get_model()  # Use key manager for rotation
        if new_model:
            model = new_model
            return await chat_with_law_assistant(request)
        raise HTTPException(status_code=500, detail="Sorry, I am unable to process your request at the moment.")

@app.post("/regenerate", response_model=ChatResponse)
async def regenerate_response(request: ChatRequest):
    logger.info(f"New regenerate request: session_id={request.session_id}, prompt={request.prompt}")
    session_data = load_session(request.session_id)
    if not session_data["history"] or not any(msg["role"] == "user" for msg in session_data["history"]):
        raise HTTPException(status_code=400, detail="No previous prompt to regenerate")

    last_prompt = next(msg["text"] for msg in reversed(session_data["history"]) if msg["role"] == "user")
    if not last_prompt:
        raise HTTPException(status_code=400, detail="No previous prompt found")

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

    prompt = f"""
    You are a legal assistant specializing in Indian law, IPC section, justice, advocates, lawyers, official Passports related, and judgment-related topics.
    You are an attorney and/or criminal lawyer to determine legal rights with full knowledge of IPC section, Indian Acts and government-related official work.
    Your task is to provide accurate, related IPC section numbers and Indian Acts, judgements, and professional answers to legal questions.
    If the question is not related to law or related to all above options, politely decline to answer.

    Guidelines:
    - Provide answers in plain language that is easy to understand.
    - If user asks question in local language, assist user in same language.
    - Provide source websites or URLs to the user. 
    - If required for specific legal precedents or case law, provide relevant citations (e.g., case names, court, and year) along with a brief summary of the judgment.
    - Format your response with clear paragraphs separated by double newlines and use bullet points (e.g., '* ') for lists or key points.

    {examples}

    Conversation History:
    {" ".join([f"{msg['role']}: {msg['text']}" for msg in session_data['history']])}

    User: {last_prompt}
    Assistant:
    """

    try:
        response = model.generate_content(prompt)
        assistant_response = format_response(response.text)
        for msg in reversed(session_data["history"]):
            if msg["role"] == "assistant":
                msg["text"] = assistant_response
                break
        save_session(request.session_id, session_data)
        logger.info(f"Regenerate response generated for session_id={request.session_id}")
        return ChatResponse(response=assistant_response)
    except Exception as e:
        logger.error(f"Error in regenerate: {str(e)}")
        new_model = key_manager.get_model()
        if new_model:
            model = new_model
            return await regenerate_response(request)
        raise HTTPException(status_code=500, detail="Sorry, I am unable to regenerate the response at the moment.")

@app.post("/upload")
async def upload_file(session_id: str, file: UploadFile = File(...)):
    logger.info(f"New upload request: session_id={session_id}, file={file.filename}")
    if not session_id:
        raise HTTPException(status_code=400, detail="Session ID is required")

    file_path = os.path.join(UPLOADS_FOLDER, secure_filename(file.filename))
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    if file.filename.lower().endswith('.pdf'):
        try:
            with open(file_path, "rb") as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() or ""
                prompt = f"Analyze the following legal document: {text[:500]}... (truncated). Provide a summary and relevant Indian law insights."
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing PDF: {e}")
    else:
        prompt = f"File {file.filename} uploaded. It is not a PDF, so I cannot process it yet. Would you like more information?"

    session_data = load_session(session_id)
    session_data["history"].append({"role": "user", "text": f"Uploaded {file.filename}"})
    save_session(session_id, session_data)

    try:
        response = model.generate_content(prompt)
        assistant_response = format_response(response.text)
        session_data["history"].append({"role": "assistant", "text": assistant_response})
        save_session(session_id, session_data)
        logger.info(f"Upload response generated for session_id={session_id}")
        return {"message": f"File {file.filename} uploaded successfully", "response": assistant_response}
    except Exception as e:
        logger.error(f"Error in upload: {str(e)}")
        new_model = key_manager.get_model()
        if new_model:
            model = new_model
            return await upload_file(session_id, file)
        raise HTTPException(status_code=500, detail="Sorry, I am unable to process the file at the moment.")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
