from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager
import time
import json
import os
import re
from datetime import datetime, timedelta
import uvicorn

from auth import get_login_url, process_login_response
from rag_pipeline import answer_question, load_retriever
import config
from document_manager import (
    load_documents_registry,
    ingest_single_pdf,
    remove_document_from_vectordb,
    toggle_document_status
)

# Global retriever variable
retriever = None

# Global conversation history storage (per session)
# Stores last 10 Q&A pairs per session token for context
conversation_histories: Dict[str, List[Dict[str, str]]] = {}

# File to persist conversation histories
CONVERSATION_HISTORY_FILE = str(config.BASE_DIR / "conversation_histories.json")

def load_conversation_histories():
    """Load conversation histories from file"""
    global conversation_histories
    try:
        if os.path.exists(CONVERSATION_HISTORY_FILE):
            with open(CONVERSATION_HISTORY_FILE, "r", encoding="utf-8") as f:
                conversation_histories = json.load(f)
    except Exception as e:
        print(f"Error loading conversation histories: {e}")
        conversation_histories = {}

def save_conversation_histories():
    """Save conversation histories to file"""
    try:
        with open(CONVERSATION_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(conversation_histories, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving conversation histories: {e}")

# -------------------------------------------------
# SHORT ANSWER DETECTION FUNCTIONS (from app.py)
# -------------------------------------------------
def is_short_answer_question(question: str) -> bool:
    """Check if the question should have a short answer format (Yes/No/Number)"""
    question_lower = question.lower()
    
    # Patterns that require short answers
    short_answer_patterns = [
        # Holiday related
        r'\bis\s+there\s+(a\s+)?holiday',
        r'holiday\s+on',
        r'how\s+many\s+holidays',
        r'how\s+many\s+paid\s+holidays',
        r'number\s+of\s+holidays',
        # Working day related
        r'\bis\s+(saturday|sunday|monday|tuesday|wednesday|thursday|friday)\s+(a\s+)?working',
        r'\bis\s+it\s+(a\s+)?working',
        # Simple leave questions (paid/available)
        r'\bis\s+(maternity|paternity|casual|sick|annual|earned)\s+leave\s+(paid|available)\s*\??$',
        r'\bis\s+there\s+(maternity|paternity|casual|sick|annual|earned)\s+leave\s*\??$',
        # How many questions
        r'how\s+many\s+(casual|sick|annual|earned|maternity|paternity)\s+leaves?',
        r'how\s+many\s+days\s+of\s+(maternity|paternity|casual|sick|annual|earned)\s+leave',
        r'number\s+of\s+(casual|sick|annual|earned|maternity|paternity)\s+leaves?',
        r'how\s+many\s+leaves?\s+(can|are|do)',
        r'how\s+many\s+days',
        # Simple yes/no without complex conditions
        r'\bis\s+(saturday|sunday)\s+(a\s+)?holiday',
    ]
    
    # Patterns that should NOT be short answer (require detailed explanation)
    complex_patterns = [
        r'without\s+',  # "without prior approval" requires explanation
        r'allowed\s+to\s+',  # "allowed to do X" often needs context
        r'how\s+to\s+',  # Process questions need full explanation
        r'what\s+is\s+the\s+process',
        r'what\s+are\s+the\s+steps',
        r'when\s+can\s+',  # Conditional questions need context
        r'under\s+what\s+',  # Conditional
        r'in\s+case\s+of',  # Conditional
    ]
    
    # Check if it matches complex patterns first
    for pattern in complex_patterns:
        if re.search(pattern, question_lower):
            return False
    
    # Then check if it matches short answer patterns
    for pattern in short_answer_patterns:
        if re.search(pattern, question_lower):
            return True
    return False

def get_one_word_answer(full_answer: str, question: str) -> str:
    """Extract a one-word or short answer (Yes/No/Number) from the full answer"""
    if not full_answer:
        return "N/A"
    
    answer_lower = full_answer.lower()
    question_lower = question.lower()
    
    # Check for negative/no patterns first (higher priority)
    negative_patterns = [
        r'\bno\b',
        r'\bnot\s+(?:allowed|available|paid|a\s+holiday|working)',
        r'\bthere\s+(?:is|are)\s+no\b',
        r'\bthere\s+are\s+not\b',
        r'\bnone\b',
        r'\bzero\b',
        r'\bno\s+public\s+holidays?\b',
        r'\bcannot\b',
        r'\bcan\'t\b',
        r'\bdoes\s+not\b',
        r'\bdoesn\'t\b'
    ]
    
    for pattern in negative_patterns:
        if re.search(pattern, answer_lower):
            # For "how many" questions with negative answer, return 0
            if 'how many' in question_lower or 'number of' in question_lower:
                return "0"
            else:
                return "No"
    
    # Check for positive Yes patterns
    positive_patterns = [
        r'\byes\b',
        r'\bis\s+(?:allowed|available|paid|a\s+(?:working\s+)?day|a\s+holiday)\b',
        r'\bthere\s+is\s+a?\s*(?:public\s+)?(?:holiday|leave)\b',  # matches "there is a public holiday"
        r'\bthere\s+is\s+a\s+public\s+holiday\b',  # explicit match
        r'\bpublic\s+holiday\s+on\b',  # "public holiday on 20th"
        r'\bholiday\s+on\s+\d+',  # "holiday on 20th"
        r'\bemployees?\s+(?:are|can|may)\b',
    ]
    
    for pattern in positive_patterns:
        if re.search(pattern, answer_lower):
            # For yes/no questions
            if question_lower.startswith(('is ', 'are ', 'can ', 'do ', 'does ', 'is there', 'are there')):
                # For date-specific questions, verify the date matches
                if 'on' in question_lower and any(day in question_lower for day in ['1st', '2nd', '3rd'] + [f'{i}th' for i in range(4, 32)]):
                    # Extract the date from the question
                    date_match = re.search(r'(\d+)(?:st|nd|rd|th)', question_lower)
                    if date_match:
                        question_date = date_match.group(1)
                        # Check if this same date appears in the answer confirming it
                        if re.search(rf'\b{question_date}(?:st|nd|rd|th)\b', answer_lower):
                            return "Yes"
                        else:
                            return "No"
                return "Yes"
    
    # For "how many" questions, extract the number with context
    if 'how many' in question_lower or 'number of' in question_lower:
        # Look for numbers in parentheses like "Fifteen (15)"
        paren_match = re.search(r'\((\d+(?:\.\d+)?)\)', full_answer)
        if paren_match:
            return paren_match.group(1)
        
        # Look for phrases like "X holidays", "X days", "X leaves" (including decimals)
        count_patterns = [
            r'\b(\d+(?:\.\d+)?)\s+(?:public\s+)?holidays?\b',
            r'\b(\d+(?:\.\d+)?)\s+days?\b',
            r'\b(\d+(?:\.\d+)?)\s+leaves?\b',
            r'\b(\d+(?:\.\d+)?)\s+(?:casual|sick|earned|annual|maternity|paternity)\b',
            r'\bapproximately\s+(\d+(?:\.\d+)?)\s+(?:earned|casual|sick|annual)\s+leaves?\b',
            r'\beligible\s+for\s+(\d+(?:\.\d+)?)\b',
            r'\b(\d+(?:\.\d+)?)\s+per\s+(?:year|month|annum)\b',
        ]
        
        for pattern in count_patterns:
            match = re.search(pattern, answer_lower)
            if match:
                return match.group(1)
        
        # Look for written numbers with context
        written_numbers = {
            'zero': '0', 'none': '0', 'one': '1', 'two': '2', 'three': '3', 
            'four': '4', 'five': '5', 'six': '6', 'seven': '7', 
            'eight': '8', 'nine': '9', 'ten': '10', 'eleven': '11', 
            'twelve': '12', 'thirteen': '13', 'fourteen': '14', 'fifteen': '15',
            'sixteen': '16', 'seventeen': '17', 'eighteen': '18', 'nineteen': '19',
            'twenty': '20', 'twenty-one': '21', 'twenty-two': '22', 'twenty-three': '23',
            'twenty-four': '24', 'twenty-five': '25', 'thirty': '30'
        }
        
        for word, num in written_numbers.items():
            if re.search(rf'\b{word}\s+(?:holidays?|days?|leaves?)\b', answer_lower):
                return num
            # Also check for written number followed by parentheses with digit
            if re.search(rf'\b{word}\b', answer_lower):
                return num
    
    # Check if answer starts with Yes/No
    if answer_lower.strip().startswith("yes"):
        return "Yes"
    if answer_lower.strip().startswith("no"):
        return "No"
    
    # Default: return first few words
    first_sentence = full_answer.split('.')[0]
    words = first_sentence.split()
    if len(words) <= 3:
        return first_sentence
    return ' '.join(words[:3]) + "..."

# -------------------------------------------------
# ANSWER FORMATTING FUNCTIONS
# -------------------------------------------------

def is_greeting(question: str) -> bool:
    """Check if the question is a greeting"""
    import string
    greetings = ['hi', 'hello', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening',
                 'hola', 'namaste', 'howdy', 'sup', "what's up", 'whats up']
    question_clean = question.strip().lower()
    question_clean = question_clean.translate(str.maketrans('', '', string.punctuation + "()[]{}'\"" ))
    question_clean = ' '.join(question_clean.split())
    return question_clean in greetings or any(question_clean.startswith(g + ' ') for g in greetings)

def is_list_question(question: str) -> bool:
    """Check if the question asks for a list of items"""
    question_lower = question.lower()
    list_patterns = [
        r'\blist\b',
        r'\ball\s+(?:the\s+)?(?:public\s+)?holidays\b',
        r'\bwhat\s+are\s+(?:the\s+)?(?:all\s+)?',
        r'\bshow\s+(?:me\s+)?(?:all|the)\b',
        r'\btell\s+(?:me\s+)?(?:all|the)\b',
        r'\benumerate\b',
        r'\bgive\s+(?:me\s+)?(?:a\s+)?list\b',
    ]
    return any(re.search(pattern, question_lower) for pattern in list_patterns)

def format_list_answer(answer: str) -> str:
    """Format answer to display list items on separate lines"""
    # Check if answer contains numbered items like "1. item 2. item" or "1) item 2) item"
    # Convert inline numbered lists to line-separated format
    
    # Pattern for numbered items like "1. Item 2. Item" or "1) Item 2) Item"
    numbered_pattern = r'(\d+[.\)]\s*[^\d]+?)(?=\d+[.\)]|$)'
    
    # Check if it looks like an inline numbered list
    if re.search(r'\d+[.\)]\s*\w+.*\d+[.\)]', answer):
        items = re.findall(numbered_pattern, answer)
        if items and len(items) > 1:
            formatted_items = [item.strip() for item in items if item.strip()]
            return '\n'.join(formatted_items)
    
    # Pattern for dash or bullet separated items
    if ' - ' in answer and answer.count(' - ') > 2:
        parts = answer.split(' - ')
        # Check if these look like list items (not date ranges)
        if len(parts) > 2:
            return '\n- '.join(parts)
    
    return answer

def strip_document_prefix(answer: str) -> str:
    """Remove 'According to...' prefixes from answers"""
    if not answer:
        return answer
    
    # Patterns to remove from start of answer
    prefixes_to_remove = [
        r'^According to the [^,]+?,\s*',
        r'^According to the [^,]+ document,\s*',
        r'^According to the [^,]+ policy,\s*',
        r'^Based on the [^,]+?,\s*',
        r'^Based on the [^,]+ document,\s*',
        r'^Based on the [^,]+ policy,\s*',
        r'^The [^,]+ document states that\s*',
        r'^The policy states that\s*',
    ]
    
    result = answer
    for pattern in prefixes_to_remove:
        result = re.sub(pattern, '', result, flags=re.IGNORECASE)
    
    # Capitalize first letter if needed
    if result and result[0].islower():
        result = result[0].upper() + result[1:]
    
    return result

def get_meaningful_short_answer(full_answer: str, question: str) -> str:
    """Generate a meaningful short answer (complete sentences, not truncated)"""
    if not full_answer:
        return "No answer available."
    
    # First strip any document prefixes
    answer = strip_document_prefix(full_answer)
    
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', answer)
    
    if not sentences:
        return answer
    
    # For yes/no type questions, include the first 1-2 sentences
    question_lower = question.lower()
    is_yes_no = question_lower.startswith(('is ', 'are ', 'can ', 'do ', 'does ', 'is there', 'are there', 'will ', 'would ', 'should '))
    
    # For "how many" questions, get the sentence with the number
    if 'how many' in question_lower or 'number of' in question_lower:
        for sent in sentences:
            if re.search(r'\d+', sent):
                return sent.strip()
        return sentences[0].strip()
    
    # For yes/no questions, return first sentence
    if is_yes_no:
        return sentences[0].strip()
    
    # For other questions, return first 1-2 complete sentences (up to ~100 words)
    short_answer = sentences[0]
    word_count = len(short_answer.split())
    
    # Add second sentence if first is very short
    if len(sentences) > 1 and word_count < 15:
        short_answer += ' ' + sentences[1]
    
    return short_answer.strip()

# -------------------------------------------------
# LIFESPAN AND APP INITIALIZATION
# -------------------------------------------------

# Lifespan context manager for startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global retriever
    retriever = load_retriever()
    print("✓ Retriever loaded successfully")
    load_conversation_histories()
    print("✓ Conversation histories loaded")
    yield
    # Shutdown - save conversation histories
    save_conversation_histories()
    print("✓ Conversation histories saved")

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="HR Policy Chatbot API", 
    version="1.0.0",
    lifespan=lifespan
)

# Get environment
ENVIRONMENT = os.environ.get("ENVIRONMENT", "development")
PRODUCTION_FRONTEND_URL = os.environ.get("PRODUCTION_FRONTEND_URL", "https://your-app.vercel.app")

# Configure CORS for React frontend - supports both local and production
allowed_origins = [
    "http://localhost:3000", 
    "http://localhost:3001",
]

# Add production URL if set
if PRODUCTION_FRONTEND_URL and PRODUCTION_FRONTEND_URL != "https://your-app.vercel.app":
    allowed_origins.append(PRODUCTION_FRONTEND_URL)

# In production, also allow the Vercel preview URLs
if ENVIRONMENT == "production":
    # Vercel preview deployments have dynamic URLs, so we need a more flexible approach
    # This will be handled by the CORSMiddleware's allow_origin_regex if needed
    pass

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_origin_regex=r"https://.*\.vercel\.app" if ENVIRONMENT == "production" else None,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# File paths
AUTH_SESSION_FILE = str(config.BASE_DIR / "auth_session.json")
CHAT_HISTORY_FILE = str(config.BASE_DIR / "chat_history.json")
SESSION_EXPIRY_DAYS = 7

# Pydantic Models
class QuestionRequest(BaseModel):
    question: str
    session_token: Optional[str] = None
    user: Optional[str] = None  # User's email/username for document filtering

class QuestionResponse(BaseModel):
    answer: str
    short_answer: Optional[str] = None
    confidence: float
    sources: List[Dict[str, Any]]
    response_time: float
    is_short_answer_type: bool
    is_greeting: bool = False
    is_list: bool = False

class DocumentInfo(BaseModel):
    name: str
    active: bool
    owner: Optional[str] = None

class DocumentToggleRequest(BaseModel):
    name: str
    active: bool
    owner: Optional[str] = None

class LoginResponse(BaseModel):
    login_url: str

class SessionResponse(BaseModel):
    authenticated: bool
    user: Optional[Dict[str, Any]] = None

# Helper Functions
def save_auth_session(user_data: dict) -> str:
    """Save authentication session and return session token"""
    try:
        session_token = f"session_{int(time.time() * 1000)}"
        session_data = {
            "token": session_token,
            "user": user_data,
            "expires_at": (datetime.now() + timedelta(days=SESSION_EXPIRY_DAYS)).isoformat()
        }
        with open(AUTH_SESSION_FILE, "w", encoding="utf-8") as f:
            json.dump(session_data, f, indent=2)
        return session_token
    except Exception as e:
        print(f"Error saving auth session: {e}")
        return None

def load_auth_session(token: str = None) -> dict:
    """Load authentication session from file if valid"""
    try:
        if os.path.exists(AUTH_SESSION_FILE):
            with open(AUTH_SESSION_FILE, "r", encoding="utf-8") as f:
                session_data = json.load(f)
            
            # Check if session has expired
            expires_at = datetime.fromisoformat(session_data["expires_at"])
            if datetime.now() < expires_at:
                if token is None or session_data.get("token") == token:
                    return session_data.get("user")
            else:
                clear_auth_session()
    except Exception as e:
        print(f"Error loading auth session: {e}")
    return None

def clear_auth_session():
    """Clear saved authentication session"""
    try:
        if os.path.exists(AUTH_SESSION_FILE):
            os.remove(AUTH_SESSION_FILE)
    except Exception as e:
        print(f"Error clearing auth session: {e}")

def save_chat_history(history: list):
    """Save chat history to file"""
    try:
        with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving chat history: {e}")

def load_chat_history() -> list:
    """Load chat history from file"""
    if os.path.exists(CHAT_HISTORY_FILE):
        try:
            with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return []
    return []

# API Endpoints

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "ok", "message": "HR Policy Chatbot API is running"}

@app.get("/api/auth/login-url", response_model=LoginResponse)
async def get_auth_login_url(request: Request):
    """Get Microsoft OAuth login URL"""
    try:
        # Get origin from request headers for automatic redirect URI detection
        origin = request.headers.get("origin", "")
        login_url = get_login_url(origin)
        return LoginResponse(login_url=login_url)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/auth/callback")
async def auth_callback(request: Request, code: str = None, state: str = None):
    """Handle OAuth callback"""
    try:
        # Create a mock query params object
        class QueryParams:
            def __init__(self, code, state):
                self._params = {}
                if code:
                    self._params['code'] = code
                if state:
                    self._params['state'] = state
            
            def get(self, key, default=None):
                return self._params.get(key, default)
        
        query_params = QueryParams(code, state)
        # Pass state to process_login_response so it can extract the redirect URI
        auth_result = process_login_response(query_params, state)
        
        if auth_result:
            session_token = save_auth_session(auth_result)
            # Redirect to React app with session token
            return JSONResponse({
                "authenticated": True,
                "user": auth_result,
                "session_token": session_token
            })
        else:
            print(f"Authentication failed - auth_result is None")
            raise HTTPException(status_code=401, detail="Authentication failed - could not process login response")
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"Auth callback error: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/auth/session", response_model=SessionResponse)
async def get_session(session_token: Optional[str] = None):
    """Check if session is valid"""
    user = load_auth_session(session_token)
    if user:
        return SessionResponse(authenticated=True, user=user)
    return SessionResponse(authenticated=False)

@app.post("/api/auth/logout")
async def logout():
    """Logout and clear session"""
    clear_auth_session()
    return {"status": "ok", "message": "Logged out successfully"}

@app.get("/api/documents", response_model=List[DocumentInfo])
async def get_documents(user: Optional[str] = None):
    """Get list of uploaded documents for a specific user"""
    try:
        registry = load_documents_registry()
        documents = registry["documents"]
        
        # Filter by user if provided
        if user:
            documents = [doc for doc in documents if doc.get("owner") == user]
        
        return [
            DocumentInfo(name=doc["name"], active=doc.get("active", True), owner=doc.get("owner"))
            for doc in documents
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...), owner: Optional[str] = None):
    """Upload and process a new document for a specific user"""
    try:
        print(f"Received upload request for: {file.filename} from user: {owner}")
        
        # Save uploaded file temporarily
        file_path = config.UPLOADS_DIR / file.filename
        print(f"Saving to: {file_path}")
        
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        print(f"File saved, size: {len(content)} bytes")
        
        # Process the document with owner
        print(f"Processing document with ingest_single_pdf for user: {owner}")
        success = ingest_single_pdf(file_path, file.filename, owner=owner)
        print(f"Processing result: {success}")
        
        if success:
            return {"status": "ok", "message": f"Document {file.filename} uploaded successfully"}
        else:
            print(f"Failed to process {file.filename}")
            raise HTTPException(status_code=500, detail="Failed to process document")
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error uploading document: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/documents/toggle")
async def toggle_document(request: DocumentToggleRequest):
    """Toggle document active status"""
    try:
        toggle_document_status(request.name, request.active, owner=request.owner)
        return {"status": "ok", "message": f"Document {request.name} status updated"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/documents/{document_name}")
async def delete_document(document_name: str, owner: Optional[str] = None):
    """Remove a document from the system"""
    try:
        remove_document_from_vectordb(document_name, owner=owner)
        return {"status": "ok", "message": f"Document {document_name} removed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat/question", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """Process a question and return answer"""
    try:
        start_time = time.time()
        
        print(f"Received question: {request.question}")
        
        # Get or initialize conversation history for this session
        session_token = request.session_token or "default"
        if session_token not in conversation_histories:
            conversation_histories[session_token] = []
        
        # Get last 3 Q&A pairs as conversation context (6 messages)
        history = conversation_histories[session_token]
        conversation_context = []
        for qa in history[-3:]:  # Last 3 Q&A pairs
            conversation_context.append(f"User: {qa['question']}")
            conversation_context.append(f"Assistant: {qa['answer']}")
        
        print(f"Sending {len(conversation_context)//2} Q&A pairs as context")
        
        # Get answer from RAG pipeline - returns a tuple (answer, confidence, sources)
        result = answer_question(request.question, conversation_context=conversation_context, user=request.user)
        
        print(f"RAG result type: {type(result)}")
        
        response_time = time.time() - start_time
        
        # Handle tuple return format from answer_question
        if isinstance(result, tuple):
            answer, confidence, sources = result
            
            # Check if this is a greeting or list question
            question_is_greeting = is_greeting(request.question)
            question_is_list = is_list_question(request.question)
            
            # Strip document prefixes from the full answer
            clean_answer = strip_document_prefix(answer)
            
            # Format list answers properly (each item on new line)
            if question_is_list:
                clean_answer = format_list_answer(clean_answer)
            
            # For greetings and list questions, short_answer = full answer (no read more needed)
            if question_is_greeting or question_is_list:
                short_answer = clean_answer
            else:
                # Generate a meaningful short answer for other questions
                short_answer = get_meaningful_short_answer(answer, request.question)
            
            # Check if it's a yes/no type for one-word answer extraction
            is_short_type = is_short_answer_question(request.question)
            if is_short_type and not question_is_greeting and not question_is_list:
                # For yes/no questions, try to extract one-word answer
                one_word = get_one_word_answer(answer, request.question)
                if one_word and one_word not in ['N/A', 'According to the...']:
                    short_answer = one_word
            
            # Store Q&A in conversation history for this session
            conversation_histories[session_token].append({
                "question": request.question,
                "answer": answer
            })
            
            # Keep only last 10 Q&A pairs per session to manage memory
            if len(conversation_histories[session_token]) > 10:
                conversation_histories[session_token] = conversation_histories[session_token][-10:]
            
            # Save to file periodically (every request for persistence)
            save_conversation_histories()
            
            return QuestionResponse(
                answer=clean_answer,
                short_answer=short_answer,
                confidence=confidence,
                sources=sources if sources else [],
                response_time=response_time,
                is_short_answer_type=is_short_type,
                is_greeting=question_is_greeting,
                is_list=question_is_list
            )
        else:
            # Handle dict format (if ever used)
            question_is_greeting = is_greeting(request.question)
            question_is_list = is_list_question(request.question)
            is_short_type = is_short_answer_question(request.question)
            full_answer = result.get("full_answer", "No answer available")
            clean_answer = strip_document_prefix(full_answer)
            
            if question_is_list:
                clean_answer = format_list_answer(clean_answer)
            
            if question_is_greeting or question_is_list:
                short_answer = clean_answer
            else:
                short_answer = get_meaningful_short_answer(full_answer, request.question)
            
            if is_short_type and not question_is_greeting and not question_is_list:
                one_word = get_one_word_answer(full_answer, request.question)
                if one_word and one_word not in ['N/A', 'According to the...']:
                    short_answer = one_word
            
            return QuestionResponse(
                answer=clean_answer,
                short_answer=short_answer,
                confidence=result.get("confidence", 0.0),
                sources=result.get("sources", []),
                response_time=response_time,
                is_short_answer_type=is_short_type,
                is_greeting=question_is_greeting,
                is_list=question_is_list
            )
    except Exception as e:
        print(f"Error answering question: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/chat/history")
async def get_chat_history():
    """Get chat history"""
    try:
        history = load_chat_history()
        return {"history": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class ClearChatRequest(BaseModel):
    session_token: Optional[str] = None

@app.post("/api/chat/clear")
async def clear_chat(request: ClearChatRequest = None):
    """Clear chat history and conversation context for a session"""
    try:
        # Clear conversation history for this session
        session_token = request.session_token if request else None
        if session_token and session_token in conversation_histories:
            del conversation_histories[session_token]
            save_conversation_histories()
            print(f"Cleared conversation history for session: {session_token}")
        elif session_token is None:
            # Clear all if no token provided (legacy support)
            conversation_histories.clear()
            save_conversation_histories()
        
        if os.path.exists(CHAT_HISTORY_FILE):
            os.remove(CHAT_HISTORY_FILE)
        return {"status": "ok", "message": "Chat history cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the server
if __name__ == "__main__":
    print("Starting HR Policy Chatbot API...")
    print("API will be available at: http://localhost:8000")
    print("API docs will be available at: http://localhost:8000/docs")
    uvicorn.run("api_backend:app", host="0.0.0.0", port=8000, reload=True)
