from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()



BASE_DIR = Path(__file__).resolve().parent

USE_GROQ_API = True

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

GROQ_MODEL = "llama-3.3-70b-versatile"

MODEL_PATH = BASE_DIR / "models/phi-3-mini-4k-instruct-q4.gguf"


N_GPU_LAYERS = 0 

N_THREADS = 12

N_CTX = 1024


TOP_K = 8  

CHROMA_DIR = BASE_DIR / "chroma_for_policychatbot"
COLLECTION_NAME = "hr_policies"


MAX_TOKENS_SHORT = 300  
MAX_TOKENS_LONG = 600   


TEMPERATURE = 0.1

CACHE_FILE = BASE_DIR / "qa_cache.json"
SIMILARITY_THRESHOLD = 0.85  


UPLOADS_DIR = BASE_DIR / "uploads"
MAX_UPLOAD_SIZE_MB = 200  
