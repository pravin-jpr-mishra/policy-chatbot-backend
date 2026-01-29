import requests
import re
import json
import os
import numpy as np
from datetime import datetime
from typing import Optional, Tuple, List, Dict
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from safety import validate_answer
from document_manager import get_active_document_names, get_all_document_names
import config 

if config.USE_GROQ_API:
    from groq import Groq
else:
    from llama_cpp import Llama

CHROMA_DIR = str(config.CHROMA_DIR)
COLLECTION_NAME = config.COLLECTION_NAME
MODEL_PATH = str(config.MODEL_PATH)
CACHE_FILE = str(config.CACHE_FILE)
SIMILARITY_THRESHOLD = config.SIMILARITY_THRESHOLD
TOP_K = config.TOP_K
_retriever = None
_embeddings = None
_qa_cache = None
_llama_model = None
_groq_client = None

# Corporate acronyms/shortforms dictionary
CORPORATE_ACRONYMS = {
    # Work arrangements
    'wfh': 'work from home',
    'wfo': 'work from office',
    'rto': 'return to office',
    'hybrid': 'hybrid work',
    'remote': 'remote work',
    
    # Leave types
    'cl': 'casual leave',
    'sl': 'sick leave',
    'el': 'earned leave',
    'pl': 'privilege leave',
    'ml': 'maternity leave',
    'ptl': 'paternity leave',
    'lop': 'loss of pay',
    'lwp': 'leave without pay',
    'al': 'annual leave',
    'comp off': 'compensatory off',
    'compoff': 'compensatory off',
    
    # HR terms
    'hr': 'human resources',
    'hrbp': 'hr business partner',
    'pf': 'provident fund',
    'epf': 'employee provident fund',
    'esi': 'employee state insurance',
    'ctc': 'cost to company',
    'nda': 'non disclosure agreement',
    'bgv': 'background verification',
    'kyc': 'know your customer',
    'posh': 'prevention of sexual harassment',
    'icc': 'internal complaints committee',
    
    # Performance
    'kpi': 'key performance indicator',
    'kra': 'key result area',
    'pip': 'performance improvement plan',
    'appraisal': 'performance appraisal',
    
    # Time related
    'ot': 'overtime',
    'flexi': 'flexible timing',
    'shift': 'shift timing',
    
    # Titles
    'ceo': 'chief executive officer',
    'cto': 'chief technology officer',
    'cfo': 'chief financial officer',
    'coo': 'chief operating officer',
    'vp': 'vice president',
    'avp': 'assistant vice president',
    'mgr': 'manager',
    'sr': 'senior',
    'jr': 'junior',
    
    # General
    'f2f': 'face to face',
    'eod': 'end of day',
    'eow': 'end of week',
    'asap': 'as soon as possible',
    'fyi': 'for your information',
    'ppt': 'presentation',
    'doc': 'document',
    'dept': 'department',
    'org': 'organization',
    'emp': 'employee',
    'mgmt': 'management',
    'admin': 'administration',
    'onboarding': 'employee onboarding',
    'offboarding': 'employee offboarding',
    'probation': 'probation period',
    'confirmation': 'employment confirmation',
    'resignation': 'resignation',
    'fnf': 'full and final settlement',
    'relieving': 'relieving letter',
    'experience': 'experience letter',
}

def expand_acronyms(query: str) -> str:
    """Expand corporate acronyms in the query to their full forms"""
    query_lower = query.lower()
    words = query_lower.split()
    
    expanded_words = []
    for word in words:
        # Remove punctuation for matching
        clean_word = re.sub(r'[^\w]', '', word)
        if clean_word in CORPORATE_ACRONYMS:
            expanded_words.append(CORPORATE_ACRONYMS[clean_word])
        else:
            expanded_words.append(word)
    
    expanded_query = ' '.join(expanded_words)
    
    # If query was just an acronym, make it a proper question
    if len(words) == 1 and words[0].lower() in CORPORATE_ACRONYMS:
        expanded_query = f"What is the {CORPORATE_ACRONYMS[words[0].lower()]} policy?"
    
    return expanded_query


def expand_version_query(query: str) -> str:
    """
    Expand version-related queries to include table-matching terms.
    This helps semantic search find version history tables.
    """
    query_lower = query.lower()
    
    # Check if this is a version-related query
    version_patterns = [
        r'version\s*(\d+\.?\d*)',  # version 1.0, version 1, etc.
        r'v(\d+\.?\d*)',           # v1.0, v1, etc.
        r'who\s+(?:is|was)\s+(?:the\s+)?author',
        r'author\s+of',
        r'version\s+history',
        r'document\s+version',
        r'created\s+(?:by|on)',
        r'reviewed\s+by',
        r'approved\s+by'
    ]
    
    is_version_query = any(re.search(p, query_lower) for p in version_patterns)
    
    if not is_version_query:
        return query
    
    # Extract version number if present
    version_match = re.search(r'version\s*(\d+\.?\d*)|v(\d+\.?\d*)', query_lower)
    version_num = None
    if version_match:
        version_num = version_match.group(1) or version_match.group(2)
    
    # Build an expanded query with terms that match table row format
    # Key insight: Queries like "1.0 Reema Jain author" work better than 
    # "who is the author of version 1.0" for finding table chunks
    
    expanded_parts = []
    
    if version_num:
        # Add version number prominently - this is critical for matching
        expanded_parts.append(version_num)
        expanded_parts.append(f"Version: {version_num}")
    
    if 'author' in query_lower or 'who' in query_lower:
        # Add "author" as a standalone term for table matching
        expanded_parts.append("author")
    
    if 'date' in query_lower or 'when' in query_lower:
        expanded_parts.append("date")
    
    if 'reviewed' in query_lower:
        expanded_parts.append("Reviewed By")
    
    if 'approved' in query_lower:
        expanded_parts.append("Approved By")
    
    if 'change' in query_lower or 'history' in query_lower:
        expanded_parts.append("Summary of Changes")
    
    # Add the original query parts
    expanded_parts.append(query)
    
    # Combine: put version/key terms first for better matching
    expanded = ' '.join(expanded_parts)
    
    expanded = ' '.join(expanded_parts)
    print(f"DEBUG: Expanded version query: '{query}' -> '{expanded}'")
    return expanded
    return expanded_query

def is_valid_query(query: str) -> tuple[bool, str]:
   
    query = query.strip().lower()

    if len(query) < 3:
        return False, "Please enter a more detailed question (at least 3 characters)."

    common_acronyms = list(CORPORATE_ACRONYMS.keys())

    words = query.split()
    if all(word in common_acronyms for word in words):
        return True, "" 
    
    if not re.search(r'[aeiou]', query):
        if not any(word in common_acronyms for word in words):
            return False, "Please enter a valid question with actual words."
    
    if len(words) == 1 and len(query) > 15:
        return False, "Please enter a proper question with meaningful words."
    
    if re.search(r'(.)\1{4,}', query):
        return False, "Please enter a meaningful question instead of repeated characters."
    
    gibberish_patterns = [
        r'asdf',
        r'qwert',
        r'zxcv',
        r'hjkl',
        r'jklm'
    ]

    pattern_count = sum(1 for pattern in gibberish_patterns if re.search(pattern, query))
    if pattern_count >= 2:
        return False, "Please enter a proper question about HR policies."
    
    if len(words) < 2:
        common_hr_terms = ['leave', 'holiday', 'vacation', 'policy', 'salary', 'benefits', 
                          'insurance', 'resignation', 'notice', 'probation', 'overtime',
                          'attendance', 'remote', 'work', 'hours', 'dress', 'code'] + common_acronyms
        if query not in common_hr_terms:
            return False, "Please ask a complete question (e.g., 'What is the leave policy?')."
    

    consonants = len(re.findall(r'[bcdfghjklmnpqrstvwxyz]', query))
    vowels = len(re.findall(r'[aeiou]', query))
    
    if vowels > 0 and consonants / vowels > 5:
        if not any(word in common_acronyms for word in words):
            return False, "Please enter a valid question with proper words."
    
    if not re.search(r'[a-z]', query):
        return False, "Please enter a question with words."
    
    return True, ""

def load_embeddings():
    global _embeddings
    if _embeddings is not None:
        return _embeddings
    _embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return _embeddings

def load_qa_cache():
    global _qa_cache
    if _qa_cache is not None:
        return _qa_cache
    
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                _qa_cache = json.load(f)
        except Exception as e:
            print(f"Error loading cache: {e}")
            _qa_cache = {"questions": [], "answers": [], "embeddings": [], "sources": [], "confidences": []}
    else:
        _qa_cache = {"questions": [], "answers": [], "embeddings": [], "sources": [], "confidences": []}
    
    return _qa_cache

def save_qa_cache():
    global _qa_cache
    if _qa_cache is None:
        return
    
    try:
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(_qa_cache, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving cache: {e}")

def remove_document_from_cache(document_name: str):
    """Remove all cached Q&A entries that reference a specific document"""
    global _qa_cache
    cache = load_qa_cache()
    
    if not cache["questions"]:
        return 0
    
    # Find indices of entries that reference this document
    indices_to_remove = []
    for idx, sources in enumerate(cache.get("sources", [])):
        # Check if any source in this entry references the document
        if sources:
            for source in sources:
                if isinstance(source, dict) and source.get("source") == document_name:
                    indices_to_remove.append(idx)
                    break
    
    if not indices_to_remove:
        print(f"No cached entries found for document: {document_name}")
        return 0
    
    # Remove entries in reverse order to maintain correct indices
    for idx in sorted(indices_to_remove, reverse=True):
        for key in ["questions", "answers", "embeddings", "sources", "confidences"]:
            if idx < len(cache.get(key, [])):
                cache[key].pop(idx)
    
    _qa_cache = cache
    save_qa_cache()
    print(f"Removed {len(indices_to_remove)} cached entries for document: {document_name}")
    return len(indices_to_remove)

def filter_relevant_sources(answer: str, docs: list, sources: list) -> list:
    """
    Filter sources to only include those whose content actually contributed to the answer.
    Uses keyword matching and content overlap to determine relevance.
    """
    if not answer or not docs or not sources:
        return sources
    
    answer_lower = answer.lower()
    relevant_sources = []
    seen_source_pages = set()  # Track unique source+page combinations
    source_scores = []  # Track scores for all sources
    
    # Extract significant words from the answer (excluding common words)
    stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                  'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                  'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
                  'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as',
                  'into', 'through', 'during', 'before', 'after', 'above', 'below',
                  'between', 'under', 'again', 'further', 'then', 'once', 'here',
                  'there', 'when', 'where', 'why', 'how', 'all', 'each', 'few',
                  'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
                  'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just',
                  'and', 'but', 'if', 'or', 'because', 'until', 'while', 'this',
                  'that', 'these', 'those', 'am', 'your', 'you', 'it', 'its', 'i',
                  'me', 'my', 'we', 'our', 'they', 'their', 'what', 'which', 'who',
                  'also', 'any', 'about', 'like', 'based', 'according', 'per',
                  'ensuring', 'provides', 'define', 'defined', 'document', 'policy',
                  'support', 'application', 'operations', 'communication'}
    
    # Extract answer keywords (words with 4+ chars, not in stop words) - more strict
    answer_words = set()
    for word in re.findall(r'\b[a-z]{4,}\b', answer_lower):
        if word not in stop_words:
            answer_words.add(word)
    
    # Also extract numbers and specific patterns from answer
    answer_numbers = set(re.findall(r'\b\d+(?:\.\d+)?%?\b', answer_lower))
    
    # Extract specific phrases from answer (more than just common words)
    # These are strong indicators of source relevance
    specific_phrases = []
    # Look for quoted terms, proper nouns patterns, or technical terms
    phrase_matches = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', answer)  # Proper nouns
    specific_phrases.extend([p.lower() for p in phrase_matches if len(p) > 5])
    
    for idx, doc in enumerate(docs):
        if idx >= len(sources):
            break
            
        doc_content = doc.page_content.lower()
        source_info = sources[idx]
        
        # Create unique key for this source+page
        source_key = f"{source_info.get('source', '')}_{source_info.get('page', '')}"
        
        # Skip if we already have this source+page combination
        if source_key in seen_source_pages:
            continue
        
        # Calculate relevance score
        relevance_score = 0
        
        # Check keyword overlap
        doc_words = set(re.findall(r'\b[a-z]{4,}\b', doc_content))
        common_words = answer_words.intersection(doc_words)
        
        # Higher weight for more specific/longer words
        for word in common_words:
            if len(word) >= 8:
                relevance_score += 5  # Very specific words
            elif len(word) >= 6:
                relevance_score += 3
            elif len(word) >= 4:
                relevance_score += 1
        
        # Check number overlap (important for policies with specific values)
        doc_numbers = set(re.findall(r'\b\d+(?:\.\d+)?%?\b', doc_content))
        common_numbers = answer_numbers.intersection(doc_numbers)
        relevance_score += len(common_numbers) * 8  # Numbers are strong indicators
        
        # Check for specific phrase matches
        for phrase in specific_phrases:
            if phrase in doc_content:
                relevance_score += 15
        
        # Check for exact sentence fragments (very strong indicator)
        answer_sentences = [s.strip() for s in re.split(r'[.!?]', answer_lower) if len(s.strip()) > 20]
        for sentence in answer_sentences:
            # Check if significant portion of sentence appears in doc
            sentence_words = [w for w in sentence.split() if len(w) > 3 and w not in stop_words]
            if len(sentence_words) >= 3:
                matches = sum(1 for w in sentence_words if w in doc_content)
                if matches >= len(sentence_words) * 0.7:  # 70% of significant words match
                    relevance_score += 20
        
        source_scores.append((idx, source_info, relevance_score, source_key))
        print(f"DEBUG: Source '{source_info.get('source')}' page {source_info.get('page')} - relevance score: {relevance_score}")
    
    # Sort by score descending
    source_scores.sort(key=lambda x: x[2], reverse=True)
    
    # Only include sources that meet minimum threshold AND are within reasonable range of best score
    if source_scores:
        best_score = source_scores[0][2]
        min_threshold = 10  # Minimum absolute score
        relative_threshold = best_score * 0.4  # Must be at least 40% of best score
        
        for idx, source_info, score, source_key in source_scores:
            if score >= min_threshold and score >= relative_threshold:
                if source_key not in seen_source_pages:
                    relevant_sources.append(source_info)
                    seen_source_pages.add(source_key)
            else:
                print(f"DEBUG: Filtered out '{source_info.get('source')}' page {source_info.get('page')} - score {score} below threshold (min={min_threshold}, relative={relative_threshold:.1f})")
    
    # If no sources passed the filter, return only the top source
    if not relevant_sources and source_scores:
        relevant_sources.append(source_scores[0][1])
        print("DEBUG: No highly relevant sources found, keeping top scoring source as fallback")
    
    return relevant_sources

def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def find_similar_question(question: str) -> Optional[Tuple[str, float, List[Dict], float]]:

    cache = load_qa_cache()
    
    if not cache["questions"]:
        return None
    
    # Time-sensitive words that should NOT match cached answers
    # because "today" and "tomorrow" refer to different actual days each time
    time_sensitive_words = ["today", "tomorrow", "yesterday", "now", "current", "this week", "next week", "this month"]
    
    question_lower = question.lower().strip()
    
    # If question contains time-sensitive words, skip cache entirely
    if any(word in question_lower for word in time_sensitive_words):
        print("DEBUG: Skipping cache - question contains time-sensitive words")
        return None
    
    # Extract version numbers from the question (e.g., "1.0", "1.1", "v2.0", "version 1.0")
    version_patterns = [
        r'\bv?(?:ersion\s*)?(\d+(?:\.\d+)+)\b',  # Matches "1.0", "v1.0", "version 1.0", "1.2.3"
        r'\bv(\d+)\b',  # Matches "v1", "v2"
    ]
    
    def extract_versions(text):
        versions = []
        for pattern in version_patterns:
            versions.extend(re.findall(pattern, text.lower()))
        return versions
    
    question_versions = extract_versions(question_lower)
    question_has_version = len(question_versions) > 0
    
    # Extract dates from the question (e.g., "20th october", "19th january", "15th", etc.)
    date_patterns = [
        r'\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?(?:january|february|march|april|may|june|july|august|september|october|november|december)\b',
        r'\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2})(?:st|nd|rd|th)?\b',
        r'\b(\d{1,2})(?:st|nd|rd|th)\b',  # Just date numbers
        r'\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b',  # Date formats like 20/10/2024
        r'\b(\d{4})-(\d{1,2})-(\d{1,2})\b',  # Date formats like 2024-10-20
    ]
    
    question_has_date = any(re.search(pattern, question_lower) for pattern in date_patterns)
    
    embeddings_model = load_embeddings()
    question_embedding = embeddings_model.embed_query(question_lower)
    
    max_similarity = 0.0
    best_match_idx = -1
    
    for idx, cached_embedding in enumerate(cache["embeddings"]):
        # Also skip cached questions that had time-sensitive words
        cached_question = cache["questions"][idx].lower() if idx < len(cache["questions"]) else ""
        if any(word in cached_question for word in time_sensitive_words):
            continue
        
        # If the current question has a version number, check if cached question has the same version
        if question_has_version:
            cached_versions = extract_versions(cached_question)
            
            # If both have version numbers, they must match exactly
            if cached_versions:
                if question_versions != cached_versions:
                    print(f"DEBUG: Skipping cache - version mismatch. Question versions: {question_versions}, Cached versions: {cached_versions}")
                    continue
        
        # If the current question has a date, check if cached question has the same exact date
        if question_has_date:
            cached_has_date = any(re.search(pattern, cached_question) for pattern in date_patterns)
            
            # If both have dates, they must match exactly
            if cached_has_date:
                # Extract all date components from both questions
                question_dates = []
                cached_dates = []
                
                for pattern in date_patterns:
                    question_dates.extend(re.findall(pattern, question_lower))
                    cached_dates.extend(re.findall(pattern, cached_question))
                
                # Flatten tuples to strings for comparison
                question_dates_str = [str(d) for d in question_dates]
                cached_dates_str = [str(d) for d in cached_dates]
                
                # If dates don't match, skip this cached question
                if question_dates_str != cached_dates_str:
                    print(f"DEBUG: Skipping cache - date mismatch. Question dates: {question_dates_str}, Cached dates: {cached_dates_str}")
                    continue
            
        similarity = cosine_similarity(question_embedding, cached_embedding)
        if similarity > max_similarity:
            max_similarity = similarity
            best_match_idx = idx
    
    if max_similarity >= SIMILARITY_THRESHOLD and best_match_idx >= 0:
        return (
            cache["answers"][best_match_idx],
            cache["confidences"][best_match_idx],
            cache["sources"][best_match_idx],
            max_similarity
        )
    
    return None

def cache_qa(question: str, answer: str, confidence: float, sources: List[Dict]):
  
    if "could not find" in answer.lower() or "please enter" in answer.lower():
        return

    if confidence < 0.3:
        return
    
    # Don't cache time-sensitive questions - they change meaning each day
    time_sensitive_words = ["today", "tomorrow", "yesterday", "now", "current", "this week", "next week", "this month"]
    if any(word in question.lower() for word in time_sensitive_words):
        print("DEBUG: Not caching - question contains time-sensitive words")
        return
    
    # Don't cache answers that contain dynamic date references
    # These patterns indicate the answer contains dates that will become stale
    answer_lower = answer.lower()
    dynamic_date_patterns = [
        r'\btoday\b',
        r'\btomorrow\b', 
        r'\byesterday\b',
        r'\bthis\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
        r'\bnext\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
        r'\bcurrent\s+(?:week|month|year)\b',
        r'\btoday.*?(?:is|being|,)\s+\w+\s+\d{1,2}(?:st|nd|rd|th)?\b',  # "today is January 16th"
        r'\btoday.*?(?:is|being|,)\s+\d{1,2}(?:st|nd|rd|th)?\s+\w+\b',  # "today is 16th January"
    ]
    
    if any(re.search(pattern, answer_lower) for pattern in dynamic_date_patterns):
        print(f"DEBUG: Not caching - answer contains dynamic date references")
        return
    
    cache = load_qa_cache()
    embeddings_model = load_embeddings()
    
    question_normalized = question.lower().strip()
    question_embedding = embeddings_model.embed_query(question_normalized)
    
    for idx, cached_q in enumerate(cache["questions"]):
        if cached_q.lower().strip() == question_normalized:
            cache["answers"][idx] = answer
            cache["confidences"][idx] = confidence
            cache["sources"][idx] = sources
            cache["embeddings"][idx] = question_embedding
            save_qa_cache()
            return
    
    cache["questions"].append(question)
    cache["answers"].append(answer)
    cache["confidences"].append(confidence)
    cache["sources"].append(sources)
    cache["embeddings"].append(question_embedding)
    
    save_qa_cache()

_vectordb = None

def load_vectordb():
    """Load the vector database directly for advanced queries"""
    global _vectordb
    if _vectordb is not None:
        return _vectordb
    embeddings = load_embeddings()
    _vectordb = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )
    return _vectordb


def load_retriever():

    global _retriever
    if _retriever is not None:
        return _retriever
    vectordb = load_vectordb()
    _retriever = vectordb.as_retriever(search_kwargs={"k": TOP_K})
    return _retriever


def find_version_table_chunks(version_num: str, active_docs: list) -> list:
    """
    Find chunks that contain both the version number AND author/table info.
    This uses keyword filtering to ensure we get the right version history rows.
    """
    vectordb = load_vectordb()
    collection = vectordb.get()
    
    matching_chunks = []
    
    for i, (doc_id, doc, meta) in enumerate(zip(collection['ids'], collection['documents'], collection['metadatas'])):
        # Skip if not in active documents
        source = meta.get('source', '')
        if source not in active_docs:
            continue
        
        doc_lower = doc.lower()
        
        # Check if this chunk has version table characteristics:
        # 1. Contains the specific version number (e.g., "1.0")
        # 2. Contains author-related terms
        # 3. Ideally contains table markers or column headers
        has_version = version_num in doc
        has_author_info = 'author' in doc_lower or 'reema' in doc_lower or 'pranjali' in doc_lower
        has_table_markers = '[table' in doc_lower or 'reviewed by' in doc_lower or 'approved by' in doc_lower
        
        if has_version and (has_author_info or has_table_markers):
            # Create a document-like object
            from langchain_core.documents import Document
            matching_chunks.append(Document(
                page_content=doc,
                metadata=meta
            ))
            print(f"DEBUG: Found version table chunk: {source} page {meta.get('page')}")
    
    return matching_chunks


def load_llama_model():
    global _llama_model
    if _llama_model is not None:
        return _llama_model
    
    print(f"Loading model from {MODEL_PATH}...")
    print(f"GPU Layers: {config.N_GPU_LAYERS} (0 = CPU only, >0 = GPU enabled)")
    
    _llama_model = Llama(
        model_path=MODEL_PATH,
        n_ctx=config.N_CTX,
        n_threads=config.N_THREADS,
        n_gpu_layers=config.N_GPU_LAYERS,  # Use config value
        verbose=False
    )
    print("Model loaded successfully!")
    return _llama_model


def load_groq_client():

    global _groq_client
    if _groq_client is not None:
        return _groq_client
    
    _groq_client = Groq(api_key=config.GROQ_API_KEY)
    print("Groq client initialized!")
    return _groq_client


def generate_answer_groq(context: str, question: str, conversation_context: list = None, document_names: list = None) -> str:
    
    is_list_question = any(word in question.lower() for word in ["list", "all", "what are the", "show me all", "tell me all", "how many"])
    max_tokens = config.MAX_TOKENS_LONG if is_list_question else config.MAX_TOKENS_SHORT
    
    # Get current date and time
    from datetime import timedelta
    now = datetime.now()
    current_date = now.strftime("%B %d, %Y")  # e.g., "January 13, 2026"
    current_day = now.strftime("%A")  # e.g., "Tuesday"
    current_time = now.strftime("%I:%M %p")  # e.g., "02:30 PM"
    
    # Calculate tomorrow and yesterday
    tomorrow = now + timedelta(days=1)
    tomorrow_date = tomorrow.strftime("%B %d, %Y")
    tomorrow_day = tomorrow.strftime("%A")
    
    yesterday = now - timedelta(days=1)
    yesterday_date = yesterday.strftime("%B %d, %Y")
    yesterday_day = yesterday.strftime("%A")
    
    # Determine the document reference
    if document_names and len(document_names) > 0:
        if len(document_names) == 1:
            doc_reference = f"the {document_names[0].replace('.pdf', '')} document"
        else:
            doc_reference = "the policy documents"
    else:
        doc_reference = "the policy documents"
    
    try:
        client = load_groq_client()
        
        if conversation_context and len(conversation_context) > 0:
            conversation_text = "\n".join(conversation_context)
            user_content = f"""IMPORTANT DATE INFORMATION:
- TODAY is {current_day}, {current_date}
- TOMORROW is {tomorrow_day}, {tomorrow_date}
- YESTERDAY was {yesterday_day}, {yesterday_date}
- Current time: {current_time}

Previous conversation:
{conversation_text}

Document Context:
{context}

Current Question: {question}

Answer the current question based on the provided document context. If the question refers to something from the previous conversation (like "it", "that", "what was it"), use the conversation history to understand what the user is referring to.

CRITICAL: When the user asks about "today", "tomorrow", or "yesterday", use the EXACT dates provided above - NOT the dates in the document context."""
        else:
            user_content = f"""IMPORTANT DATE INFORMATION:
- TODAY is {current_day}, {current_date}
- TOMORROW is {tomorrow_day}, {tomorrow_date}
- YESTERDAY was {yesterday_day}, {yesterday_date}
- Current time: {current_time}

Based on the following document information, please answer the question.

Document Context:
{context}

Question: {question}

CRITICAL: When answering questions about "today", "tomorrow", "yesterday", or current time/date, use the EXACT dates provided above - NOT dates from the document.

Answer:"""
        
        response = client.chat.completions.create(
            model=config.GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": f"""You are a helpful assistant. Answer questions based on the policy documents provided.

When answering, refer to the source as '{doc_reference}' instead of using generic terms.

IMPORTANT: Pay attention to the current date and day of the week provided in the user message. When users ask about "today", use the actual current date and day provided, not any dates mentioned in the document context.

CRITICAL INSTRUCTIONS:
1. When answering questions about time periods (per year, per month, annually, monthly, etc.), pay close attention to what the user is asking for:
   - If asked "per year" or "in a year" or "annually", provide the YEARLY amount (multiply monthly amounts by 12)
   - If asked "per month" or "in a month" or "monthly", provide the MONTHLY amount
   - Always match your answer to the time period the user asked about

2. When answering questions about VERSION NUMBERS (like 1.0, 1.1, 2.0, v1, v2):
   - Pay VERY close attention to the EXACT version number mentioned in the question
   - Version 1.0 and version 1.1 are DIFFERENT versions - do NOT confuse them
   - Look carefully at tables, version history sections, or change logs in the document
   - Match the EXACT version number to find the corresponding author, date, or description

3. When the document contains TABLES:
   - Read each row carefully and match the specific values the user is asking about
   - Tables often contain version history with columns like Version, Author, Date, Description
   - Extract the EXACT information from the correct row based on what the user asks

Be helpful and answer questions using the information in the context. If the context contains relevant information, use it to answer. Only say you cannot find information if the context truly has nothing related to the topic."""
                },
                {
                    "role": "user",
                    "content": user_content
                }
            ],
            max_tokens=max_tokens,
            temperature=0.0
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"Error with Groq API: {e}")
        return f"Error: {str(e)}"


def generate_answer_local(context: str, question: str, conversation_context: list = None, document_names: list = None) -> str:
    
    is_list_question = any(word in question.lower() for word in ["list", "all", "what are the", "show me all", "tell me all", "how many"])
    max_tokens = config.MAX_TOKENS_LONG if is_list_question else config.MAX_TOKENS_SHORT

    # Get current date and time
    from datetime import timedelta
    now = datetime.now()
    current_date = now.strftime("%B %d, %Y")
    current_day = now.strftime("%A")
    
    # Calculate tomorrow and yesterday
    tomorrow = now + timedelta(days=1)
    tomorrow_date = tomorrow.strftime("%B %d, %Y")
    tomorrow_day = tomorrow.strftime("%A")
    
    yesterday = now - timedelta(days=1)
    yesterday_date = yesterday.strftime("%B %d, %Y")
    yesterday_day = yesterday.strftime("%A")
    
    # Pre-process question to replace "today", "tomorrow", "yesterday" with actual days
    processed_question = question.lower()
    
    # Determine which day user is asking about
    asking_about_day = None
    if "tomorrow" in processed_question:
        asking_about_day = tomorrow_day
        day_phrase = f"on {tomorrow_day} ({tomorrow_date})"
    elif "yesterday" in processed_question:
        asking_about_day = yesterday_day
        day_phrase = f"on {yesterday_day} ({yesterday_date})"
    elif "today" in processed_question:
        asking_about_day = current_day
        day_phrase = f"on {current_day} ({current_date})"
    else:
        asking_about_day = None
        day_phrase = ""
    
    # Create a simpler, more direct prompt for the smaller model
    if asking_about_day:
        prompt = f"""<|system|>You are a helpful HR assistant. Answer based on the policy context.<|end|>
<|user|>Policy Context:
{context}

The user is asking about {asking_about_day} ({day_phrase}).

Question: {question}

IMPORTANT: The user is asking about {asking_about_day.upper()}, not any other day. Check if the policy applies to {asking_about_day}.<|end|>
<|assistant|>"""
    else:
        prompt = f"""<|system|>You are a helpful HR assistant. Answer based on the policy context.<|end|>
<|user|>Policy Context:
{context}

Question: {question}<|end|>
<|assistant|>"""

    try:
        llm = load_llama_model()
        
        output = llm(
            prompt,
            max_tokens=max_tokens,
            temperature=config.TEMPERATURE,
            top_k=10,
            top_p=0.9,
            repeat_penalty=1.2,
            stop=["<|end|>", "<|user|>", "Question:"],
            echo=False
        )
        
        return output['choices'][0]['text'].strip()
        
    except Exception as e:
        print(f"Error generating answer with llama.cpp: {e}")
        return ""


def generate_answer(context: str, question: str, conversation_context: list = None, document_names: list = None) -> str:
    if config.USE_GROQ_API:
        return generate_answer_groq(context, question, conversation_context, document_names)
    else:
        return generate_answer_local(context, question, conversation_context, document_names)


def expand_follow_up_question(question: str, conversation_context: list) -> str:
    """
    Expand a follow-up question by incorporating context from previous conversation.
    Detects if the question references previous topics (using pronouns like "it", "that", "this")
    and expands the question to include the topic for better retrieval.
    """
    if not conversation_context or len(conversation_context) == 0:
        return question
    
    question_lower = question.lower().strip()
    
    # Patterns that indicate a follow-up question referencing previous topic
    follow_up_indicators = [
        r'\bit\b',           # "is it paid", "how long is it"
        r'\bthat\b',         # "is that paid"
        r'\bthis\b',         # "is this mandatory"
        r'\bthey\b',         # "are they paid"
        r'\bthose\b',        # "are those required"
        r'\bthe same\b',     # "is the same applicable"
        r'\bwhat about\b',   # "what about version 1.1"
        r'\bhow about\b',    # "how about the other one"
    ]
    
    # Short questions are likely follow-ups (less than 5 words for pronouns, 8 for "what about")
    words = question_lower.split()
    is_short_question = len(words) <= 5
    is_what_about = 'what about' in question_lower or 'how about' in question_lower
    
    # Check if question contains follow-up indicators
    has_follow_up_indicator = any(re.search(pattern, question_lower) for pattern in follow_up_indicators)
    
    # If it's not clearly a follow-up, return original question
    if not ((is_short_question and has_follow_up_indicator) or is_what_about):
        return question
    
    # Extract the topic from the MOST RECENT user question and assistant answer
    # Only use the immediately previous exchange to avoid using stale context
    # conversation_context format: ["User: question1", "Assistant: answer1", "User: question2", "Assistant: answer2"]
    last_user_question = None
    last_assistant_answer = None
    
    # Get the most recent Q&A pair (not older ones)
    for i in range(len(conversation_context) - 1, -1, -1):
        msg = conversation_context[i]
        if msg.startswith("Assistant:") and last_assistant_answer is None:
            last_assistant_answer = msg.replace("Assistant:", "").strip()
        elif msg.startswith("User:") and last_user_question is None:
            last_user_question = msg.replace("User:", "").strip()
            break
    
    if not last_user_question or not last_assistant_answer:
        return question
    
    # DON'T expand if the last answer was negative/not found
    # This prevents using stale context from earlier successful queries
    negative_indicators = [
        'could not find', 'no information', 'not mentioned', 'does not mention',
        'not available', 'not specified', 'not provided'
    ]
    if any(indicator in last_assistant_answer.lower() for indicator in negative_indicators):
        print(f"DEBUG: Skipping follow-up expansion - previous answer was negative/not found")
        return question
    
    # Extract key topic from BOTH the last question AND answer
    # This ensures we use the actual topic that was discussed, not an old one
    hr_topics = [
        'maternity leave', 'paternity leave', 'sick leave', 'casual leave', 'earned leave',
        'annual leave', 'privilege leave', 'medical leave', 'bereavement leave', 'compensatory off',
        'work from home', 'wfh', 'remote work', 'hybrid', 'flexible working',
        'probation', 'notice period', 'resignation', 'termination', 'appraisal',
        'salary', 'bonus', 'increment', 'insurance', 'benefits', 'allowance',
        'overtime', 'working hours', 'shift', 'attendance', 'holiday', 'vacation',
        'training', 'onboarding', 'offboarding', 'background verification', 'bgv',
        'provident fund', 'pf', 'epf', 'gratuity', 'pension',
        'dress code', 'code of conduct', 'harassment', 'posh', 'grievance',
        'travel policy', 'expense', 'reimbursement', 'relocation',
        'version history', 'author', 'reviewer', 'approver', 'version'
    ]
    
    last_question_lower = last_user_question.lower()
    last_answer_lower = last_assistant_answer.lower()
    detected_topic = None
    
    # First try to find topics in the ANSWER (more reliable than question)
    for topic in sorted(hr_topics, key=len, reverse=True):
        if topic in last_answer_lower:
            detected_topic = topic
            break
    
    # If not found in answer, try the question
    if not detected_topic:
        for topic in sorted(hr_topics, key=len, reverse=True):
            if topic in last_question_lower:
                detected_topic = topic
                break
    
    # If no specific topic found, extract main subject from last question
    if not detected_topic:
        # Try to extract the main subject from the last question
        # Remove common question words
        clean_last_q = re.sub(r'\b(what|how|when|where|why|who|is|are|can|do|does|the|a|an|my|about|for|of|in|to|exactly|many)\b', '', last_question_lower)
        clean_last_q = ' '.join(clean_last_q.split()).strip()
        if len(clean_last_q) > 3:  # More strict - at least 4 chars
            detected_topic = clean_last_q
    
    if detected_topic:
        # Special handling for "what about X" or "how about X" patterns
        # "what about version 1.1" should become "version history author version 1.1" using previous context
        what_about_match = re.search(r'(?:what|how)\s+about\s+(.+)', question_lower)
        if what_about_match:
            new_subject = what_about_match.group(1).strip()
            # If the new_subject looks like a version number (e.g., "1.3"), add "version" prefix
            # This ensures "what about 1.3" becomes "author version 1.3" not just "author 1.3"
            is_version_number = re.match(r'^v?\d+\.?\d*$', new_subject)
            if is_version_number:
                new_subject = f"version {new_subject}"
            
            # For version-related follow-ups, also check if "author" was discussed
            # This ensures we include "author" context for better retrieval
            topic_parts = [detected_topic]
            if is_version_number or 'version' in new_subject.lower():
                # Check if the previous conversation was about author
                combined_context = (last_question_lower + ' ' + last_answer_lower).lower()
                if 'author' in combined_context and 'author' not in detected_topic:
                    topic_parts.insert(0, 'author')  # Add "author" at the beginning
            
            # Combine: use the action/query from previous question with new subject
            # e.g., previous: "who is the author of version 1.0" -> detected_topic contains "author"
            # current: "what about version 1.1" -> expand to "who is the author of version 1.1"
            expanded = f"{' '.join(topic_parts)} {new_subject}"
            print(f"DEBUG: Expanded 'what about' question: '{question}' -> '{expanded}' (topic: {detected_topic})")
            return expanded
        
        # Replace pronouns with the topic
        expanded = question_lower
        expanded = re.sub(r'\bit\b', detected_topic, expanded)
        expanded = re.sub(r'\bthat\b', detected_topic, expanded)
        expanded = re.sub(r'\bthis\b', detected_topic, expanded)
        expanded = re.sub(r'\bthey\b', detected_topic, expanded)
        expanded = re.sub(r'\bthose\b', detected_topic, expanded)
        
        # If the question still doesn't have the topic, prepend it
        if detected_topic not in expanded:
            expanded = f"{detected_topic} - {expanded}"
        
        print(f"DEBUG: Expanded follow-up question: '{question}' -> '{expanded}' (topic: {detected_topic})")
        return expanded
    
    return question


def answer_question(question: str, conversation_context: list = None, user: str = None):
    
    # Check for greetings first (robust to punctuation and parens)
    import string
    greetings = ['hi', 'hello', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening',
                 'hola', 'namaste', 'howdy', 'sup', "what's up", 'whats up']
    question_clean = question.strip().lower()
    # Remove punctuation and parens
    question_clean = question_clean.translate(str.maketrans('', '', string.punctuation + "()[]{}'\""))
    question_clean = question_clean.strip()
    # Remove extra spaces
    question_clean = ' '.join(question_clean.split())
    if question_clean in greetings or any(question_clean.startswith(g + ' ') for g in greetings):
        greeting_response = "Hello! 👋 Welcome to the HR Policy Chatbot. How can I help you today? Feel free to ask any questions about the uploaded documents."
        return (
            greeting_response,
            1.0,
            []
        )
    
    # Check if there are any active documents for this user BEFORE processing the query
    active_docs_names = get_active_document_names(user=user)
    all_docs_names = get_all_document_names(user=user)
    print(f"DEBUG: Active docs for user '{user}': {active_docs_names}")
    print(f"DEBUG: All docs for user '{user}': {all_docs_names}")
    
    if not active_docs_names or len(active_docs_names) == 0:
        # Check if user has documents but they're all inactive
        if all_docs_names and len(all_docs_names) > 0:
            return (
                "Your uploaded documents are inactive. Please activate at least one document from the sidebar to get a response.",
                0.0,
                []
            )
        else:
            return (
                "No documents uploaded. Please upload at least one document to ask questions.",
                0.0,
                []
            )
    
    # First, expand follow-up questions with conversation context
    # This ensures "is it paid" becomes "is maternity leave paid" for better retrieval
    original_question = question
    if conversation_context and len(conversation_context) > 0:
        question = expand_follow_up_question(question, conversation_context)
    
    # Expand corporate acronyms in the question
    expanded_question = expand_acronyms(question)
    if expanded_question != question.lower():
        print(f"DEBUG: Acronym expanded '{question}' -> '{expanded_question}'")
        question = expanded_question
    
    # Expand version-related queries to improve semantic matching with tables
    search_query = expand_version_query(question)
    
    is_valid, error_msg = is_valid_query(original_question)
    if not is_valid:
        return (
            error_msg,
            0.0,
            []
        )

    # Try cache with both original and expanded question
    cached_result = find_similar_question(question)
    if cached_result is None and original_question != question:
        cached_result = find_similar_question(original_question)
    if cached_result is not None:
        answer, confidence, sources, similarity = cached_result
        print(f"✓ Cache hit! Similarity: {similarity:.2f}")
        return answer, confidence, sources
    
    retriever = load_retriever()
    # Use the expanded search_query for retrieval (better semantic matching)
    docs = retriever.invoke(search_query)

    print(f"DEBUG: Retrieved {len(docs)} raw documents")
    for i, doc in enumerate(docs):
        print(f"  Doc {i}: source='{doc.metadata.get('source')}', page={doc.metadata.get('page')}")
    
    # Use the same active_docs_names that was checked earlier (already filtered by user)
    print(f"DEBUG: Active document names for user '{user}': {active_docs_names}")
    
    docs = [doc for doc in docs if doc.metadata.get("source") in active_docs_names]
    print(f"DEBUG: After filtering: {len(docs)} documents remain")
    
    # For version queries, supplement with keyword-matched version table chunks
    # Check BOTH original_question AND expanded search_query for version patterns
    # This handles follow-up questions like "what about 1.3" which expand to "author version 1.3"
    query_to_check = search_query.lower() if 'version' in search_query.lower() else original_question.lower()
    version_match = re.search(r'version\s*(\d+\.?\d*)|v(\d+\.?\d*)', query_to_check)
    # Also check for author/who in either the original or expanded query
    has_author_context = ('author' in original_question.lower() or 'who' in original_question.lower() or 
                          'author' in search_query.lower())
    if version_match and has_author_context:
        version_num = version_match.group(1) or version_match.group(2)
        print(f"DEBUG: Looking for version table chunks for version {version_num}")
        table_chunks = find_version_table_chunks(version_num, active_docs_names)
        print(f"DEBUG: Found {len(table_chunks)} version table chunks")
        
        # Add table chunks that aren't already in docs
        existing_contents = set(doc.page_content for doc in docs)
        for chunk in table_chunks:
            if chunk.page_content not in existing_contents:
                docs.insert(0, chunk)  # Insert at beginning for priority
                existing_contents.add(chunk.page_content)
                print(f"DEBUG: Added version table chunk from {chunk.metadata.get('source')} page {chunk.metadata.get('page')}")
    
    # Check if NO active documents exist at all
    if len(active_docs_names) == 0:
        return (
            "No active documents selected. Please enable at least one document from the sidebar.",
            0.0,
            []
        )

    # Active documents exist, but no relevant content found in them
    if not docs:
        return (
            "I could not find this information in the uploaded active documents.",
            0.0,
            []
        )
    context = "\n\n".join(doc.page_content for doc in docs)
    
    is_list_question = any(word in question.lower() for word in ["list", "all", "what are the", "show me all", "tell me all"])
    max_context_length = 6000 if is_list_question else 3000
    context = context[:max_context_length]
    
    print(f"DEBUG: Retrieved {len(docs)} documents, context length: {len(context)} chars")
    print(f"DEBUG: Context preview: {context[:200]}...") 

    sources = []
    document_names = []
    for doc in docs:
        sources.append({
            "source": doc.metadata.get("source"),
            "page": doc.metadata.get("page"),
            "section": doc.metadata.get("section")
        })
        doc_name = doc.metadata.get("source")
        if doc_name and doc_name not in document_names:
            document_names.append(doc_name)
    
    months = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december']
    target_month = None
    
    for month in months:
        if month in question.lower():
            target_month = month
            break
    
    if target_month and any(word in question.lower() for word in ["how many", "count", "number"]) and "holiday" in question.lower():
        month_holidays = []
        lines = context.split('\n')
        days_of_week = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        
        for line in lines:
            if target_month in line.lower() and any(day in line.lower() for day in days_of_week):
                month_holidays.append(line.strip())
        
        if month_holidays:
            unique_holidays = list(set(month_holidays))
            count = len(unique_holidays)
            
            answer = f"There are {count} public holiday{'s' if count != 1 else ''} in {target_month.capitalize()}:\n"
            for holiday in sorted(unique_holidays):
                answer += f"- {holiday}\n"
            
            is_valid, confidence = validate_answer(answer, context)
            # Filter sources to only those relevant to the answer
            filtered_sources = filter_relevant_sources(answer, docs, sources)
            cache_qa(question, answer, confidence, filtered_sources)
            return answer, confidence, filtered_sources
    
    answer = generate_answer(context, question, conversation_context, document_names)

    is_valid, confidence = validate_answer(answer, context)

    print(f"Generated answer: {answer[:100]}...")
    print(f"Confidence: {confidence:.2f}")
    print(f"Is valid: {is_valid}")

    if not is_valid:
        answer = "I could not find this information in the uploaded active documents."
        return answer, confidence, []
    else:
        # Filter sources to only include those that actually contributed to the answer
        filtered_sources = filter_relevant_sources(answer, docs, sources)
        print(f"DEBUG: Filtered sources from {len(sources)} to {len(filtered_sources)}")
        cache_qa(question, answer, confidence, filtered_sources)
    
    return answer, confidence, filtered_sources