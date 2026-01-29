import numpy as np
from typing import List, Tuple


_embeddings_model = None

def get_embeddings_model():
    """Lazy load the embeddings model to avoid circular imports"""
    global _embeddings_model
    if _embeddings_model is None:
        from langchain_huggingface import HuggingFaceEmbeddings
        _embeddings_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    return _embeddings_model

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors"""
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0.0
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def compute_confidence_semantic(answer: str, context: str) -> float:
    """
    Compute confidence score using Answer-to-Source Semantic Grounding (Embedding-based)
    
    1. Embed the answer
    2. Embed each retrieved chunk (split context into chunks)
    3. Compute cosine similarity between answer and each chunk
    4. Return weighted max similarity as confidence score
    """
    if not answer or not context:
        return 0.0
    
    try:
        embeddings = get_embeddings_model()
        
        
        answer_embedding = np.array(embeddings.embed_query(answer))
     
        chunks = [chunk.strip() for chunk in context.split('\n\n') if chunk.strip()]
        
      
        if len(chunks) <= 1:
            chunks = [chunk.strip() for chunk in context.split('\n') if chunk.strip()]
        
      
        if len(chunks) <= 2:
            import re
            sentences = re.split(r'(?<=[.!?])\s+', context)
        
            chunks = []
            for i in range(0, len(sentences), 3):
                chunk = ' '.join(sentences[i:i+3])
                if chunk.strip():
                    chunks.append(chunk.strip())
        
        if not chunks:
            chunks = [context]
        
        similarities = []
        for chunk in chunks:
            if len(chunk) > 10: 
                chunk_embedding = np.array(embeddings.embed_query(chunk))
                similarity = cosine_similarity(answer_embedding, chunk_embedding)
                similarities.append(similarity)
        
        if not similarities:
            return 0.0
        
        
        similarities_sorted = sorted(similarities, reverse=True)
        
        max_sim = similarities_sorted[0]
        
        
        top_k = min(3, len(similarities_sorted))
        top_k_avg = sum(similarities_sorted[:top_k]) / top_k
  
        confidence = 0.6 * max_sim + 0.4 * top_k_avg
     
        confidence = min(1.0, max(0.0, (confidence - 0.3) / 0.6))
        
        return round(confidence, 2)
        
    except Exception as e:
        print(f"Error computing semantic confidence: {e}")
     
        return compute_confidence_fallback(answer, context)

def compute_confidence_fallback(answer: str, context: str) -> float:
    """Fallback word-overlap based confidence (used if embedding fails)"""
    if not answer or not context:
        return 0.0

    answer_lower = answer.lower()
    context_lower = context.lower()

    answer_words = set(answer_lower.split())
    context_words = set(context_lower.split())

    if not answer_words:
        return 0.0

    overlap = answer_words.intersection(context_words)
    word_overlap_ratio = len(overlap) / len(answer_words)
    common_words = {'the', 'and', 'for', 'that', 'this', 'with', 'from', 'have', 'will', 'your', 'are', 'can', 'was', 'were', 'been', 'has', 'had', 'yes', 'not'}
    key_terms = [word for word in answer_words if len(word) > 3 and word not in common_words]
    
    if key_terms:
        key_term_found = sum(1 for term in key_terms if term in context_lower)
        key_term_ratio = key_term_found / len(key_terms)
        
        return max(word_overlap_ratio, key_term_ratio * 0.8)
    
    return word_overlap_ratio


def compute_confidence(answer: str, context: str) -> float:
    """Main confidence computation using semantic grounding"""
    return compute_confidence_semantic(answer, context)


def validate_answer(answer: str, context: str, threshold: float = 0.05):
    answer_lower = answer.lower()
    
  
    not_found_patterns = [
        "could not find",
        "couldn't find", 
        "i don't have",
        "i do not have",
        "no information available",
        "not found in",
        "unable to find",
        "cannot find",
        "does not contain information about",
        "does not have information about",
    ]
    
    for pattern in not_found_patterns:
        if pattern in answer_lower:
            return False, 0.0
    
    confidence = compute_confidence(answer, context)
    is_valid = confidence >= threshold
    return is_valid, confidence
