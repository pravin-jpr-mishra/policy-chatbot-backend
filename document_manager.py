import os
import json
import pdfplumber
import pandas as pd
from docx import Document
import config
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

CHROMA_DIR = config.CHROMA_DIR
COLLECTION_NAME = "hr_policies"
UPLOADS_DIR = config.UPLOADS_DIR
DOCS_REGISTRY = config.BASE_DIR / "documents_registry.json"

os.makedirs(str(UPLOADS_DIR), exist_ok=True)

def detect_section(text):
    SECTION_KEYWORDS = [
        "Annual Leave",
        "Sick Leave",
        "Work From Home",
        "Code of Conduct",
        "Internship Policy",
        "Termination",
        "Notice Period",
        "Holiday",
        "Benefits",
        "Compensation"
    ]
    for keyword in SECTION_KEYWORDS:
        if keyword.lower() in text.lower():
            return keyword
    return "General"

def load_documents_registry() -> Dict:
    if os.path.exists(str(DOCS_REGISTRY)):
        try:
            with open(str(DOCS_REGISTRY), 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading registry: {e}")
            return {"documents": []}
    return {"documents": []}

def save_documents_registry(registry: Dict):
    try:
        with open(str(DOCS_REGISTRY), 'w', encoding='utf-8') as f:
            json.dump(registry, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving registry: {e}")

def get_active_document_names(user: str = None) -> List[str]:
    registry = load_documents_registry()
    documents = registry["documents"]
    if user:
        documents = [doc for doc in documents if doc.get("owner") == user]
    return [doc["name"] for doc in documents if doc.get("active", True)]

def get_all_document_names(user: str = None) -> List[str]:
    """Get all document names for a user (including inactive ones)"""
    registry = load_documents_registry()
    documents = registry["documents"]
    if user:
        documents = [doc for doc in documents if doc.get("owner") == user]
    return [doc["name"] for doc in documents]

def add_document_to_registry(filename: str, file_path: str, active: bool = True, owner: str = None):
    registry = load_documents_registry()
    
    for doc in registry["documents"]:
        if doc["name"] == filename and doc.get("owner") == owner:
            doc["active"] = active
            doc["file_path"] = filename
            save_documents_registry(registry)
            return
    
    registry["documents"].append({
        "name": filename,
        "file_path": filename,
        "active": active,
        "owner": owner
    })
    save_documents_registry(registry)

def remove_document_from_registry(filename: str, owner: str = None):
    registry = load_documents_registry()
    if owner:
        registry["documents"] = [doc for doc in registry["documents"] if not (doc["name"] == filename and doc.get("owner") == owner)]
    else:
        registry["documents"] = [doc for doc in registry["documents"] if doc["name"] != filename]
    save_documents_registry(registry)

def toggle_document_status(filename: str, active: bool, owner: str = None):
    registry = load_documents_registry()
    for doc in registry["documents"]:
        if doc["name"] == filename:
            if owner is None or doc.get("owner") == owner:
                doc["active"] = active
                break
    save_documents_registry(registry)

def load_pdf(file_path: str, filename: str) -> List[Dict]:
    docs = []
    with pdfplumber.open(file_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            page_content = []
            
            # Extract regular text
            text = page.extract_text()
            if text:
                page_content.append(text)
            
            # Extract tables and format them as structured text
            tables = page.extract_tables()
            for table in tables:
                if table and len(table) > 0:
                    # Format table as readable text with clear row structure
                    table_text = "\n[TABLE START]\n"
                    headers = table[0] if table else []
                    
                    for row_idx, row in enumerate(table):
                        if row:
                            # Clean up None values
                            cleaned_row = [str(cell).strip() if cell else "" for cell in row]
                            
                            if row_idx == 0:
                                # Header row
                                table_text += "Headers: " + " | ".join(cleaned_row) + "\n"
                            else:
                                # Data row - format as "Column: Value" pairs for clarity
                                row_parts = []
                                for col_idx, cell in enumerate(cleaned_row):
                                    if cell and col_idx < len(headers) and headers[col_idx]:
                                        header_name = str(headers[col_idx]).strip()
                                        row_parts.append(f"{header_name}: {cell}")
                                    elif cell:
                                        row_parts.append(cell)
                                if row_parts:
                                    table_text += "Row: " + " | ".join(row_parts) + "\n"
                    
                    table_text += "[TABLE END]\n"
                    page_content.append(table_text)
            
            if page_content:
                docs.append({
                    "text": "\n\n".join(page_content),
                    "metadata": {
                        "source": filename,
                        "page": page_number
                    }
                })
    return docs

def load_docx(file_path: str, filename: str) -> List[Dict]:
    docs = []
    doc = Document(file_path)
    full_text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    
    if full_text.strip():
        docs.append({
            "text": full_text,
            "metadata": {
                "source": filename,
                "page": None
            }
        })
    return docs

def load_txt(file_path: str, filename: str) -> List[Dict]:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    
    if text.strip():
        return [{
            "text": text,
            "metadata": {
                "source": filename,
                "page": None
            }
        }]
    return []

def load_excel(file_path: str, filename: str) -> List[Dict]:
    docs = []
    xls = pd.ExcelFile(file_path)
    
    for sheet_name in xls.sheet_names:
        df = xls.parse(sheet_name)
        text = df.astype(str).fillna("").to_string(index=False)
        
        if text.strip():
            docs.append({
                "text": text,
                "metadata": {
                    "source": filename,
                    "sheet": sheet_name
                }
            })
    return docs

def ingest_single_pdf(uploaded_file, filename: str, owner: str = None) -> bool:
    try:
        # Handle both file paths (from FastAPI) and file objects (from Streamlit)
        if isinstance(uploaded_file, (str, os.PathLike)):
            # uploaded_file is already a file path
            file_path = str(uploaded_file)
        else:
            # uploaded_file is a file object (Streamlit UploadedFile)
            file_path = os.path.join(str(UPLOADS_DIR), filename)
            with open(file_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())

        file_ext = filename.lower().split('.')[-1]
        
        if file_ext == 'pdf':
            documents = load_pdf(file_path, filename)
        elif file_ext in ['doc', 'docx']:
            documents = load_docx(file_path, filename)
        elif file_ext == 'txt':
            documents = load_txt(file_path, filename)
        elif file_ext in ['xls', 'xlsx']:
            documents = load_excel(file_path, filename)
        else:
            print(f"Unsupported file type: {file_ext}")
            return False
        
        if not documents:
            print(f"No content found in {filename}")
            return False
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )
        
        texts = []
        metadatas = []
        
        for doc in documents:
            chunks = splitter.split_text(doc["text"])
            for chunk in chunks:
                texts.append(chunk)
                metadatas.append({
                    **doc["metadata"],
                    "section": detect_section(chunk),
                    "owner": owner  # Add owner to metadata for filtering
                })
        
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        vectordb = Chroma(
            persist_directory=str(CHROMA_DIR),
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME
        )

        vectordb.add_texts(texts=texts, metadatas=metadatas)

        add_document_to_registry(filename, file_path, active=True, owner=owner)
        
        print(f"Successfully ingested {filename} for user: {owner}")
        return True
        
    except Exception as e:
        print(f"Error ingesting {filename}: {e}")
        return False

def remove_document_from_vectordb(filename: str, owner: str = None) -> bool:

    try:
        # Import here to avoid circular imports
        from rag_pipeline import remove_document_from_cache
        
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        vectordb = Chroma(
            persist_directory=str(CHROMA_DIR),
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME
        )
        
        # Build filter based on filename and optionally owner
        where_filter = {"source": filename}
        if owner:
            where_filter = {"$and": [{"source": filename}, {"owner": owner}]}
        
        results = vectordb.get(where=where_filter)
        
        if results and results['ids']:
            vectordb.delete(ids=results['ids'])
            print(f"Removed {len(results['ids'])} chunks from {filename} for user: {owner}")
        
        remove_document_from_registry(filename, owner=owner)
        
        # Also remove cached Q&A entries for this document
        try:
            removed_count = remove_document_from_cache(filename)
            print(f"Removed {removed_count} cached Q&A entries for {filename}")
        except Exception as cache_error:
            print(f"Warning: Could not clear cache for {filename}: {cache_error}")
        
        registry = load_documents_registry()
        for doc in registry["documents"]:
            if doc.get("owner") == owner or owner is None:
                file_path = os.path.join(str(UPLOADS_DIR), doc["file_path"])
                if os.path.exists(file_path):
                    os.remove(file_path)
                    break
        
        return True
        
    except Exception as e:
        print(f"Error removing {filename}: {e}")
        return False
