# =====================================================================
# RAG Pipeline — Company Policy Chatbot
# =====================================================================

from dotenv import load_dotenv
load_dotenv()
from pathlib import Path
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
from groq import Groq
import os

# =====================================================================
# CONFIG
# =====================================================================
CHUNK_SIZE = 100
CHUNK_OVERLAP = 20
TOP_K = 2
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = "openai/gpt-oss-20b"
KNOWLEDGE_DIR = "knowledge"

# =====================================================================
# STEP 1 — INGEST
# =====================================================================
def load_pdfs(knowledge_dir: str) -> str:
    pdf_paths = list(Path(knowledge_dir).glob("*.pdf"))
      
    if not pdf_paths:
        print(f"No PDFs found in '{knowledge_dir}'")
        return ""
    
    print(f"[INGEST] Found {len(pdf_paths)} PDF files")
    
    full_text = ""
    for pdf_path in pdf_paths:
        print(f"  Reading: {pdf_path.name}")
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"
    
    print(f"[INGEST] Total characters: {len(full_text)}\n")
    return full_text

# =====================================================================
# STEP 2 — CHUNK
# =====================================================================
def chunk_text(full_text: str) -> list:
    words = full_text.split()
    
    chunks = []
    for i in range(0, len(words), CHUNK_SIZE - CHUNK_OVERLAP):
        chunk = " ".join(words[i : i + CHUNK_SIZE])
        if chunk.strip():
            chunks.append(chunk)
    
    print(f"[CHUNK] Total words: {len(words)}")
    print(f"[CHUNK] Total chunks: {len(chunks)}\n")
    return chunks

# =====================================================================
# STEP 3 — EMBED
# =====================================================================
def embed_chunks(chunks: list, model: SentenceTransformer):
    print(f"[EMBED] Embedding {len(chunks)} chunks...")
    embeddings = model.encode(chunks)
    print(f"[EMBED] Shape: {embeddings.shape}\n")
    return embeddings

# =====================================================================
# STEP 4 — STORE
# =====================================================================
def store_in_chromadb(chunks: list, embeddings):
    client = chromadb.PersistentClient(path="./chroma_db")
    
    # Get existing or create new
    collection = client.get_or_create_collection("company_policies")
    
    # Clear existing data to avoid duplicates
    collection.delete(where={"source": "pdf"})
    
    collection.add(
        documents=chunks,
        embeddings=embeddings.tolist(),
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )
    
    print(f"[STORE] Stored {len(chunks)} chunks in ChromaDB\n")
    return collection

# =====================================================================
# STEP 5 — RETRIEVE
# =====================================================================
def retrieve_chunks(collection, question: str, model: SentenceTransformer) -> list:
    question_embedding = model.encode([question]).tolist()
    
    results = collection.query(
        query_embeddings=question_embedding,
        n_results=TOP_K
    )
    
    retrieved = results["documents"][0]
    print(f"[RETRIEVE] Question: {question}")
    print(f"[RETRIEVE] Found {len(retrieved)} relevant chunks\n")
    return retrieved

# =====================================================================
# STEP 6 — GENERATE
# =====================================================================
def generate_answer(question: str, retrieved_chunks: list) -> str:
    groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
    context = "\n---\n".join(retrieved_chunks)
    
    prompt = f"""Answer the question using ONLY the context below.
If the answer is not in the context, say 'I don't have that information.'

Context:
{context}

Question: {question}
Answer:"""
    
    response = groq_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    
    answer = response.choices[0].message.content
    # Print safely (avoid encoding issues on Windows console)
    try:
        print(f"[GENERATE] Answer: {answer}\n")
    except Exception:
        print("[GENERATE] Answer (raw):")
        print(answer)
        print()
    return answer

# =====================================================================
# STEP 7 — EVALUATE (basic)
# =====================================================================
def evaluate(question: str, answer: str):
    print(f"[EVALUATE] Q: {question}")
    try:
        print(f"[EVALUATE] A: {answer}")
    except Exception:
        print("[EVALUATE] A (raw):")
        print(answer)
    try:
        score = input("Score this answer (good/bad/hallucinated): ").strip()
    except EOFError:
        score = "n/a"
    except Exception:
        score = "n/a"
    print(f"[EVALUATE] Score: {score}\n")
    return {"question": question, "answer": answer, "score": score}

# =====================================================================
# MAIN — RUN THE PIPELINE
# =====================================================================
if __name__ == "__main__":
    
    # Load embedding model once
    print("[SETUP] Loading embedding model...")
    embed_model = SentenceTransformer(EMBEDDING_MODEL)
    print("[SETUP] Model ready\n")
    
    # Steps 1-4: Build the knowledge base
    raw_text = load_pdfs(KNOWLEDGE_DIR)
    if not raw_text.strip():
        raise ValueError(f"No text extracted from PDFs in '{KNOWLEDGE_DIR}'. "
                         f"Check: directory exists? PDFs have extractable text (not scanned)?")
    chunks = chunk_text(raw_text)
    embeddings = embed_chunks(chunks, embed_model)
    collection = store_in_chromadb(chunks, embeddings)
    
    # Steps 5-7: Answer questions
    questions = [
        "How many sick leave days do employees get?",
        "What is the remote work policy?",
        "What happens if company equipment is lost or stolen?",
        "What is the capital of France?"
    ]
    
    results = []
    for question in questions:
        print("="*50)
        retrieved = retrieve_chunks(collection, question, embed_model)
        answer = generate_answer(question, retrieved)
        result = evaluate(question, answer)
        results.append(result)

    # Summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)   
    for r in results:
        print(f"Q: {r['question']}")
        print(f"Score: {r['score']}\n") 