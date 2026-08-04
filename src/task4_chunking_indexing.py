"""
Task 4 — Chunking & Indexing vào Vector Store (ChromaDB).
"""

import os
import sys
from pathlib import Path

# Clean sys.path if old OPERA-main conflict exists
sys.path = [p for p in sys.path if "OPERA-main" not in p]

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"

# CONFIGURATION
CHUNK_SIZE = 500        # 500 ký tự phù hợp với nội dung điều khoản tuyển sinh & điểm chuẩn
CHUNK_OVERLAP = 50      # Overlap 50 ký tự để giữ ngữ cảnh giữa các đoạn giáp ranh
CHUNKING_METHOD = "recursive"

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "google").lower()
VECTOR_STORE = "chromadb"
COLLECTION_NAME = "university_admissions_docs"

_model = None


class GoogleEmbeddingWrapper:
    def __init__(self, api_key: str, model_name: str = "models/text-embedding-004"):
        import google.generativeai as genai
        self.genai = genai
        self.api_key = api_key
        self.model_name = model_name
        self.genai.configure(api_key=api_key)
        os.environ["GOOGLE_API_KEY"] = api_key
        os.environ["GEMINI_API_KEY"] = api_key

    def _embed_batch(self, batch):
        candidate_models = [
            self.model_name,
            "models/text-embedding-004",
            "text-embedding-004",
            "models/embedding-001",
            "embedding-001",
        ]
        # Remove duplicates while preserving order
        unique_models = []
        for m in candidate_models:
            if m and m not in unique_models:
                unique_models.append(m)

        last_exc = None
        for m in unique_models:
            try:
                res = self.genai.embed_content(
                    model=m,
                    content=batch,
                    task_type="retrieval_document"
                )
                emb = res.get("embedding")
                if emb:
                    return emb
            except Exception as e:
                last_exc = e
                continue
        if last_exc:
            raise last_exc
        raise RuntimeError("Không thể tạo embedding với các mô hình Gemini.")

    def encode(self, texts, show_progress_bar=False):
        is_single = isinstance(texts, str)
        text_list = [texts] if is_single else list(texts)

        embeddings = []
        batch_size = 50
        for i in range(0, len(text_list), batch_size):
            batch = text_list[i : i + batch_size]
            emb = self._embed_batch(batch)
            if isinstance(emb, list):
                if emb and isinstance(emb[0], (int, float)):
                    embeddings.append(emb)
                else:
                    embeddings.extend(emb)

        if is_single:
            return embeddings[0] if embeddings else []
        return embeddings


def get_embedding_model():
    global _model
    provider = os.getenv("EMBEDDING_PROVIDER", "google").lower()
    if _model is None:
        if provider == "google":
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                # If no key set yet, fallback to sentence transformers
                print("[WARNING] GEMINI_API_KEY is not set. Falling back to sentence-transformers.")
                from sentence_transformers import SentenceTransformer
                _model = SentenceTransformer("all-MiniLM-L6-v2")
            else:
                _model = GoogleEmbeddingWrapper(api_key=api_key)
        else:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    documents = []
    if not STANDARDIZED_DIR.exists():
        return documents

    for md_file in STANDARDIZED_DIR.rglob("*.md"):
        content = md_file.read_text(encoding="utf-8")
        doc_type = "legal" if "legal" in str(md_file) else "news"
        documents.append({
            "content": content,
            "metadata": {"source": md_file.name, "type": doc_type}
        })
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents theo strategy đã chọn.

    Returns:
        List of {'content': str, 'metadata': dict}
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc["content"])
        for i, chunk_text in enumerate(splits):
            if chunk_text.strip():
                chunks.append({
                    "content": chunk_text.strip(),
                    "metadata": {
                        **doc["metadata"],
                        "chunk_index": i
                    }
                })
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng model đã chọn.

    Returns:
        Mỗi chunk dict được thêm key 'embedding': list[float]
    """
    model = get_embedding_model()
    texts = [c["content"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=False)
    for chunk, emb in zip(chunks, embeddings):
        if hasattr(emb, "tolist"):
            chunk["embedding"] = emb.tolist()
        else:
            chunk["embedding"] = list(emb)
    return chunks


def get_collection():
    """Get ChromaDB collection."""
    import chromadb
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )


def index_to_vectorstore(chunks: list[dict]):
    """
    Lưu chunks vào vector store đã chọn (ChromaDB).
    """
    if not chunks:
        return

    collection = get_collection()
    ids = [f"{c['metadata']['source']}_chunk_{c['metadata']['chunk_index']}" for c in chunks]
    
    # ChromaDB accepts metadata dict with primitive types (str, int, float, bool)
    clean_metadatas = []
    for c in chunks:
        meta = {}
        for k, v in c["metadata"].items():
            if isinstance(v, (str, int, float, bool)):
                meta[k] = v
            else:
                meta[k] = str(v)
        clean_metadatas.append(meta)

    collection.upsert(
        ids=ids,
        documents=[c["content"] for c in chunks],
        embeddings=[c["embedding"] for c in chunks],
        metadatas=clean_metadatas,
    )


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\n✓ Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"✓ Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"✓ Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print("✓ Indexed to vector store")


if __name__ == "__main__":
    run_pipeline()
