from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

from src.config import EMBED_MODEL, CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME
from src.policy_corpus import CATEGORY_POLICIES

_vectorstore = None


def get_vectorstore() -> Chroma:
    """
    Lazy-loaded singleton so the embedding model and the Chroma client are
    only initialized once per process, no matter how many times retrieval or
    a notebook cell asks for the store.
    """
    global _vectorstore
    if _vectorstore is None:
        embeddings = OllamaEmbeddings(model=EMBED_MODEL)
        _vectorstore = Chroma(
            collection_name=CHROMA_COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=CHROMA_PERSIST_DIR,
        )
    return _vectorstore


def build_policy_corpus(force_rebuild: bool = False) -> Chroma:
    """
    Embeds every category policy document into the Chroma vector store.
    Chroma persists to disk, so this is safe to call at the start of every
    notebook run — it only re-embeds when the collection is empty or
    force_rebuild=True is passed (e.g. after editing policy_corpus.py).
    """
    vs = get_vectorstore()
    existing = vs.get()

    if existing["ids"] and not force_rebuild:
        return vs

    if existing["ids"]:
        vs.delete(ids=existing["ids"])

    docs = [
        Document(page_content=p["text"], metadata={"category": p["category"]})
        for p in CATEGORY_POLICIES
    ]
    ids = [p["category"] for p in CATEGORY_POLICIES]
    vs.add_documents(docs, ids=ids)
    return vs


def _resolve_policy(category: str) -> dict:
    """
    Resolves a category string to its full policy entry — exact match
    against CATEGORY_POLICIES first (no need to touch the vector store at
    all for the common case), then RAG semantic search as a fallback for
    anything that doesn't resolve to a known category name (e.g.
    detect_category returned "something else"), then the "default" policy
    if even that comes back empty. This is called deterministically now —
    no LLM decides to call this, so there's no tool docstring; the calling
    code in agent.py always calls it once category detection is done.
    """
    for p in CATEGORY_POLICIES:
        if p["category"] == category:
            return p

    vs = get_vectorstore()
    results = vs.similarity_search(category, k=1)
    if results:
        retrieved_category = results[0].metadata.get("category")
        for p in CATEGORY_POLICIES:
            if p["category"] == retrieved_category:
                return p

    return next(p for p in CATEGORY_POLICIES if p["category"] == "default")


def retrieve_category_policy(category: str) -> str:
    """Returns the policy prose text for a resolved category (RAG lookup)."""
    return _resolve_policy(category)["text"]


def get_checklist_features(category: str) -> tuple[list[str], list[str]]:
    """
    Returns (critical_features, cosmetic_features) for a resolved category —
    the structured lists the deterministic checklist-builder in agent.py
    reads, as opposed to the prose text above (which exists for the RAG
    document itself and for the final explanation's context, not for code
    to parse).
    """
    policy = _resolve_policy(category)
    return policy.get("critical_features", []), policy.get("cosmetic_features", [])


def get_bounded_vocabulary(category: str) -> dict[str, list[str]]:
    """
    Returns {feature_name: [allowed values]} for features with a genuinely
    bounded real-world vocabulary (e.g. variant, container) — used by
    extraction.py to constrain those specific fields to a closest-match pick
    instead of open text extraction. A feature not present in the returned
    dict has no bounded vocabulary and should be extracted as open text.
    """
    policy = _resolve_policy(category)
    return policy.get("bounded_features", {})
