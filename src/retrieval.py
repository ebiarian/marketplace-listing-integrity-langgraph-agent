from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_core.tools import tool

from src.config import EMBED_MODEL, CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME
from src.policy_corpus import CATEGORY_POLICIES

_vectorstore = None


def get_vectorstore() -> Chroma:
    """
    Lazy-loaded singleton so the embedding model and the Chroma client are
    only initialized once per process, no matter how many times a tool call
    or notebook cell asks for the store.
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


@tool
def retrieve_category_policy(category: str) -> str:
    """
    Call this after detect_category (and detect_product, including a retry if
    one was needed) has resolved the listing's category. Retrieves the
    verification policy for that category — which features are critical (any
    mismatch flags the listing) versus cosmetic (a mismatch is only a soft
    warning).

    Pass the resolved category name (e.g. "grocery"). If detect_category
    returned "something else" or the category never resolved, pass that
    through as-is — this falls back to a general default policy.

    Use the returned policy to decide which features are worth checking with
    read_label_text and ask_vision_question next, and how strictly to weigh
    each one.
    """
    vs = get_vectorstore()

    # Fast path: `category` is already one of the corpus's own document IDs
    # (see build_policy_corpus — ids are the category names themselves), so
    # an exact lookup is strictly more reliable than a similarity search here
    # — no risk of the embedding model picking a near-but-wrong match when we
    # already know precisely which document we want.
    exact = vs.get(ids=[category])
    if exact["ids"]:
        return exact["documents"][0]

    # Fallback: category is unresolved (e.g. "something else") or otherwise
    # doesn't match a known ID — this is where semantic search actually earns
    # its keep, finding the closest policy to whatever text is available
    # rather than failing outright.
    results = vs.similarity_search(category, k=1)
    if results:
        return results[0].page_content

    default = vs.get(ids=["default"])
    if default["ids"]:
        return default["documents"][0]
    return (
        "No policy found — apply a conservative default: treat brand, "
        "product_type, and size as critical."
    )
