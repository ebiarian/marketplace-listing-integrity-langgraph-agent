# Ollama model used as the agent's reasoning brain — matches the sibling
# ridesharing project's final default (see its README), already proven
# reliable for structured tool-calling on a 16GB M2 Mac.
OLLAMA_MODEL = "qwen2.5:14b"

# Local embedding model for the RAG policy corpus, served via Ollama — keeps
# the whole stack offline, no cloud embeddings API required.
EMBED_MODEL = "nomic-embed-text"

CHROMA_PERSIST_DIR = "./chroma_db"
CHROMA_COLLECTION_NAME = "category_verification_policies"

# Vision models — reused as-is from product-image-description-alignment
# (https://github.com/ebiarian/product-image-description-alignment).
MOONDREAM_MODEL = "vikhyatk/moondream2"
MOONDREAM_REVISION = "2025-01-09"

OCR_MIN_CONFIDENCE = 0.4

# Category names and per-category product lists live only in
# policy_corpus.CATEGORY_POLICIES — not duplicated here — so the VQA
# question options and the RAG corpus can never silently drift apart.

# Safety net for the ReAct loop: caps total graph steps (detect_category,
# detect_product, an optional one-time retry of both, retrieve_category_policy,
# several read_label_text/ask_vision_question calls, final answer) well above
# what a normal run needs, so a genuinely confused model can't loop forever.
RECURSION_LIMIT = 30
