import re
from io import BytesIO

import numpy as np
import requests
import torch
from PIL import Image
from langchain_core.tools import tool
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import MOONDREAM_MODEL, MOONDREAM_REVISION, OCR_MIN_CONFIDENCE
from src.policy_corpus import CATEGORY_POLICIES


def fetch_image_from_url(url: str) -> Image.Image:
    """
    Fetch an image from the internet and return as a PIL Image. Ported from
    product-image-description-alignment/src/utils.py.
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")


def _get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


# EasyOCR is cached at module level for the same reason Moondream2 is below:
# instantiating a fresh easyocr.Reader on every call reloads its detector +
# recognizer weights from disk each time — unnecessary repeated I/O and
# memory churn over a session that calls this tool many times.
_ocr_reader = None


def _get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        _ocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _ocr_reader


@tool
def read_label_text(image_url: str) -> str:
    """
    Reads all visible printed text from a product image using OCR (EasyOCR).
    Fast, and the most reliable source for anything printed on the label —
    brand name, size, variant.

    Call this before ask_vision_question for any text feature you still need
    to check. Only fall back to ask_vision_question if the text you need
    isn't found here — e.g. a stylised logo OCR can't read, or a purely
    visual property like container type that was never printed as text.
    """
    img = fetch_image_from_url(image_url)
    img_array = np.array(img)

    reader = _get_ocr_reader()
    results = reader.readtext(img_array)

    texts = [text for _, text, conf in results if conf >= OCR_MIN_CONFIDENCE]
    return " ".join(texts).lower() or "(no text detected)"


# Moondream2 loads once per process (module-level cache below). The encoded
# image is cached too, but for only ONE image URL at a time — a multi-question
# sequence about the SAME image (the actual reason this cache exists) still
# only encodes once. An earlier version cached every image URL ever seen for
# the whole session, which pinned multiple large tensors in GPU memory
# simultaneously — memory torch.mps.empty_cache() cannot reclaim, since live
# references aren't idle cache. Capping this at one entry means switching to
# a new image releases the previous one immediately.
_moondream_model = None
_moondream_tokenizer = None
_cached_image_url = None
_cached_encoded_image = None


def _get_moondream():
    global _moondream_model, _moondream_tokenizer
    if _moondream_model is None:
        device = _get_device()
        _moondream_model = AutoModelForCausalLM.from_pretrained(
            MOONDREAM_MODEL,
            revision=MOONDREAM_REVISION,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        ).to(device)
        _moondream_tokenizer = AutoTokenizer.from_pretrained(
            MOONDREAM_MODEL, revision=MOONDREAM_REVISION,
        )
    return _moondream_model, _moondream_tokenizer


def _release_gpu_memory() -> None:
    """
    Only called when a live reference was just dropped (evicting the
    previously cached image below) — that's the only time there's actually
    idle memory for MPS to reclaim. NOT called after every single question:
    clearing the allocator needlessly adds reallocation overhead to every
    call, which was making the agent slower without fixing the real leak.
    """
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _get_encoded_image(image_url: str):
    global _cached_image_url, _cached_encoded_image
    if image_url != _cached_image_url:
        if _cached_encoded_image is not None:
            _cached_encoded_image = None   # drop the old tensor reference first...
            _release_gpu_memory()          # ...then reclaim its memory before allocating the new one
        model, _ = _get_moondream()
        img = fetch_image_from_url(image_url)
        _cached_encoded_image = model.encode_image(img)
        _cached_image_url = image_url
    return _cached_encoded_image


def _ask(image_url: str, question: str) -> str:
    model, tokenizer = _get_moondream()
    enc_image = _get_encoded_image(image_url)
    answer = model.answer_question(enc_image, question, tokenizer)
    return answer.strip().lower()


def _category_names() -> list[str]:
    return [p["category"] for p in CATEGORY_POLICIES if p["category"] != "default"]


def _join_options(options: list[str]) -> str:
    if len(options) == 1:
        return options[0]
    return ", ".join(options[:-1]) + f", or {options[-1]}"


def resolve_category(category: str) -> dict | None:
    """
    Looks up a category's policy entry leniently. VQA answers aren't
    guaranteed to be the bare category name — e.g. "grocery item" instead of
    "grocery" — so an exact-match-only lookup fails on normal model phrasing,
    not just genuine ambiguity. Exact match first, then substring match
    against known category names, mirroring the same
    exact-match-then-fallback approach retrieve_category_policy uses.
    """
    for p in CATEGORY_POLICIES:
        if p["category"] == category:
            return p
    category_lower = category.lower()
    for p in CATEGORY_POLICIES:
        if p["category"] != "default" and p["category"] in category_lower:
            return p
    return None


# ── Deterministic category/product detection ────────────────────────────────
# Plain functions, not @tool — the agent no longer decides WHETHER or WHEN to
# call these; they always run in this fixed order for every listing, so
# there's nothing for an LLM to reason about here. Removing them from the
# LLM's tool list (and thus from its reasoning loop) is most of where the
# token/latency savings in this redesign come from.

def detect_category(image_url: str) -> str:
    """
    Determines the product's category directly from the image — NOT from the
    listing's description text, which may be wrong (checking that is the
    whole point of this agent). Returns one of the known categories (e.g.
    "grocery") or "something else" if the image doesn't clearly fit any.
    """
    options = _join_options(_category_names())
    question = f"Is this a {options} item, or something else?"
    return _ask(image_url, question)


def detect_product(image_url: str, category: str) -> str:
    """
    Determines the specific product directly from the image, scoped to the
    given category's known products (e.g. for "grocery": milk, coffee,
    butter, chicken, bread, or juice) — far more reliable than one
    open-ended question spanning every category at once.
    """
    policy = resolve_category(category)
    common_products = policy.get("common_products") if policy else None
    if not common_products:
        return "unknown — no product list defined for this category"

    options = _join_options(common_products)
    question = f"Is this {options}, or something else?"
    return _ask(image_url, question)


def detect_category_and_product(image_url: str) -> tuple[str, str, bool]:
    """
    Runs detect_category -> detect_product, with the self-consistency check
    and one-time retry built in as plain code — no LLM call needed to decide
    "is this consistent?" or "should I retry?"; those are fixed rules, not
    judgment calls. Returns (category, product, product_confirmed) — if the
    two answers are still inconsistent after one retry, product_confirmed is
    False and callers should trust category alone, not force a guess.
    """
    category = detect_category(image_url)
    product = detect_product(image_url, category)
    policy = resolve_category(category)

    def _is_consistent(cat_policy: dict | None, prod: str) -> bool:
        if cat_policy is None:
            return False
        products = cat_policy.get("common_products") or []
        return any(normalize_value(p) in normalize_value(prod) or normalize_value(prod) in normalize_value(p) for p in products)

    if _is_consistent(policy, product):
        return category, product, True

    # One retry, exactly as previously specified in the system prompt —
    # now enforced by code instead of hoping the model remembers the rule.
    category = detect_category(image_url)
    product = detect_product(image_url, category)
    policy = resolve_category(category)

    if _is_consistent(policy, product):
        return category, product, True

    return category, "unconfirmed", False


@tool
def ask_vision_question(image_url: str, question: str) -> str:
    """
    Asks Moondream2 a single targeted question about a product image and
    returns a short answer. Use this to check ONE feature the OCR text
    didn't resolve — a purely visual property (e.g. "Is this in a bottle,
    can, carton, or box?"), or a text feature OCR missed (e.g. a stylised
    logo).

    Ask one specific, closed-ended question per call — offering a small set
    of likely options in the question itself (e.g. "Is this whole, 2%, or
    skim milk?") gets a far more reliable answer than an open-ended question.
    """
    return _ask(image_url, question)


# ── Deterministic feature verification helpers ──────────────────────────────
# Fixed question templates per feature name, reused across categories that
# share a feature (e.g. "brand" means the same thing for grocery and
# electronics). Mirrors product-image-description-alignment's own
# FEATURE_QUESTIONS — a proven approach, not a new experiment — rather than
# asking an LLM to invent a question's wording each time.
FEATURE_QUESTION_TEMPLATES = {
    "brand": "What brand name is shown on this product?",
    "product_type": "What type of product is this?",
    "size": "What is the size or quantity of this product?",
    "variant": "What variant or flavor is this?",
    "container": "Is this in a bottle, can, carton, container, canister, bag, or box?",
    "model_number": "What model number is shown on this product?",
    "key_spec": "What are the key specifications (e.g. wattage, capacity, connector type) shown on this product?",
    "color": "What color is this product?",
    "material": "What material is this product made of?",
    "packaging_style": "How is this product packaged?",
}


def feature_question(feature: str) -> str:
    return FEATURE_QUESTION_TEMPLATES.get(
        feature, f"What is the {feature.replace('_', ' ')} of this product?"
    )


def normalize_value(text: str) -> str:
    """
    Lowercase, strip punctuation, collapse whitespace, and fold a trailing
    "s" off each word as a cheap plural/singular normalization (e.g.
    "chicken breasts" -> "chicken breast"). This is a deterministic fix for
    the plural/singular false-mismatch bug found during testing — it can't
    be "forgotten" the way a prompt instruction could be.
    """
    text = str(text).lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    words = [w[:-1] if w.endswith("s") and len(w) > 3 else w for w in text.split()]
    return " ".join(words)


_UNCERTAIN_PHRASES = ("not visible", "not shown", "can't tell", "cannot tell", "unclear", "unknown", "not sure")


def check_against_ocr(claimed_value: str, ocr_text: str) -> bool | None:
    """
    OCR output is one unstructured blob of every text fragment on a label —
    nutrition facts, serving sizes, and other numbers unrelated to the
    claimed feature are mixed in with no labeling of which is which. So this
    only ever returns True (the claimed value is genuinely present) or None
    (not found — NOT the same as a mismatch, since the blob is incomplete
    and noisy, not a reliable source of "the true value" to compare against).
    Never returns False — that determination requires a targeted question,
    which is what check_against_vqa_answer (below) is for.
    """
    claimed_norm = normalize_value(claimed_value)
    if not claimed_norm:
        return None
    return True if claimed_norm in normalize_value(ocr_text) else None


def check_against_vqa_answer(claimed_value: str, vqa_answer: str) -> bool | None:
    """
    Unlike OCR's unstructured blob, a VQA answer is a direct response to a
    targeted question about ONE feature — so an answer that clearly states
    something else counts as a real mismatch, not just an absence.
    """
    if any(phrase in vqa_answer.lower() for phrase in _UNCERTAIN_PHRASES):
        return None
    claimed_norm = normalize_value(claimed_value)
    answer_norm = normalize_value(vqa_answer)
    if not claimed_norm or not answer_norm:
        return None
    if claimed_norm in answer_norm or answer_norm in claimed_norm:
        return True
    return False
