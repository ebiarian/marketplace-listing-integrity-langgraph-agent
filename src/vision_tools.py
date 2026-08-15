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


@tool
def read_label_text(image_url: str) -> str:
    """
    Reads all visible printed text from a product image using OCR (EasyOCR).
    Fast, and the most reliable source for anything printed on the label —
    brand name, size, variant.

    Call this before ask_vision_question for any text feature the retrieved
    category policy marks as critical. Only fall back to ask_vision_question
    if the text you need isn't found here — e.g. a stylised logo OCR can't
    read, or a purely visual property like container type that was never
    printed as text at all.
    """
    img = fetch_image_from_url(image_url)
    img_array = np.array(img)

    import easyocr
    reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    results = reader.readtext(img_array)

    texts = [text for _, text, conf in results if conf >= OCR_MIN_CONFIDENCE]
    return " ".join(texts).lower() or "(no text detected)"


# Moondream2 and each image's encoding are cached at module level. Reloading
# the model or re-encoding the same image on every single tool call would
# make a multi-question ReAct loop prohibitively slow — the model loads once
# per process, and each image is encoded once per URL the first time it's
# asked about (by any of the three VQA-backed tools below), then reused for
# every later question about that same image.
_moondream_model = None
_moondream_tokenizer = None
_encoded_image_cache: dict = {}


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


def _get_encoded_image(image_url: str):
    model, _ = _get_moondream()
    if image_url not in _encoded_image_cache:
        img = fetch_image_from_url(image_url)
        _encoded_image_cache[image_url] = model.encode_image(img)
    return _encoded_image_cache[image_url]


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


@tool
def detect_category(image_url: str) -> str:
    """
    First tool to call for any new listing. Determines the product's category
    directly from the image — NOT from the listing's description text, which
    may be wrong (checking that is the whole point of this agent). Returns
    one of the known categories (e.g. "grocery") or "something else" if the
    image doesn't clearly fit any of them.

    The question offers only a short list of known categories rather than an
    open-ended "what category is this?" — Moondream2 is small enough that a
    short, constrained multiple-choice question is meaningfully more reliable
    than an open one.

    Always call this before detect_product or retrieve_category_policy.
    """
    options = _join_options(_category_names())
    question = f"Is this a {options} item, or something else?"
    return _ask(image_url, question)


@tool
def detect_product(image_url: str, category: str) -> str:
    """
    Second tool to call, immediately after detect_category — pass it the
    category detect_category just returned. Asks a follow-up question scoped
    to ONLY that category's known products (e.g. for "grocery": milk,
    coffee, butter, chicken, bread, or juice) — far more reliable than one
    open-ended "what product is this?" question spanning every category at
    once.

    Use this answer for two things: (1) as the argument to
    retrieve_category_policy, and (2) to sanity-check against the listing's
    own description — if they clearly disagree, that is itself the strongest
    possible mismatch signal, and no further checking is needed.

    Self-consistency check: if this answer doesn't plausibly belong to the
    category you passed in (e.g. category="grocery" but this returns
    "battery"), the image evidence is inconsistent — call detect_category and
    detect_product again, once. If still inconsistent after that one retry,
    trust detect_category's answer only and treat the specific product
    identity as unconfirmed rather than forcing a guess; proceed using the
    category alone.
    """
    policy = next((p for p in CATEGORY_POLICIES if p["category"] == category), None)
    common_products = policy.get("common_products") if policy else None
    if not common_products:
        return "unknown — no product list defined for this category"

    options = _join_options(common_products)
    question = f"Is this {options}, or something else?"
    return _ask(image_url, question)


@tool
def ask_vision_question(image_url: str, question: str) -> str:
    """
    Asks Moondream2 a single targeted question about a product image and
    returns a short answer. Use this for a specific feature check — visual
    properties not printed as text (e.g. "Is this in a bottle, can, carton,
    or box?"), or as a fallback when read_label_text doesn't contain a
    feature the category policy marks as critical (e.g. a stylised logo OCR
    can't read).

    This is for checking ONE feature at a time during the verification step,
    not for category/product detection — use detect_category and
    detect_product for that instead, since their constrained option lists are
    more reliable than an open-ended question here would be.

    Ask one specific, closed-ended question per call — offering a small set
    of likely options in the question itself (e.g. "Is this whole, 2%, or
    skim milk?") gets a far more reliable answer than an open-ended question.
    """
    return _ask(image_url, question)
