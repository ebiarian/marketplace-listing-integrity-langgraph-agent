from src.vision_tools import read_label_text, ask_vision_question

# detect_category, detect_product, and retrieve_category_policy are no
# longer in this list — they're called deterministically in agent.py's
# pipeline, not decided by the LLM. Only the two tools genuinely still worth
# an LLM's judgment (which one to call, and what to ask) remain here: OCR
# first, VQA as a targeted fallback. See agent.py and project.md for why.
ALL_TOOLS = [
    read_label_text,
    ask_vision_question,
]
