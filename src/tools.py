from src.retrieval import retrieve_category_policy
from src.vision_tools import (
    detect_category,
    detect_product,
    read_label_text,
    ask_vision_question,
)

# Order roughly mirrors the intended pipeline (detect -> retrieve -> verify),
# though the agent decides the actual call order itself — this list only
# defines what's available, not a fixed sequence.
ALL_TOOLS = [
    detect_category,
    detect_product,
    retrieve_category_policy,
    read_label_text,
    ask_vision_question,
]
