from typing import Literal

from langchain_ollama import ChatOllama
from pydantic import Field, create_model

from src.config import OLLAMA_MODEL
from src.retrieval import get_bounded_vocabulary
from src.vision_tools import feature_question

# Two worked examples embedded directly in the prompt. Tested to be
# necessary, not decorative: without them, extraction missed brand and
# product_type entirely even with full field descriptions (2 of 5 fields
# extracted); with them, accuracy jumped to 4 of 5, with the last gap
# (variant/product_type boundary) fixed separately by the bounded-vocabulary
# schema below. See project.md for the test log.
_FEW_SHOT_EXAMPLES = """Example 1:
Description: "Amazon Fresh Boneless Skinless Chicken Breast, 30 oz Plastic Container"
Phrases -> features:
  "Amazon Fresh" -> brand
  "Chicken Breast" -> product_type
  "Boneless Skinless" -> variant
  "30 oz" -> size
  "Plastic Container" -> container

Example 2:
Description: "Land O Lakes Salted Butter, 8 oz Paper Box"
Phrases -> features:
  "Land O Lakes" -> brand
  "Butter" -> product_type
  "Salted" -> variant
  "8 oz" -> size
  "Paper Box" -> container
"""


def _build_schema(all_features: list[str], bounded_vocab: dict[str, list[str]]):
    """
    Open text field for unbounded features (brand, size — no realistic
    closed list exists); closed Literal choice for features with a bounded
    real-world vocabulary (variant, container). Mixing both in one schema is
    deliberate: open extraction was tested and found reliable for the
    unbounded fields, but not for variant specifically — it either merged a
    compound phrase like "Whole Milk" into product_type or dropped variant
    entirely. Constraining just that field to a closest-match pick fixed it
    immediately, same principle as detect_category/detect_product's closed
    option lists.
    """
    fields = {}
    for f in all_features:
        if f in bounded_vocab:
            fields[f] = (
                Literal[tuple(bounded_vocab[f])],
                Field(description=(
                    f"Pick the closest matching {f} from the allowed list. "
                    f"Use the catch-all option if the text describes no {f} at all."
                )),
            )
        else:
            fields[f] = (
                str,
                Field(description=f"{feature_question(f)} Empty string if not mentioned in the text."),
            )
    return create_model("ExtractedFeatures", **fields)


def extract_description_features(
    raw_description: str,
    category: str,
    critical_features: list[str],
    cosmetic_features: list[str],
    model: str = OLLAMA_MODEL,
) -> dict[str, str]:
    """
    Single-shot LLM call: maps raw listing text onto the category's known
    feature schema (from get_checklist_features). This is the one step in
    the whole pipeline where an LLM does genuine free-form language
    understanding rather than a fixed deterministic step or a narrow
    escalation check — every other step (category/product detection,
    checklist building, feature verification) turned out to be reliably
    handled by code once tested; this one genuinely needs it, because
    mapping arbitrary listing phrasing onto known feature names is a
    language-understanding task a rule-based parser can't approximate at
    real marketplace scale.

    Only validated against grocery listings so far — see project.md for the
    test history (2/5 fields extracted with no few-shot examples, 4/5 with
    them, 5/5 once variant was made a bounded-vocabulary field instead of
    open text).
    """
    all_features = critical_features + cosmetic_features
    bounded_vocab = get_bounded_vocabulary(category)
    schema = _build_schema(all_features, bounded_vocab)

    prompt = (
        "Divide a product description into phrases, and map each phrase to "
        "the feature it describes.\n\n"
        f"{_FEW_SHOT_EXAMPLES}\n"
        "For any field with a fixed list of allowed values, pick the "
        "closest match rather than copying text verbatim.\n\n"
        "Now do the same for this description, then fill in the structured "
        f"output:\n\nDescription: \"{raw_description}\"\n"
    )

    llm = ChatOllama(model=model, temperature=0)
    structured_llm = llm.with_structured_output(schema)
    result = structured_llm.invoke(prompt)

    data = result.model_dump()
    # Drop empty/not-applicable values -- downstream checklist logic treats
    # an absent key as "no claim made", same convention the pre-structured
    # test fixtures already used.
    return {k: v for k, v in data.items() if v and v != "not applicable"}
