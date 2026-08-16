import operator
import re
from typing import Annotated

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict

from src.config import OLLAMA_MODEL, RECURSION_LIMIT
from src.retrieval import get_checklist_features
from src.tools import ALL_TOOLS
from src.vision_tools import (
    detect_category_and_product,
    read_label_text,
    ask_vision_question,
    feature_question,
    check_against_ocr,
    check_against_vqa_answer,
    normalize_value,
    resolve_category,
)


def _clean_category_name(category: str) -> str:
    """
    detect_category's raw VQA answer is often a full sentence (e.g. "this is
    a grocery item, specifically a carton of milk.") — fine for the lenient
    substring matching resolve_category/get_checklist_features already do,
    but not what should be shown in a final verdict. Resolves to the actual
    matched category name for display; falls back to the raw string only
    when nothing resolved (e.g. genuinely "something else").
    """
    policy = resolve_category(category)
    return policy["category"] if policy else category

# ── Why this pipeline is mostly NOT an LLM decision loop ─────────────────────
# The first version of this agent put every step behind an LLM's judgment —
# genuine ReAct end to end. Testing surfaced two problems: ~8-12 sequential
# LLM turns per listing made it slow (~1-2 min/listing locally) and, more
# importantly, most of those turns weren't judgment calls at all — "call
# detect_category, then detect_product" is the same fixed sequence every
# time, with no ambiguity for an LLM to resolve. Worse, exposing OCR/VQA
# fallback as an LLM *choice* meant the model sometimes skipped it —
# "inferring" an unconfirmed brand from context instead of actually checking
# — which is precisely the failure mode this whole project exists to catch.
#
# This version keeps genuine LLM judgment only where testing showed it's
# actually needed: interpreting a VQA answer that's genuinely ambiguous after
# automatic OCR/VQA matching already tried and failed to resolve it. A
# listing where every feature resolves cleanly never invokes the LLM at all.
# This mirrors the exact lesson the sibling ridesharing project already
# learned in its own Stage 1/2 split — the LLM's job is narrowed to what a
# lookup table genuinely can't do, not asked to also re-confirm decisions
# that were never actually in question.


def _build_checklist(critical_features: list[str], description_features: dict) -> tuple[dict, list[str]]:
    """
    Union of the policy's critical features (that the description actually
    makes a claim about — there's nothing to check for a claim that was
    never made) and any other concrete claim in the description not already
    covered. Returns (checklist: feature -> claimed_value, names of the
    checklist entries that are policy-critical). product_type is excluded —
    it's already resolved by detect_category_and_product plus the hard-stop
    check below, not re-checked here.
    """
    checklist: dict = {}
    checklist_critical: list[str] = []

    for f in critical_features:
        if f == "product_type":
            continue
        if description_features.get(f):
            checklist[f] = description_features[f]
            checklist_critical.append(f)

    for f, v in description_features.items():
        if f == "product_type" or f in checklist or not v:
            continue
        checklist[f] = v  # an "other claim" — not on the policy's critical list

    return checklist, checklist_critical


def _hard_stop_mismatch(resolved_product: str, claimed_product_type: str) -> bool:
    """
    Deterministic normalized comparison between what the image actually
    shows and what the description claims the product is — the check that
    catches a listing lying about its own product entirely (e.g. a milk
    photo described as juice). Only returns True for a clear, confident
    mismatch (no meaningful overlap between the two normalized strings);
    partial overlap is treated as consistent rather than escalated to the
    LLM, since testing showed this simple comparison was already reliable
    for this specific check.
    """
    if not claimed_product_type:
        return False
    resolved_norm = normalize_value(resolved_product)
    claimed_norm = normalize_value(claimed_product_type)
    if not resolved_norm or not claimed_norm:
        return False
    return claimed_norm not in resolved_norm and resolved_norm not in claimed_norm


def _deterministic_verify(image_url: str, checklist: dict) -> tuple[dict, list[str]]:
    """
    OCR first (checked once against the whole label, not once per feature),
    VQA fallback for anything OCR didn't find — the same two tools as
    before, just called directly instead of being offered to an LLM to
    decide whether to use. Returns (results, ambiguous_feature_names) —
    ambiguous is only ever non-empty when a targeted VQA question's answer
    couldn't be confidently matched OR contradicted, which is genuinely rare
    given how targeted these fallback questions are.
    """
    ocr_text = read_label_text.invoke({"image_url": image_url})

    results: dict = {}
    ambiguous: list[str] = []

    for feature, claimed in checklist.items():
        ocr_match = check_against_ocr(claimed, ocr_text)
        if ocr_match is True:
            results[feature] = {"value": claimed, "match": True, "source": "OCR"}
            continue

        question = feature_question(feature)
        answer = ask_vision_question.invoke({"image_url": image_url, "question": question})
        vqa_match = check_against_vqa_answer(claimed, answer)
        results[feature] = {
            "value": claimed, "match": vqa_match, "source": "VQA", "vqa_answer": answer,
        }
        if vqa_match is None:
            ambiguous.append(feature)

    return results, ambiguous


# ── Narrow ReAct escalation — the only LLM involvement in this pipeline ─────
# Only reached when the deterministic pass above genuinely couldn't resolve
# something. Scoped to just the ambiguous features, with only the two tools
# a fallback check could ever need — not the full original tool list.

class _EscalationState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]

ESCALATION_SYSTEM_PROMPT = """You are resolving product feature checks that automatic OCR/VQA matching already tried and could not confidently resolve. For each feature listed, you'll be told what was already checked and why it was inconclusive — investigate further using read_label_text and/or ask_vision_question as needed, then report your finding.

Ignore trivial surface-level differences — plural vs. singular, capitalization, or minor spacing/punctuation are NOT mismatches. Only flag a genuinely different value. If you truly cannot determine the answer after investigating, report it as unconfirmed rather than guessing — never infer an answer from the claimed value itself or from general assumptions instead of actual tool evidence.

Once you've investigated every feature listed, respond with EXACTLY one line per feature, in this format, and nothing else:
<feature_name>: match|mismatch|unconfirmed"""


def build_graph():
    """
    Compiles the narrow escalation sub-graph — standard LangGraph ReAct
    shape (agent <-> tools via tools_condition), but bound to only
    read_label_text and ask_vision_question, and only ever invoked by
    _escalate_ambiguous below when there's genuinely something to escalate.
    """
    llm = ChatOllama(model=OLLAMA_MODEL, temperature=0)
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    def agent_node(state: _EscalationState) -> _EscalationState:
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    graph = StateGraph(_EscalationState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(ALL_TOOLS))

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")

    return graph.compile()


def _escalate_ambiguous(graph, image_url: str, ambiguous_results: dict) -> dict:
    lines = "\n".join(
        f"- {f}: claimed value is '{r['value']}'; a targeted question got the answer "
        f"'{r.get('vqa_answer', 'n/a')}', which wasn't conclusive"
        for f, r in ambiguous_results.items()
    )
    human_msg = HumanMessage(content=(
        f"Product image: {image_url}\n\n"
        f"Features needing further investigation:\n{lines}\n\n"
        "Investigate each and report your findings."
    ))

    result = graph.invoke(
        {"messages": [SystemMessage(content=ESCALATION_SYSTEM_PROMPT), human_msg]},
        config={"recursion_limit": RECURSION_LIMIT},
    )
    final_text = result["messages"][-1].content

    resolved = {}
    for feature in ambiguous_results:
        m = re.search(rf"{re.escape(feature)}\s*:\s*(match|mismatch|unconfirmed)", final_text, re.IGNORECASE)
        if m:
            verdict = m.group(1).lower()
            resolved[feature] = True if verdict == "match" else False if verdict == "mismatch" else None
        else:
            resolved[feature] = None  # couldn't parse a verdict — stays unconfirmed, never guessed
    return resolved


# ── Deterministic verdict assembly ───────────────────────────────────────────

def _format_checked(d: dict) -> str:
    if not d:
        return "None"
    return ", ".join(
        f"{f}: {'match' if r['match'] is True else 'mismatch' if r['match'] is False else 'unconfirmed'}"
        for f, r in d.items()
    )


def _build_reasoning(critical_checked: dict, other_checked: dict) -> str:
    mismatched = [f for f, r in critical_checked.items() if r["match"] is False]
    unconfirmed = [f for f, r in critical_checked.items() if r["match"] is None]
    other_mismatched = [f for f, r in other_checked.items() if r["match"] is False]

    if mismatched:
        claims = ", ".join(f"{f} (claimed '{critical_checked[f]['value']}')" for f in mismatched)
        return f"Critical feature mismatch on: {claims}. This alone is enough to flag the listing."
    if unconfirmed:
        return (
            f"All checked critical features matched except {', '.join(unconfirmed)}, which could not "
            "be confirmed from the image — reported as uncertain rather than assumed correct."
        )
    if other_mismatched:
        return (
            f"All critical features matched; a non-critical claim ({', '.join(other_mismatched)}) "
            "did not match, which is a soft warning, not a hard veto."
        )
    return "All checked critical features matched the description."


def _assemble_result(category: str, results: dict, critical_feature_names: list[str], **extra) -> dict:
    critical_checked = {f: r for f, r in results.items() if f in critical_feature_names}
    other_checked = {f: r for f, r in results.items() if f not in critical_feature_names}

    hard_veto = any(r["match"] is False for r in critical_checked.values())
    any_unconfirmed_critical = any(r["match"] is None for r in critical_checked.values())
    any_other_mismatch = any(r["match"] is False for r in other_checked.values())

    if hard_veto:
        verdict = "Likely mismatch"
    elif any_unconfirmed_critical or any_other_mismatch:
        verdict = "Uncertain"
    else:
        verdict = "Likely match"

    answer = (
        f"Verdict: {verdict}\n"
        f"Category: {category}\n"
        f"Critical features checked: {_format_checked(critical_checked)}\n"
        f"Other claims checked: {_format_checked(other_checked)}\n"
        f"Reasoning: {_build_reasoning(critical_checked, other_checked)}"
    )
    return {
        "verdict": verdict,
        "category": category,
        "critical_checked": critical_checked,
        "other_checked": other_checked,
        "answer": answer,
        **extra,
    }


# ── Top-level orchestration ──────────────────────────────────────────────────

def verify_listing(graph, image_url: str, description_features: dict) -> dict:
    """
    Runs the full pipeline for one listing. `graph` is the compiled
    escalation sub-graph from build_graph() — only ever actually invoked if
    a feature genuinely can't be resolved deterministically, so most
    listings complete this function without a single LLM call.

    Returns a dict with the final "verdict"/"answer" plus diagnostic fields
    (category, product, escalated, ambiguous_features, ...) useful for
    inspecting exactly which stage resolved (or didn't resolve) each part of
    the listing — mainly for the notebook's demonstration, not required by
    the verdict logic itself.
    """
    category, product, product_confirmed = detect_category_and_product(image_url)
    resolved_product = product if product_confirmed else category
    display_category = _clean_category_name(category)

    claimed_product_type = description_features.get("product_type", "")
    if _hard_stop_mismatch(resolved_product, claimed_product_type):
        results = {"product_type": {"value": claimed_product_type, "match": False, "source": "detect_category/detect_product"}}
        return _assemble_result(
            display_category, results, critical_feature_names=["product_type"],
            product=product, product_confirmed=product_confirmed,
            hard_stop=True, escalated=False, ambiguous_features=[],
        )

    critical_features, _cosmetic_features = get_checklist_features(category)
    checklist, checklist_critical = _build_checklist(critical_features, description_features)

    results, ambiguous = _deterministic_verify(image_url, checklist)

    escalated = bool(ambiguous)
    if ambiguous:
        ambiguous_results = {f: results[f] for f in ambiguous}
        resolved = _escalate_ambiguous(graph, image_url, ambiguous_results)
        for f, verdict in resolved.items():
            results[f]["match"] = verdict

    results["product_type"] = {"value": claimed_product_type, "match": True, "source": "detect_category/detect_product"}
    checklist_critical_with_product = checklist_critical + (["product_type"] if claimed_product_type else [])

    return _assemble_result(
        display_category, results, critical_feature_names=checklist_critical_with_product,
        product=product, product_confirmed=product_confirmed,
        hard_stop=False, escalated=escalated, ambiguous_features=ambiguous,
    )
