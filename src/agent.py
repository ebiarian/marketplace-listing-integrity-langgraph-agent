import operator
import re
from typing import Annotated, Literal

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict

from src.config import OLLAMA_MODEL, RECURSION_LIMIT
from src.extraction import extract_description_features
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
    VQA fallback for anything OCR didn't find. Returns (results,
    ambiguous_feature_names) — ambiguous is only ever non-empty when a
    targeted VQA question's answer couldn't be confidently matched OR
    contradicted, which is genuinely rare given how targeted these fallback
    questions are.
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


# ── Escalation sub-graph — one node of the full pipeline graph below ───────
# Only reached when the deterministic pass genuinely couldn't resolve
# something. Scoped to just the ambiguous features, with only the two tools
# a fallback check could ever need. This is a compiled sub-graph invoked as
# a single atomic node in the pipeline graph, the same pattern the sibling
# ridesharing project used for its own per-zone sub-graphs in Stage 5.

class _EscalationState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]

ESCALATION_SYSTEM_PROMPT = """You are resolving product feature checks that automatic OCR/VQA matching already tried and could not confidently resolve. For each feature listed, you'll be told what was already checked and why it was inconclusive — investigate further using read_label_text and/or ask_vision_question as needed, then report your finding.

Ignore trivial surface-level differences — plural vs. singular, capitalization, or minor spacing/punctuation are NOT mismatches. Only flag a genuinely different value. If you truly cannot determine the answer after investigating, report it as unconfirmed rather than guessing — never infer an answer from the claimed value itself or from general assumptions instead of actual tool evidence.

Once you've investigated every feature listed, respond with EXACTLY one line per feature, in this format, and nothing else:
<feature_name>: match|mismatch|unconfirmed"""


def build_escalation_graph():
    """
    Compiles the escalation sub-graph — a small agent-and-tools loop bound
    to only read_label_text and ask_vision_question, invoked as a single
    node inside the full pipeline graph built by build_graph() below.
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


def _escalate_ambiguous(escalation_graph, image_url: str, ambiguous_results: dict) -> dict:
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

    result = escalation_graph.invoke(
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


# ── The full pipeline, as one StateGraph ────────────────────────────────────
# Every step is a node — including the deterministic ones. A deterministic
# node is still a legitimate graph node: the sibling ridesharing project's
# Stage 1 built a fully rule-based StateGraph with no LLM at all, for the
# same reason this one does — one state object threads through every step,
# and one diagram shows the whole agent, not just the one part that happens
# to call an LLM.

class PipelineState(TypedDict):
    # inputs
    image_url: str
    raw_description: str
    # set by detect_category_node
    raw_category: str          # the raw VQA answer, e.g. "this is a grocery item..."
    product: str
    product_confirmed: bool
    display_category: str      # resolved, clean category name (e.g. "grocery")
    # set by resolve_schema_node
    critical_features: list[str]
    cosmetic_features: list[str]
    # set by extract_features_node
    description_features: dict
    # set by hard_stop_check_node
    hard_stop: bool
    # set by build_checklist_node
    checklist: dict
    checklist_critical: list[str]
    # set by verify_features_node / escalate_node
    results: dict
    ambiguous: list[str]
    escalated: bool
    # set by assemble_result_node
    verdict: str
    category: str
    critical_checked: dict
    other_checked: dict
    answer: str
    ambiguous_features: list[str]


def detect_category_node(state: PipelineState) -> dict:
    """Ask the image what category and product it shows. Always runs first — nothing here depends on the description."""
    category, product, product_confirmed = detect_category_and_product(state["image_url"])
    return {
        "raw_category": category,
        "product": product,
        "product_confirmed": product_confirmed,
        "display_category": _clean_category_name(category),
    }


def resolve_schema_node(state: PipelineState) -> dict:
    """Look up which features this category's policy marks as critical vs. cosmetic."""
    critical_features, cosmetic_features = get_checklist_features(state["raw_category"])
    return {"critical_features": critical_features, "cosmetic_features": cosmetic_features}


def extract_features_node(state: PipelineState) -> dict:
    """Map the listing's raw text onto the resolved category's known schema. The one LLM call most listings need."""
    description_features = extract_description_features(
        state["raw_description"], state["raw_category"],
        state["critical_features"], state["cosmetic_features"],
    )
    return {"description_features": description_features}


def hard_stop_check_node(state: PipelineState) -> dict:
    """Does the image-grounded product agree with what the description claims? A confident disagreement ends the check here."""
    resolved_product = state["product"] if state["product_confirmed"] else state["raw_category"]
    claimed_product_type = state["description_features"].get("product_type", "")
    if _hard_stop_mismatch(resolved_product, claimed_product_type):
        results = {"product_type": {"value": claimed_product_type, "match": False, "source": "detect_category/detect_product"}}
        return {
            "hard_stop": True, "results": results, "checklist_critical": ["product_type"],
            "escalated": False, "ambiguous": [],
        }
    return {"hard_stop": False}


def route_after_hard_stop(state: PipelineState) -> Literal["hard_stop", "continue"]:
    return "hard_stop" if state["hard_stop"] else "continue"


def build_checklist_node(state: PipelineState) -> dict:
    """Union the policy's critical features with any other concrete claim the description makes."""
    checklist, checklist_critical = _build_checklist(state["critical_features"], state["description_features"])
    return {"checklist": checklist, "checklist_critical": checklist_critical}


def verify_features_node(state: PipelineState) -> dict:
    """Check each checklist feature against the image: OCR first, a targeted question as fallback."""
    results, ambiguous = _deterministic_verify(state["image_url"], state["checklist"])
    claimed_product_type = state["description_features"].get("product_type", "")
    results["product_type"] = {"value": claimed_product_type, "match": True, "source": "detect_category/detect_product"}
    checklist_critical = state["checklist_critical"] + (["product_type"] if claimed_product_type else [])
    return {
        "results": results, "ambiguous": ambiguous, "checklist_critical": checklist_critical,
        "escalated": bool(ambiguous),
    }


def route_after_verify(state: PipelineState) -> Literal["escalate", "assemble"]:
    return "escalate" if state["ambiguous"] else "assemble"


def assemble_result_node(state: PipelineState) -> dict:
    """Compute the final verdict from everything gathered — any critical mismatch is a hard veto."""
    return _assemble_result(
        state["display_category"], state["results"], critical_feature_names=state["checklist_critical"],
        product=state.get("product"), product_confirmed=state.get("product_confirmed"),
        description_features=state.get("description_features", {}),
        hard_stop=state["hard_stop"], escalated=state.get("escalated", False),
        ambiguous_features=state.get("ambiguous", []),
    )


def make_escalate_node(escalation_graph):
    """
    Wraps the compiled escalation sub-graph as a single pipeline node. A
    factory rather than a plain function because the node needs the
    already-compiled sub-graph closed over it — building a fresh one on
    every call would reload the LLM binding for no reason.
    """
    def escalate_node(state: PipelineState) -> dict:
        """Investigate any feature that stayed ambiguous, using the escalation sub-graph."""
        ambiguous_results = {f: state["results"][f] for f in state["ambiguous"]}
        resolved = _escalate_ambiguous(escalation_graph, state["image_url"], ambiguous_results)
        results = dict(state["results"])
        for f, verdict in resolved.items():
            results[f]["match"] = verdict
        return {"results": results}
    return escalate_node


def build_graph():
    """
    Builds the full pipeline as one StateGraph — category/product detection,
    schema resolution, extraction, the hard-stop check, checklist building,
    feature verification, escalation, and verdict assembly are all nodes on
    the same graph, wired with two conditional edges (skip straight to the
    verdict on a hard-stop; only escalate if something stayed ambiguous).
    """
    escalation_graph = build_escalation_graph()
    escalate_node = make_escalate_node(escalation_graph)

    g = StateGraph(PipelineState)
    g.add_node("detect_category", detect_category_node)
    g.add_node("resolve_schema", resolve_schema_node)
    g.add_node("extract_features", extract_features_node)
    g.add_node("hard_stop_check", hard_stop_check_node)
    g.add_node("build_checklist", build_checklist_node)
    g.add_node("verify_features", verify_features_node)
    g.add_node("escalate", escalate_node)
    g.add_node("assemble_result", assemble_result_node)

    g.add_edge(START, "detect_category")
    g.add_edge("detect_category", "resolve_schema")
    g.add_edge("resolve_schema", "extract_features")
    g.add_edge("extract_features", "hard_stop_check")
    g.add_conditional_edges("hard_stop_check", route_after_hard_stop, {
        "hard_stop": "assemble_result",
        "continue": "build_checklist",
    })
    g.add_edge("build_checklist", "verify_features")
    g.add_conditional_edges("verify_features", route_after_verify, {
        "escalate": "escalate",
        "assemble": "assemble_result",
    })
    g.add_edge("escalate", "assemble_result")
    g.add_edge("assemble_result", END)

    return g.compile()


def verify_listing(graph, image_url: str, raw_description: str) -> dict:
    """
    Runs the full pipeline graph for one listing. `raw_description` is the
    listing's actual text — a title/description a seller wrote, e.g.
    "Prairie Farms Whole Milk, 1 Quart Carton" — not a pre-structured dict.

    `graph` is the compiled pipeline graph from build_graph(). Most listings
    complete it with only one LLM call (extraction); escalation only adds
    more if a feature genuinely can't be resolved deterministically.
    """
    initial_state: PipelineState = {
        "image_url": image_url,
        "raw_description": raw_description,
        "raw_category": "", "product": "", "product_confirmed": False, "display_category": "",
        "critical_features": [], "cosmetic_features": [],
        "description_features": {},
        "hard_stop": False,
        "checklist": {}, "checklist_critical": [],
        "results": {}, "ambiguous": [], "escalated": False,
        "verdict": "", "category": "", "critical_checked": {}, "other_checked": {},
        "answer": "", "ambiguous_features": [],
    }
    return graph.invoke(initial_state)
