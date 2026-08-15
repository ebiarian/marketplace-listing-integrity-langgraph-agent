import operator
from typing import Annotated

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict

from src.config import OLLAMA_MODEL, RECURSION_LIMIT
from src.tools import ALL_TOOLS

# The LLM is the brain here, not a narrow structured-output call like in the
# sibling ridesharing project's Stage 2. It decides, one step at a time,
# whether it has enough evidence yet or needs another tool call — genuine
# ReAct, not a fixed pipeline that always runs every tool in the same order.
SYSTEM_PROMPT = """You are a marketplace listing integrity agent. Given a product image URL and its text description, decide whether the image and description genuinely describe the same product.

Work step by step:

1. Call detect_category first — this determines the product's category directly from the image, not from the listing's (untrusted) description text.

2. Call detect_product next, passing the category from step 1 — this determines the specific product directly from the image, scoped to that category's known products.

3. Self-consistency check: does the product from step 2 plausibly belong to the category from step 1? If not, the image evidence is inconsistent — call detect_category and detect_product again, once. If still inconsistent after that one retry, trust detect_category's answer only and treat the specific product identity as unconfirmed; do not force a guess, and proceed using the category alone.

4. Compare the resolved product (or category, if product identity stayed unconfirmed) against the description's own claimed product_type. If they clearly disagree — the image shows a genuinely different product than the description claims — stop here and give your final answer immediately: this is the strongest possible mismatch signal and no further checking is needed.

5. If the product is consistent with the description, call retrieve_category_policy with the resolved category to learn which features are critical (any mismatch is an automatic flag) versus cosmetic (a mismatch is only a soft warning).

6. Build your checklist: every feature the retrieved policy marks as critical, PLUS any other specific, checkable claim present in the listing's description even if the policy doesn't name that feature (e.g. a claimed weight in a category whose policy doesn't list weight as critical — check it anyway, it just carries less weight in the final verdict).

7. Check each checklist item against the image: read_label_text (OCR) first, falling back to ask_vision_question only when OCR doesn't find it or the feature is a purely visual property.

8. You do not need to check every item once you already have enough evidence — if you've confirmed one critical-feature mismatch, you can stop and give your final answer immediately, since the verdict is already decided.

9. Once you have enough evidence, stop calling tools and give your final answer as plain text in exactly this format:

Verdict: <Likely match | Uncertain | Likely mismatch>
Category: <resolved category>
Critical features checked: <feature: match/mismatch, ...>
Other claims checked: <feature: match/mismatch, ... (if any)>
Reasoning: <1-3 sentences>

Any single critical-feature mismatch — including the product/category check in step 4 — means the verdict must be "Likely mismatch". Critical features carry a hard veto. A mismatch on a cosmetic feature or an "other claim" from step 6 alone should not push the verdict below "Uncertain"."""


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]


def build_graph():
    """
    Standard LangGraph ReAct shape: an `agent` node (the LLM, bound to tools)
    and a `tools` node (ToolNode), wired with `tools_condition` so the graph
    loops agent -> tools -> agent for as many steps as the model decides it
    needs, and only exits to END once the model responds without a tool call.
    """
    llm = ChatOllama(model=OLLAMA_MODEL, temperature=0)
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    def agent_node(state: AgentState) -> AgentState:
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(ALL_TOOLS))

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")

    return graph.compile()


def verify_listing(graph, image_url: str, description_features: dict) -> dict:
    """
    Runs the compiled ReAct agent against one listing. description_features
    is a plain feature_name -> expected_value dict, the same shape used
    throughout product-image-description-alignment.
    """
    desc_lines = "\n".join(f"- {k}: {v}" for k, v in description_features.items())
    human_msg = HumanMessage(content=(
        f"Product image: {image_url}\n"
        f"Listed description features:\n{desc_lines}\n\n"
        "Verify whether the image matches this description."
    ))
    return graph.invoke(
        {"messages": [SystemMessage(content=SYSTEM_PROMPT), human_msg]},
        config={"recursion_limit": RECURSION_LIMIT},
    )
