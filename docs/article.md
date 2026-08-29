# Building a Marketplace Listing Integrity Agent with LangGraph

---

You're shopping for milk online. The photo shows a Prairie Farms carton. The listing says Organic Valley. You hesitate, maybe abandon the cart, maybe order anyway and find out the hard way that the label lied.

That small moment of doubt is a real cost at marketplace scale — lower conversion, more returns, less trust in the platform. A previous project of mine, [product-image-description-alignment](https://github.com/ebiarian/product-image-description-alignment), built a working pipeline to catch exactly this: OCR reads the label, a vision-language model answers targeted questions about what it sees, and the two get compared against the listing's claims. It worked. It also had a gap its own roadmap named directly:

> *"Category-aware feature sets — different product categories need different features."*

That pipeline checks the same five things — brand, size, variant, container, product type — on every listing, whether it's milk or a phone charger. A wrong color is a return for a T-shirt and irrelevant for a charger. Encoding that as a growing pile of `if category == ...` branches doesn't scale, and it's not how a real trust & safety team would want the logic to live.

This project picks up that gap, using [LangGraph](https://www.langchain.com/langgraph): retrieve the right verification policy for a listing's category, then reason about what actually needs checking.

<div align="center">
  <img width="600" height="200" alt="article_2_product_pics" src="https://github.com/user-attachments/assets/3bc36fb4-348e-41de-aaa1-e391d76e3b6c" />
  <br/>
  <em>Left to right: Prairie Farms Whole Milk, Amazon Fresh Chicken Breast, Land O Lakes Salted Butter — the three products used in all experiments </em>
</div>

## The Agent

The agent runs a fixed sequence for every listing. First it looks at the image directly: what category is this, and what specific product is it? That resolved category drives a RAG lookup against a small knowledge base of category policies — each one naming which features matter for that category, what's critical versus cosmetic, and what vocabulary those features actually use. An embedding model retrieves the right policy for the detected category: grocery listings get checked on brand, size, variant, and container; electronics get checked on brand, model number, and key specs instead; apparel gets size, color, and material. None of that mapping is hardcoded in the pipeline's own code. It lives in the corpus, and the embedding model is what connects a listing to the right policy — and, by extension, the right questions to ask about it.

That retrieved policy also shapes the next step: turning the listing's raw text into structured data. A seller writes `"Prairie Farms Whole Milk, 1 Quart Carton,"` not a JSON object. Something has to build `{"brand": "prairie farms", "size": "1 quart", "variant": "whole", ...}`, and the shape of that JSON comes straight from the category's policy. This is where the LLM does its real work: reading free text and mapping it onto the exact fields the policy just named. Most fields — brand, size — get extracted as open text. A field with a genuinely bounded real-world vocabulary, like `variant` (whole, 2%, skim, salted, unsalted, boneless-skinless, and so on), gets matched against that vocabulary instead of extracted freely — the corpus carries the list, and the LLM picks the closest match rather than inventing its own phrasing:

```python
fields[f] = (
    Literal[tuple(bounded_vocab[f])],
    Field(description=f"Pick the closest matching {f} from the allowed list."),
)
```

With category, product, and a structured description in hand, the agent checks whether the image and the description even agree on what the product *is*, before checking anything else. Then it verifies each feature the policy marked critical: OCR reads the label first, and a targeted question to the image fills in whatever OCR missed. If a feature still can't be confidently matched or contradicted, it escalates to a small LLM loop for a closer look.

Here's every node, what it does, and what it runs on — using the milk listing (`"Prairie Farms Whole Milk, 1 Quart Carton"`) as the running example:

| Node | Explanation | Model | Input example | Output example |
|---|---|---|---|---|
| `detect_category` | Asks the image alone what category and product it shows, before the description is ever read | Moondream2 (VQA) | product image | `category: "grocery, a carton of milk"`, `product: "milk"` |
| `resolve_schema` | Retrieves the detected category's policy — its critical/cosmetic features and vocabularies | `nomic-embed-text` (embedding, RAG) | `"grocery"` | `critical: [brand, size, variant, container, product_type]` |
| `extract_features` | Maps the listing's raw text onto the fields the policy just named | `qwen2.5:14b` | `"Prairie Farms Whole Milk, 1 Quart Carton"` | `{brand: "Prairie Farms", product_type: "Milk", variant: "whole", size: "1 quart", container: "carton"}` |
| `hard_stop_check` | Does the image-grounded product agree with the claimed `product_type`? | none (code) | `product: "milk"`, claimed `"Milk"` | `hard_stop: False` |
| `build_checklist` | Unions the policy's critical features with any other concrete claim in the description | none (code) | critical features + extracted fields | `checklist: {brand, size, variant, container}` |
| `verify_features` | Checks each checklist feature against the image: OCR first, a targeted VQA question as fallback | EasyOCR + Moondream2 (VQA) | `checklist`, image | `{brand: {value: "Prairie Farms", match: True, source: "OCR"}, ...}` |
| `escalate` | Investigates anything still ambiguous after OCR + VQA — the only other LLM call, and rare | `qwen2.5:14b` (agent + tools) | ambiguous feature + inconclusive VQA answer | `match \| mismatch \| unconfirmed` |
| `assemble_result` | Computes the final verdict from code — any critical mismatch is a hard veto | none (code) | verified results | `verdict: "Likely match"` |

## The Graph

Here's the shape of the whole pipeline, end to end:

```
detect_category
      |
resolve_schema
      |
extract_features
      |
hard_stop_check
      | (conditional edge)
 -----+------------------
 |                       |
 v continue              v hard stop
build_checklist          |
      |                  |
verify_features          |
      | (conditional edge)
 -----+------------      |
 |                  |     |
 v resolved         v ambiguous
 |               escalate |
 -----+------------+------
      |
assemble_result
      |
     END
```

Two conditional edges are the only real branching in the graph. `hard_stop_check` compares what the image actually shows against what the description claims the product *is* — not a feature like brand or size, the product itself. If a milk carton is described as orange juice, that's caught here, immediately, before any brand or size checking wastes a single tool call. `verify_features` only routes through `escalate` if a feature stays ambiguous after OCR and a targeted VQA question both fail to confidently confirm or contradict it. Every other edge is fixed.

## The State

Every node reads from and writes to the same shared `TypedDict`:

```python
class PipelineState(TypedDict):
    image_url: str
    raw_description: str
    raw_category: str
    product: str
    product_confirmed: bool
    display_category: str
    critical_features: list[str]
    cosmetic_features: list[str]
    description_features: dict
    hard_stop: bool
    checklist: dict
    checklist_critical: list[str]
    results: dict
    ambiguous: list[str]
    escalated: bool
    verdict: str
    category: str
    critical_checked: dict
    other_checked: dict
    answer: str
    ambiguous_features: list[str]
```

Nothing here is LangGraph-specific magic — it's a dictionary shape the framework makes sure every node agrees on.

## Wiring the Graph

Escalation is itself a small compiled sub-graph — an agent-and-tools loop bound to only two tools, `read_label_text` and `ask_vision_question` — invoked as a single atomic node in the larger graph, the same "sub-graph as one node" pattern the ride-share project used for coordinating two zones at once in its final stage.

```python
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

graph = g.compile()
```

That's the whole routing logic — two conditional branches, expressed as data, and everything else a straight line.

## Running It

Across five test listings — a correct one, a wrong brand, a wrong size, a wrong product entirely, and one more correct listing as a control — on `qwen2.5:14b` for extraction and, when needed, escalation:

| Listing | Verdict | Time |
|---|---|---|
| Prairie Farms Whole Milk — correct | Likely match | 23.8s |
| Prairie Farms Milk — wrong brand | Likely mismatch | 8.0s |
| Amazon Fresh Chicken Breast — correct | Likely match | 32.0s |
| Land O Lakes Salted Butter — wrong size | Likely mismatch | 13.7s |
| Milk photo, described as juice | Likely mismatch | 6.1s |

**5 of 5 correct**, average 16.7 seconds, zero escalations needed on any of them.

## Why Not Just Let the LLM Decide Everything

The design above wasn't the first one. The obvious starting point is a ReAct loop — give the model every tool (detect category, detect product, retrieve the policy, read the label, ask the image a question) and let it decide the order for itself, the way a human investigator would.

I built that first, and it was slow: 8–12 sequential reasoning turns per listing, roughly 1–2 minutes locally. Worse than the speed was what happened when a claim couldn't be confirmed. Given a listing claiming `brand: organic valley` against a Prairie Farms carton, the model would sometimes skip the verification tool entirely and just trust the claim — its own words, *"brand can be inferred from context."* That's the exact failure this project exists to catch, now baked into the checker itself.

Most of those 8–12 turns weren't real decisions anyway — "call `detect_category`, then `detect_product`" is the same fixed sequence every time, nothing for a model to weigh. The lesson wasn't new; it's the same one from an earlier LangGraph project of mine on ride-share zone balancing: *narrow the LLM's job to what a lookup table genuinely can't do.* That's what led to the pipeline above.

Before fully committing, I ran one more test: could full ReAct be salvaged by constraining every LLM output to a strict schema, keeping agentic control while fixing reliability structurally? Tested on `qwen2.5:7b`: **3 of 5 correct, 36.9 seconds per listing.** It fixed the brand-skipping bug, but new failures took its place — contradicting its own judgments between attempts, and one run hit LangGraph's recursion limit outright, stuck in a loop.

I tried `qwen3:8b` next, built with agentic tool-use in mind rather than raw reasoning depth. On the listing that had previously crashed: correct verdict, clean single pass — and **174.5 seconds**, roughly five times slower than `qwen2.5:7b`, sixteen times slower than the deterministic pipeline.

A newer model genuinely closes the *reliability* gap. It does nothing for the *speed* gap, because that gap was never about model quality — a full ReAct loop makes 8–12 sequential decisions regardless of which model makes them, and most were never real choices to begin with.

## What This Agent Does

- Detects category and product straight from the image, before trusting anything the description claims — a listing that lies about its own product is caught immediately, not after checking four other features first
- Extracts structured fields from a listing's raw text with exactly one LLM call, using a closed vocabulary for the one field that genuinely needed it
- Retrieves each category's verification policy from a small RAG corpus, so different product categories check different things instead of one fixed feature list for everything
- Verifies each feature deterministically — OCR first, a targeted question as fallback — and escalates to a narrow LLM loop only when that genuinely isn't enough
- Computes the final verdict from code, not generation: any critical-feature mismatch is a hard veto, no exceptions

## Limitations

The category/product consistency check has no escalation path of its own — a genuinely unclear disagreement falls back to a fixed rule rather than getting investigated further. The bounded-vocabulary lists only exist for grocery; electronics and apparel would fall back to open text extraction on anything similarly bounded, untested. Live testing overall is still grocery-only. And there's no human-in-the-loop gate yet — every verdict executes automatically, with no pause for a high-risk category or a repeat offender before it does.

---

**Code for this project:** [github.com/ebiarian/marketplace-listing-integrity-langgraph-agent](https://github.com/ebiarian/marketplace-listing-integrity-langgraph-agent)
