# Marketplace Listing Integrity Agent

An agent that uses RAG and an LLM to decide whether a marketplace listing's image genuinely matches its text description.

> **Full article:** [Building a Marketplace Listing Integrity Agent with LangGraph](docs/article.md)

---

## The Problem

[product-image-description-alignment](https://github.com/ebiarian/product-image-description-alignment) built a working OCR + VQA pipeline for catching image–description mismatches — but it runs the same fixed sequence of checks on every listing, regardless of product category. That project's own roadmap named the gap directly:

> *"Category-aware feature sets — different product categories need different features."*

A wrong color is a return for a T-shirt and irrelevant for a phone charger. A wrong model number is a functional failure for a charger and meaningless for a T-shirt. Encoding that as a growing pile of `if category == ...` branches doesn't scale — and it isn't how a real trust & safety team would want the logic to live, either. This project reframes the same underlying problem as an **agent** decision: retrieve the right policy for this listing's category, then reason about what actually needs checking.

---

## Approach

An agent looks at the listing's image to figure out the category and product, looks up what that category needs checked (milk cares about variant and container; a phone charger cares about model number instead), turns the listing's text into structured data, then verifies each feature against the image. The LLM is only needed for two things: turning raw text into structured fields, and stepping in on the rare feature that can't be confidently resolved from the image alone.

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

| Node | Explanation | Model | Input example | Output example |
|---|---|---|---|---|
| `detect_category` | Asks the image alone what category and product it shows, before the description is ever read | Moondream2 (VQA) | product image | `category: "grocery, a carton of milk"`, `product: "milk"` |
| `resolve_schema` | Looks up the detected category's checklist — which features matter and their vocabularies | `nomic-embed-text` (embedding, RAG) | `"grocery"` | `critical: [brand, size, variant, container, product_type]` |
| `extract_features` | Maps the listing's raw text onto the fields the checklist just named | `qwen2.5:14b` | `"Prairie Farms Whole Milk, 1 Quart Carton"` | `{brand: "Prairie Farms", product_type: "Milk", variant: "whole", size: "1 quart", container: "carton"}` |
| `hard_stop_check` | Does the image-grounded product agree with the claimed `product_type`? | none (code) | `product: "milk"`, claimed `"Milk"` | `hard_stop: False` |
| `build_checklist` | Unions the required features with any other concrete claim in the description | none (code) | required features + extracted fields | `checklist: {brand, size, variant, container}` |
| `verify_features` | Checks each checklist feature against the image: OCR first, a targeted VQA question as fallback | EasyOCR + Moondream2 (VQA) | `checklist`, image | `{brand: {value: "Prairie Farms", match: True, source: "OCR"}, ...}` |
| `escalate` | Investigates anything still ambiguous after OCR + VQA — the only other LLM call, and rare | `qwen2.5:14b` (agent + tools) | ambiguous feature + inconclusive VQA answer | `match \| mismatch \| unconfirmed` |
| `assemble_result` | Computes the final verdict from code — any critical mismatch is a hard veto | none (code) | verified results | `verdict: "Likely match"` |

---

## Project Structure

```
marketplace-listing-integrity-langgraph-agent/
├── src/
│   ├── config.py             # Model names, Chroma settings, recursion limit
│   ├── policy_corpus.py      # Category policy documents + critical/cosmetic feature lists + common_products + bounded_features vocabularies (the RAG corpus)
│   ├── retrieval.py          # Chroma vector store + retrieve_category_policy / get_checklist_features / get_bounded_vocabulary (exact ID lookup, semantic fallback)
│   ├── vision_tools.py       # OCR + Moondream2 VQA: detect_category_and_product (deterministic), read_label_text, ask_vision_question (LLM-facing), normalization/matching helpers
│   ├── extraction.py         # extract_description_features() — the one genuine LLM-judgment step: raw text -> structured schema
│   ├── tools.py              # ALL_TOOLS — only the two tools still exposed to the LLM's narrow escalation loop
│   ├── listings.py           # Synthetic test listings (real product images + raw description text, injected mismatches)
│   └── agent.py              # Deterministic pipeline + extraction + narrow escalation sub-graph + verify_listing()
├── notebooks/
│   └── 01_marketplace_listing_integrity_agent.ipynb
└── environment.yml
```

---

## Environment Setup

```bash
conda env create -f environment.yml
conda activate listing-integrity-agent
```

This project runs entirely on local models — no cloud API keys required. Pull both Ollama models before running the notebook:

```bash
ollama pull qwen2.5:14b        # reasoning — the agent's brain
ollama pull nomic-embed-text   # embeddings — for the RAG policy corpus
ollama serve
```

`qwen2.5:14b` (~9GB) matches the default already validated on a 16GB M2 Mac in the sibling [zone-balancing-ridesharing-langgraph-agent](https://github.com/ebiarian/zone-balancing-ridesharing-langgraph-agent) project. Moondream2 (~4GB, float16) loads the same way it did there.

---

## Usage

```bash
jupyter notebook notebooks/01_marketplace_listing_integrity_agent.ipynb
```

The notebook builds the RAG policy corpus, walks through each pipeline node individually, builds and diagrams the full `StateGraph`, then runs it against five test listings — a correct one and four with injected mismatches — timing each and reporting the verdict and whether escalation was needed.

### Use in your own code

```python
from src.agent import build_graph, verify_listing
from src.retrieval import build_policy_corpus

build_policy_corpus()   # embeds the category policy corpus into Chroma (once)
graph = build_graph()   # the narrow escalation sub-graph — only invoked if needed

result = verify_listing(
    graph,
    image_url="https://example.com/product.jpg",
    raw_description="Prairie Farms Whole Milk, 1 Quart Carton",  # real listing text, not a pre-structured dict
)

print(result["answer"])         # Verdict / Category / Critical features / Reasoning
print(result["escalated"])      # True only if a feature needed the LLM sub-loop
```

---

## Limitations

- **Category corpus is small and synthetic** — three categories plus a default, written for this project, not sourced from a real trust & safety policy document
- **`common_products` lists are hand-curated and narrow** — a genuinely novel product within a known category (e.g. a grocery item that isn't milk/coffee/butter/chicken/bread/juice) still gets forced into `"something else"` by `detect_product`, the same "not exhaustive at marketplace scale" limitation the original `product-image-description-alignment` project already named for its own VQA prompts — just relocated here, one layer up
- **`feature_questions` are generic, one template per feature name reused across every category** — a real system would likely need category-specific phrasing, not "What is the size or quantity of this product?" for both a gallon of milk and a pair of headphones
- **`bounded_features` vocabularies (`variant`, `container`) are hand-curated and grocery-only** — electronics and apparel don't have one yet, so any bounded-ish field they might need (e.g. a closed list for `color`) would currently extract as open text, untested
- **The hard-stop and self-consistency checks have no escalation path** — unlike feature-level ambiguity, a genuinely unclear product/category disagreement is resolved by a fixed rule (trust category, treat product as unconfirmed) rather than investigated further by the LLM
- **Live testing is grocery-only** — the RAG corpus includes electronics and apparel policies and retrieval correctly discriminates between all three, but the full agent has only been exercised end-to-end on real grocery product images
- **No human-in-the-loop yet** — every verdict is fully automatic; a production system would likely gate high-risk verdicts behind human review, the way the sibling ridesharing project gates critical-severity decisions with `interrupt()`

---

## License

[MIT](LICENSE)
