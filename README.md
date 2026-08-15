# Marketplace Listing Integrity Agent

A LangGraph **ReAct** agent that decides whether a marketplace listing's image genuinely matches its text description — grounded by a **RAG** knowledge base of category-specific verification policy, with an **LLM as the brain** deciding which tools to call, in what order, and when it has enough evidence to stop.

---

## The Problem

[product-image-description-alignment](https://github.com/ebiarian/product-image-description-alignment) built a working OCR + VQA pipeline for catching image–description mismatches — but it runs the same fixed sequence of checks on every listing, regardless of product category. That project's own roadmap named the gap directly:

> *"Category-aware feature sets — different product categories need different features."*

A wrong color is a return for a T-shirt and irrelevant for a phone charger. A wrong model number is a functional failure for a charger and meaningless for a T-shirt. Encoding that as a growing pile of `if category == ...` branches doesn't scale — and it isn't how a real trust & safety team would want the logic to live, either. This project reframes the same underlying problem as an **agent** decision: retrieve the right policy for this listing's category, then reason about what actually needs checking.

---

## Approach

One agent, one article — not a staged series. Five models, each with a distinct job, orchestrated by a standard LangGraph ReAct loop:

| Stage | What happens | Model |
|---|---|---|
| 1. Detect category | short constrained VQA question, image-grounded — not from the (untrusted) description | VQA (Moondream2) |
| 2. Detect specific product | scoped to that category's known products | VQA (Moondream2) |
| 3. Self-consistency check | does the detected product belong to the detected category? Retry once if not, then fall back to trusting category alone | LLM (brain) |
| 4. Hard-stop vs. description | does the image-grounded product agree with what the description claims? Disagreement ends the check immediately | LLM (brain) |
| 5. Retrieve category policy | exact ID lookup if the category is known; semantic search as fallback | Embedding model (nomic-embed-text) |
| 6. Build checklist | union of the policy's critical features and any other concrete claim in the description | LLM (brain) |
| 7. Verify each feature | OCR first, VQA fallback, loops until enough evidence | OCR / VQA |
| 8. Final verdict | hard veto on any critical-feature mismatch | LLM (brain) |

**Why category detection starts from the image, not the description.** The whole point of this project is to check whether the description can be trusted — so nothing here is allowed to trust it first. Category and product identity are established from the image alone (stages 1–2) before the description's own claims are even looked at (stage 4). A description that lies about `product_type` entirely (e.g. milk photographed, described as juice) is caught immediately here, before any brand/size/variant checking happens.

**Why detection is two questions, not one.** Moondream2 is small enough that a single open-ended "what product is this?" question — or one flat list spanning every category — is unreliable. A short, category-scoped multiple-choice question is meaningfully more reliable, which is why category (stage 1) and product (stage 2) are asked separately, the second scoped to the first's answer.

**Why retrieval isn't always semantic search.** Once category is resolved, it's already one of the corpus's own document IDs — an exact lookup is strictly more reliable than embedding similarity for that case. RAG's semantic search is reserved for the fallback path, when the detected category doesn't cleanly resolve (`"something else"`) — that's where it actually earns its keep.

The OCR and VQA extraction logic itself is ported directly from `product-image-description-alignment` — this project doesn't reinvent feature extraction, it changes who decides *when* and *what* to extract, and adds the category/product detection layer that pipeline never had.

---

## Project Structure

```
marketplace-listing-integrity-langgraph-agent/
├── src/
│   ├── config.py             # Model names, Chroma settings, recursion limit
│   ├── policy_corpus.py      # Category policy documents + common_products (the RAG corpus)
│   ├── retrieval.py          # Chroma vector store + retrieve_category_policy (exact ID lookup, semantic fallback)
│   ├── vision_tools.py       # OCR + Moondream2 VQA tools: detect_category, detect_product, read_label_text, ask_vision_question
│   ├── tools.py              # ALL_TOOLS — the full tool list the agent can call
│   ├── listings.py           # Synthetic test listings (real product images, injected mismatches)
│   └── agent.py              # The ReAct graph: agent <-> tools, system prompt, verify_listing()
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

The notebook builds the RAG policy corpus, demonstrates retrieval in isolation — both the exact-ID fast path and the semantic-search fallback — defines the agent's five tools, builds and diagrams the LangGraph graph, then runs the agent against five listings: a correct one, two with injected critical-feature mismatches (wrong brand, wrong size), one more correct listing as a control, and one with a wrong product entirely — printing the full tool-call reasoning trace for each. The last case is worth watching closely: it should resolve in visibly fewer tool calls than the others, since the mismatch is caught at the category/product detection stage before any brand/size/variant checking begins.

### Use in your own code

```python
from src.agent import build_graph, verify_listing
from src.retrieval import build_policy_corpus

build_policy_corpus()   # embeds the category policy corpus into Chroma (once)
graph = build_graph()

result = verify_listing(
    graph,
    image_url="https://example.com/product.jpg",
    description_features={
        "product_type": "milk",
        "brand":        "prairie farms",
        "size":         "1 quart",
        "variant":      "whole",
        "container":    "carton",
    },
)

# result["messages"][-1].content holds the final Verdict / Reasoning text
print(result["messages"][-1].content)
```

---

## Limitations

- **Category corpus is small and synthetic** — three categories plus a default, written for this project, not sourced from a real trust & safety policy document
- **`common_products` lists are hand-curated and narrow** — a genuinely novel product within a known category (e.g. a grocery item that isn't milk/coffee/butter/chicken/bread/juice) still gets forced into `"something else"` by `detect_product`, the same "not exhaustive at marketplace scale" limitation the original `product-image-description-alignment` project already named for its own VQA prompts — just relocated here, one layer up
- **Live testing is grocery-only** — the RAG corpus includes electronics and apparel policies and retrieval correctly discriminates between all three, but the full agent has only been exercised end-to-end on real grocery product images
- **Local Ollama reasoning** — `qwen2.5:14b` is capable but smaller and less reliable at multi-step tool orchestration than a frontier cloud model; a genuinely ambiguous listing may exhaust the one-time self-consistency retry without truly resolving
- **No human-in-the-loop yet** — every verdict is fully automatic; a production system would likely gate high-risk verdicts behind human review, the way the sibling ridesharing project gates critical-severity decisions with `interrupt()`

---

## License

[MIT](LICENSE)
