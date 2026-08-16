# Marketplace Listing Integrity Agent

A hybrid pipeline that decides whether a marketplace listing's image genuinely matches its text description — grounded by a **RAG** knowledge base of category-specific verification policy, with an **LLM** reserved for the small number of steps that are genuine judgment calls, orchestrated as a narrow **LangGraph ReAct** loop rather than deciding every step.

---

## The Problem

[product-image-description-alignment](https://github.com/ebiarian/product-image-description-alignment) built a working OCR + VQA pipeline for catching image–description mismatches — but it runs the same fixed sequence of checks on every listing, regardless of product category. That project's own roadmap named the gap directly:

> *"Category-aware feature sets — different product categories need different features."*

A wrong color is a return for a T-shirt and irrelevant for a phone charger. A wrong model number is a functional failure for a charger and meaningless for a T-shirt. Encoding that as a growing pile of `if category == ...` branches doesn't scale — and it isn't how a real trust & safety team would want the logic to live, either. This project reframes the same underlying problem as an **agent** decision: retrieve the right policy for this listing's category, then reason about what actually needs checking.

---

## Approach

One agent, one article — not a staged series. The first working version put every step behind an LLM decision — genuine ReAct end to end. Testing surfaced two problems: it took ~1-2 minutes per listing locally (8-12 sequential LLM reasoning turns, each re-processing a growing context), and most of those turns weren't actually judgment calls — "call `detect_category`, then `detect_product`" is the same fixed sequence every time, with nothing for an LLM to reason about. Worse, making the OCR→VQA fallback an LLM *choice* meant the model sometimes skipped it — "inferring" an unconfirmed brand from context instead of actually checking — which is precisely the failure mode this project exists to catch.

The current version keeps the LLM only where testing showed it's genuinely needed — the same lesson the sibling ridesharing project already learned for its own Stage 1/2 split:

| Stage | What happens | Needs an LLM? |
|---|---|---|
| 1. Detect category, then product | two short constrained VQA questions, image-grounded — not from the (untrusted) description | No — fixed order, no judgment |
| 2. Self-consistency check + retry | does the product belong to the category? A membership check, with a fixed one-time retry rule | No |
| 3. Hard-stop vs. description | does the image-grounded product agree with the claim? Normalized string comparison | No |
| 4. Retrieve category policy | exact ID lookup if known, semantic search as fallback | No — embedding model only |
| 5. Build checklist | union of the policy's critical features and any other concrete claim in the description | No — set logic |
| 6. Verify each feature | OCR first, then a fixed-template VQA fallback question, both matched via normalized string comparison | No |
| 7. Resolve anything still ambiguous | **only if step 6 left something genuinely unresolved** | **Yes — the only LLM involvement** |
| 8. Final verdict | hard veto on any critical mismatch | No — computed, not generated |

A listing where every feature resolves cleanly in step 6 never invokes the LLM at all.

**Why category detection starts from the image, not the description.** The whole point of this project is to check whether the description can be trusted — so nothing here is allowed to trust it first. Category and product identity are established from the image alone (stages 1–2) before the description's own claims are even looked at (stage 3). A description that lies about `product_type` entirely (e.g. milk photographed, described as juice) is caught immediately here, before any brand/size/variant checking happens.

**Why detection is two questions, not one.** Moondream2 is small enough that a single open-ended "what product is this?" question — or one flat list spanning every category — is unreliable. A short, category-scoped multiple-choice question is meaningfully more reliable, which is why category (stage 1) and product are asked separately, the second scoped to the first's answer.

**Why retrieval isn't always semantic search.** Once category is resolved, it's already one of the corpus's own document IDs — an exact lookup is strictly more reliable than embedding similarity for that case. RAG's semantic search is reserved for the fallback path, when the detected category doesn't cleanly resolve (`"something else"`) — that's where it actually earns its keep.

**Why feature verification is deterministic, and what that fixed.** OCR output is one unstructured blob of every text fragment on a label; testing found a model would sometimes grab the wrong number as "the size" (a nutrition-panel fragment instead of the actual size claim). The fix: search the OCR text for whether the *claimed* value is present, rather than asking a model to independently interpret which fragment means what — the same substring-matching approach the original deterministic pipeline always used. Normalizing values (lowercase, strip punctuation, fold plural/singular) before comparing also fixed a false "chicken breast vs. chicken breasts" mismatch a model kept flagging despite an explicit prompt rule against it. Making both fixes structural rather than prompt-based means they can't be "forgotten" by a smaller or less reliable model.

**Why the LLM only appears in step 7.** Everything through step 6 is either a fixed sequence or a comparison with a confident answer. The LLM is reserved for genuinely ambiguous cases — a VQA answer that neither clearly confirms nor clearly contradicts a claim — via a narrow LangGraph ReAct sub-graph bound to only two tools (`read_label_text`, `ask_vision_question`), not the full original tool list.

The OCR and VQA extraction logic itself is ported directly from `product-image-description-alignment` — this project doesn't reinvent feature extraction, it changes who decides *when* and *what* to extract, and adds the category/product detection layer that pipeline never had.

---

## Project Structure

```
marketplace-listing-integrity-langgraph-agent/
├── src/
│   ├── config.py             # Model names, Chroma settings, recursion limit
│   ├── policy_corpus.py      # Category policy documents + structured critical/cosmetic feature lists + common_products (the RAG corpus)
│   ├── retrieval.py          # Chroma vector store + retrieve_category_policy / get_checklist_features (exact ID lookup, semantic fallback)
│   ├── vision_tools.py       # OCR + Moondream2 VQA: detect_category_and_product (deterministic), read_label_text, ask_vision_question (LLM-facing), normalization/matching helpers
│   ├── tools.py              # ALL_TOOLS — only the two tools still exposed to the LLM's narrow escalation loop
│   ├── listings.py           # Synthetic test listings (real product images, injected mismatches)
│   └── agent.py              # Deterministic pipeline + narrow escalation sub-graph + verify_listing()
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

The notebook builds the RAG policy corpus, demonstrates retrieval in isolation, walks through the deterministic category/product detection and feature verification steps individually (so you can see exactly what resolves without an LLM call), builds and diagrams the narrow escalation sub-graph, then runs the full pipeline against five listings — a correct one, three with injected mismatches (wrong brand, wrong size, wrong product entirely), and one more correct listing as a control — timing each and reporting whether escalation was needed. Most listings should complete in a few seconds with `escalated: False`; only genuinely ambiguous cases should reach the LLM at all.

### Use in your own code

```python
from src.agent import build_graph, verify_listing
from src.retrieval import build_policy_corpus

build_policy_corpus()   # embeds the category policy corpus into Chroma (once)
graph = build_graph()   # the narrow escalation sub-graph — only invoked if needed

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

print(result["answer"])         # Verdict / Category / Critical features / Reasoning
print(result["escalated"])      # True only if a feature needed the LLM sub-loop
```

---

## Limitations

- **Category corpus is small and synthetic** — three categories plus a default, written for this project, not sourced from a real trust & safety policy document
- **`common_products` lists are hand-curated and narrow** — a genuinely novel product within a known category (e.g. a grocery item that isn't milk/coffee/butter/chicken/bread/juice) still gets forced into `"something else"` by `detect_product`, the same "not exhaustive at marketplace scale" limitation the original `product-image-description-alignment` project already named for its own VQA prompts — just relocated here, one layer up
- **`FEATURE_QUESTION_TEMPLATES` are generic, one template per feature name reused across every category** — a real system would likely need category-specific phrasing, not "What is the size or quantity of this product?" for both a gallon of milk and a pair of headphones
- **The hard-stop and self-consistency checks have no escalation path** — unlike feature-level ambiguity, a genuinely unclear product/category disagreement is resolved by a fixed rule (trust category, treat product as unconfirmed) rather than investigated further by the LLM
- **Live testing is grocery-only** — the RAG corpus includes electronics and apparel policies and retrieval correctly discriminates between all three, but the full agent has only been exercised end-to-end on real grocery product images
- **No human-in-the-loop yet** — every verdict is fully automatic; a production system would likely gate high-risk verdicts behind human review, the way the sibling ridesharing project gates critical-severity decisions with `interrupt()`

---

## License

[MIT](LICENSE)
