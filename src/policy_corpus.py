# The RAG knowledge base this agent retrieves from. Each entry is a short,
# self-contained policy document for one product category, written the way an
# internal trust & safety wiki page would be — critical features (any
# mismatch is an automatic flag) versus cosmetic features (a mismatch is only
# a soft warning). This is what the agent actually reasons over: retrieval
# changes WHICH features it bothers checking and how strictly, not just what
# text gets pasted into the prompt.
#
# common_products backs the two-question VQA category/product detection in
# vision_tools.py: Moondream2 is small enough that a single open-ended "what
# product is this?" question is unreliable, but a short, category-scoped
# multiple-choice question is not — see the project README for why. Keeping
# the list here, not duplicated in vision_tools.py, is what stops the
# question options and the RAG corpus from silently drifting apart. "default"
# has no common_products: it's never something VQA is asked to detect, only
# a fallback retrieval lands on when the image genuinely doesn't fit any
# known category.
CATEGORY_POLICIES = [
    {
        "category": "grocery",
        "common_products": ["milk", "coffee", "butter", "chicken", "bread", "juice"],
        "text": (
            "Category: grocery (e.g. milk, chicken, butter, produce, packaged food).\n"
            "Critical features — any mismatch here is an automatic listing flag: "
            "brand, product_type, size/quantity, variant (e.g. whole vs 2%, salted vs "
            "unsalted, boneless vs bone-in), container type.\n"
            "Notes: Grocery buyers are highly sensitive to size and variant because a "
            "wrong delivery is a real fulfilment failure — 8oz vs 16oz butter, or "
            "boneless vs bone-in chicken, is not a cosmetic difference. Container "
            "material (carton vs plastic vs box) is still critical because it often "
            "correlates with an entirely different product line, not just packaging."
        ),
    },
    {
        "category": "electronics",
        "common_products": ["headphones", "a phone case", "a charger", "a cable"],
        "text": (
            "Category: electronics (e.g. headphones, phone cases, chargers, cables).\n"
            "Critical features — any mismatch here is an automatic listing flag: "
            "brand, model_number, product_type, key_spec (e.g. wattage, storage "
            "capacity, connector type).\n"
            "Cosmetic / tolerant features — a mismatch here is a warning, not an "
            "automatic flag: color, packaging_style.\n"
            "Notes: Electronics buyers care about functional compatibility far more "
            "than appearance — a black vs white charger is the same product "
            "functionally, but a wrong model number can mean the item physically "
            "does not fit or work as described."
        ),
    },
    {
        "category": "apparel",
        "common_products": ["a t-shirt", "a jacket", "shoes"],
        "text": (
            "Category: apparel (e.g. t-shirts, jackets, shoes).\n"
            "Critical features — any mismatch here is an automatic listing flag: "
            "size, color, product_type, material.\n"
            "Cosmetic / tolerant features — a mismatch here is a warning, not an "
            "automatic flag: brand.\n"
            "Notes: Apparel buyers are highly sensitive to size and color mismatches "
            "since these directly affect fit and satisfaction. Brand is comparatively "
            "less critical here because manufacturing is frequently shared or "
            "white-labelled across nominally different brand tags."
        ),
    },
    {
        "category": "default",
        "text": (
            "Category: default / general — used when no more specific category "
            "policy is a good match for the listing.\n"
            "Critical features — any mismatch here is an automatic listing flag: "
            "brand, product_type, size.\n"
            "Notes: Apply this conservative general-purpose policy when the "
            "listing's category is unclear or not covered by a more specific policy."
        ),
    },
]
