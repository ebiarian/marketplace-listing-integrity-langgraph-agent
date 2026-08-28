# Synthetic marketplace listings reused from product-image-description-alignment
# (https://github.com/ebiarian/product-image-description-alignment) — the same
# three real product images, already verified as reliable OCR/VQA targets.
# Three mismatches are injected (wrong brand, wrong size, and a wrong product
# entirely) to test the agent's veto behaviour end to end — the last one
# specifically exercises the detect_category/detect_product hard-stop check,
# not the OCR/VQA feature-level checks the other two exercise.
#
# "description" is raw listing text, the same shape a real seller would type
# — not a pre-structured dict. Earlier versions of this file used a
# pre-structured description_features dict, which was a testing convenience
# from before extract_description_features() existed, not something that
# reflects the real interface: a marketplace never hands you clean
# {"brand": "...", "size": "..."} pairs, every real listing is raw text.
LISTINGS = [
    {
        "name": "Prairie Farms Whole Milk — correct",
        "image_url": "https://m.media-amazon.com/images/I/71Arcui4vUL._AC_UF894,1000_QL80_.jpg",
        "description": "Prairie Farms Whole Milk, 1 Quart Carton",
    },
    {
        "name": "Prairie Farms Milk — wrong brand",
        "image_url": "https://m.media-amazon.com/images/I/71Arcui4vUL._AC_UF894,1000_QL80_.jpg",
        "description": "Organic Valley Whole Milk, 1 Quart Carton",
    },
    {
        "name": "Amazon Fresh Chicken Breast — correct",
        "image_url": "https://m.media-amazon.com/images/I/71kUSh8pyiL.jpg",
        "description": "Amazon Fresh Boneless Skinless Chicken Breast, 30 oz Plastic Container",
    },
    {
        "name": "Land O Lakes Salted Butter — wrong size",
        "image_url": "https://m.media-amazon.com/images/I/618LdADpvRL.jpg",
        "description": "Land O Lakes Salted Butter, 16 oz Paper Box",
    },
    {
        "name": "Prairie Farms Milk photo — described as juice (wrong product)",
        "image_url": "https://m.media-amazon.com/images/I/71Arcui4vUL._AC_UF894,1000_QL80_.jpg",
        "description": "Tropicana Orange Juice, 1 Quart Carton",
    },
]
