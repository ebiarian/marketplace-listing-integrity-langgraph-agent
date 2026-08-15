# Synthetic marketplace listings reused from product-image-description-alignment
# (https://github.com/ebiarian/product-image-description-alignment) — the same
# three real product images, already verified as reliable OCR/VQA targets.
# Three mismatches are injected (wrong brand, wrong size, and a wrong product
# entirely) to test the agent's veto behaviour end to end — the last one
# specifically exercises the detect_category/detect_product hard-stop check,
# not the OCR/VQA feature-level checks the other two exercise.
LISTINGS = [
    {
        "name": "Prairie Farms Whole Milk — correct",
        "image_url": "https://m.media-amazon.com/images/I/71Arcui4vUL._AC_UF894,1000_QL80_.jpg",
        "description_features": {
            "product_type": "milk",
            "brand": "prairie farms",
            "size": "1 quart",
            "variant": "whole",
            "container": "carton",
        },
    },
    {
        "name": "Prairie Farms Milk — wrong brand",
        "image_url": "https://m.media-amazon.com/images/I/71Arcui4vUL._AC_UF894,1000_QL80_.jpg",
        "description_features": {
            "product_type": "milk",
            "brand": "organic valley",
            "size": "1 quart",
            "variant": "whole",
            "container": "carton",
        },
    },
    {
        "name": "Amazon Fresh Chicken Breast — correct",
        "image_url": "https://m.media-amazon.com/images/I/71kUSh8pyiL.jpg",
        "description_features": {
            "product_type": "chicken breast",
            "brand": "amazon fresh",
            "size": "30 oz",
            "variant": "boneless skinless",
            "container": "plastic container",
        },
    },
    {
        "name": "Land O Lakes Salted Butter — wrong size",
        "image_url": "https://m.media-amazon.com/images/I/618LdADpvRL.jpg",
        "description_features": {
            "product_type": "butter",
            "brand": "land o lakes",
            "size": "16 oz",
            "variant": "salted",
            "container": "paper box",
        },
    },
    {
        "name": "Prairie Farms Milk photo — described as juice (wrong product)",
        "image_url": "https://m.media-amazon.com/images/I/71Arcui4vUL._AC_UF894,1000_QL80_.jpg",
        "description_features": {
            "product_type": "juice",
            "brand": "tropicana",
            "size": "1 quart",
            "variant": "orange",
            "container": "carton",
        },
    },
]
