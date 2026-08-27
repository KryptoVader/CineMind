"""
CineMind V2 — Deterministic Question Templates
Pure function templates for question text formatting. NO LLM generation.
"""


def format_question_text(category: str, key: str, default_desc: str = "") -> str:
    """Format question text deterministically given category and feature key."""
    if default_desc:
        return default_desc

    category = category.lower()
    key_clean = key.replace("_", " ").title()

    if category == "media_type":
        return f"Is it a {key_clean}?"
    elif category == "genre":
        return f"Does it belong to the {key_clean} genre?"
    elif category == "language":
        return f"Is the original language {key_clean}?"
    elif category == "decade":
        return f"Was it released in the {key_clean}?"
    elif category == "origin_country":
        return f"Is it a {key_clean} production?"
    elif category == "rating":
        return f"Is it rated {key_clean}?"
    elif category == "runtime":
        return f"Is the runtime {key_clean}?"
    elif category == "episodes":
        return f"Does it have {key_clean} episodes?"
    elif category == "actor":
        return f"Does it feature {key_clean}?"
    elif category == "director":
        return f"Was it directed by {key_clean}?"
    elif category == "concept":
        return f"Does it involve {key_clean}?"
    else:
        return f"Is it related to {key_clean}?"
