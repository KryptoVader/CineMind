"""
Canonical Genre Taxonomy Unification Module.

Provides GENRE_CANONICAL_MAP to split compound genre tags, merge cross-source synonyms,
and produce clean, unified genre arrays across the canonical entity universe.
"""

from typing import Iterable

# Declarative canonical mapping
GENRE_CANONICAL_MAP: dict[str, list[str]] = {
    # --- Compound Tag Splits ---
    "Action & Adventure": ["Action", "Adventure"],
    "Sci-Fi & Fantasy": ["Science Fiction", "Fantasy"],
    "War & Politics": ["War", "Politics"],
    
    # --- Synonym Merges ---
    "Sci-Fi": ["Science Fiction"],
    "Science Fiction": ["Science Fiction"],
    "Historical": ["History"],
    "History": ["History"],
    "TV Movie": ["TV Movie"],
    
    # --- Standard Pass-throughs ---
    "Action": ["Action"],
    "Adventure": ["Adventure"],
    "Animation": ["Animation"],
    "Comedy": ["Comedy"],
    "Crime": ["Crime"],
    "Documentary": ["Documentary"],
    "Drama": ["Drama"],
    "Family": ["Family"],
    "Fantasy": ["Fantasy"],
    "Horror": ["Horror"],
    "Music": ["Music"],
    "Mystery": ["Mystery"],
    "Romance": ["Romance"],
    "Thriller": ["Thriller"],
    "War": ["War"],
    "Western": ["Western"],
    "Kids": ["Kids"],
    "Reality": ["Reality"],
    "Talk": ["Talk"],
    "Soap": ["Soap"],
    "News": ["News"],
    
    # --- Anime-Specific / Specialized Genres ---
    "Shounen": ["Shounen"],
    "Seinen": ["Seinen"],
    "Shoujo": ["Shoujo"],
    "Josei": ["Josei"],
    "School": ["School"],
    "Supernatural": ["Supernatural"],
    "Slice of Life": ["Slice of Life"],
    "Mecha": ["Mecha"],
    "Avant Garde": ["Avant Garde"],
    "Anthropomorphic": ["Anthropomorphic"],
    "Parody": ["Parody"],
    "Sports": ["Sports"],
    "Ecchi": ["Ecchi"],
}


def unify_genres(raw_genres: Iterable[str] | None) -> list[str]:
    """
    Map raw genre names through the canonical taxonomy table,
    splitting compound tags, merging synonyms, and deduplicating order-dependently.
    """
    if not raw_genres:
        return []

    unified: list[str] = []
    seen: set[str] = set()

    for g in raw_genres:
        if not g or not isinstance(g, str):
            continue
        g_clean = g.strip()
        if not g_clean:
            continue

        # Look up in canonical map, or keep cleaned string if unmapped
        mapped_list = GENRE_CANONICAL_MAP.get(g_clean, [g_clean])
        for mapped in mapped_list:
            if mapped not in seen:
                seen.add(mapped)
                unified.append(mapped)

    return unified
