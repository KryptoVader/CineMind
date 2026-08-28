"""
Feature Registry module for CineMind.

Every canonical feature is formally registered and classified into exactly one category:
- DIRECT_QUESTION: Direct categorical/relational metadata suitable for human questions.
- TRANSFORMED_QUESTION: Numeric/binned attributes requiring threshold transformations.
- NLP_DERIVED: Text fields reserved for future classical NLP processing.
- LEARNING_ONLY: Statistical/popularity features for ML modeling/priors, not exposed to player.
- NEVER_EXPOSE: Strict identity and leakage fields that MUST NEVER be used as gameplay questions.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional


class FeatureCategory(Enum):
    DIRECT_QUESTION = "DIRECT_QUESTION"
    TRANSFORMED_QUESTION = "TRANSFORMED_QUESTION"
    NLP_DERIVED = "NLP_DERIVED"
    LEARNING_ONLY = "LEARNING_ONLY"
    NEVER_EXPOSE = "NEVER_EXPOSE"


@dataclass
class FeatureDefinition:
    name: str
    category: FeatureCategory
    data_type: str
    description: str
    human_label: Optional[str] = None

    @property
    def is_gameplay_eligible(self) -> bool:
        """Returns True if feature can be transformed into or used as a player question."""
        return self.category in (
            FeatureCategory.DIRECT_QUESTION,
            FeatureCategory.TRANSFORMED_QUESTION,
            FeatureCategory.NLP_DERIVED,
        )

    @property
    def is_leakage(self) -> bool:
        """Returns True if feature is an identity/leakage field."""
        return self.category == FeatureCategory.NEVER_EXPOSE


class FeatureRegistry:
    """Central registry enforcing classification and zero-leakage protection for entity features."""

    def __init__(self) -> None:
        self._registry: Dict[str, FeatureDefinition] = {}
        self._register_canonical_features()

    def register(self, feature_def: FeatureDefinition) -> None:
        """Register a feature definition."""
        self._registry[feature_def.name] = feature_def

    def get(self, name: str) -> FeatureDefinition:
        """Retrieve feature definition by name."""
        if name not in self._registry:
            raise KeyError(f"Feature '{name}' is not registered in FeatureRegistry.")
        return self._registry[name]

    def get_category(self, name: str) -> FeatureCategory:
        """Get category of a registered feature."""
        return self.get(name).category

    def is_leakage(self, name: str) -> bool:
        """Check if feature is classified as NEVER_EXPOSE / identity leakage."""
        if name in self._registry:
            return self.get(name).is_leakage
        # Safety default: unknown features treated as leakage
        return True

    def validate_for_gameplay(self, name: str) -> None:
        """Enforce strict non-leakage check. Raises ValueError if feature is NEVER_EXPOSE or LEARNING_ONLY."""
        feature_def = self.get(name)
        if feature_def.is_leakage:
            raise ValueError(
                f"LEAKAGE PROTECTION ERROR: Feature '{name}' is classified as NEVER_EXPOSE and cannot be used in gameplay questions."
            )
        if feature_def.category == FeatureCategory.LEARNING_ONLY:
            raise ValueError(
                f"GAMEPLAY RESTRICTION ERROR: Feature '{name}' is classified as LEARNING_ONLY and cannot be directly exposed to players."
            )

    def list_by_category(self, category: FeatureCategory) -> List[FeatureDefinition]:
        """Return all features belonging to a specific category."""
        return [f for f in self._registry.values() if f.category == category]

    def get_gameplay_features(self) -> List[FeatureDefinition]:
        """Return all features eligible for question generation."""
        return [f for f in self._registry.values() if f.is_gameplay_eligible]

    def _register_canonical_features(self) -> None:
        """Register all 37 canonical entity dataset features into their exact categories."""

        # ------------------------------------------------------------------
        # NEVER_EXPOSE: Identity, title, and pipeline lineage fields
        # ------------------------------------------------------------------
        never_expose_fields = [
            ("cinemind_id", "string", "CineMind unique entity identifier"),
            ("tmdb_id", "float", "TMDB API identifier"),
            ("mal_id", "float", "MyAnimeList API identifier"),
            ("source_id", "string", "Raw source identifier"),
            ("source", "string", "Data source provider (tmdb, mal)"),
            ("title", "string", "Canonical English title"),
            ("original_title", "string", "Original native title"),
            ("alternative_titles", "list[string]", "Alternative title aliases"),
            ("discovered_from", "list[string]", "Internal pipeline discovery tracking"),
            ("source_presence", "list[string]", "Pipeline source presence tracking"),
            ("match_confidence", "string", "Entity cross-matching confidence level"),
        ]
        for name, dtype, desc in never_expose_fields:
            self.register(FeatureDefinition(name=name, category=FeatureCategory.NEVER_EXPOSE, data_type=dtype, description=desc))

        # ------------------------------------------------------------------
        # DIRECT_QUESTION: Directly answerable categorical features
        # ------------------------------------------------------------------
        direct_question_fields = [
            ("media_type", "string", "Primary media format (movie, tv, anime)", "Media Type"),
            ("genres", "list[string]", "Associated genres", "Genre"),
            ("original_language", "string", "Primary audio language code", "Language"),
            ("origin_country", "list[string]", "Country of origin codes", "Country"),
            ("status", "string", "Release / airing status", "Status"),
            ("source_material", "string", "Adaptation source material (manga, original, etc.)", "Source Material"),
            ("themes", "list[string]", "Thematic tags", "Theme"),
            ("demographics", "list[string]", "Target audience demographics (shounen, etc.)", "Demographic"),
            ("studios", "list[string]", "Animation / production studios", "Studio"),
            ("production_companies", "list[string]", "Production companies", "Production Company"),
            ("production_countries", "list[string]", "Countries of production", "Production Country"),
        ]
        for name, dtype, desc, label in direct_question_fields:
            self.register(FeatureDefinition(name=name, category=FeatureCategory.DIRECT_QUESTION, data_type=dtype, description=desc, human_label=label))

        # ------------------------------------------------------------------
        # TRANSFORMED_QUESTION: Numeric/continuous features requiring thresholds
        # ------------------------------------------------------------------
        transformed_question_fields = [
            ("release_year", "float", "Year of release", "Release Year"),
            ("release_date", "string", "Full ISO release date", "Release Date"),
            ("end_date", "string", "Full ISO ending date", "End Date"),
            ("runtime", "float", "Runtime in minutes", "Runtime"),
            ("num_episodes", "float", "Total episode count", "Episode Count"),
            ("rating", "float", "Average user rating / score (0-10)", "Rating Score"),
        ]
        for name, dtype, desc, label in transformed_question_fields:
            self.register(FeatureDefinition(name=name, category=FeatureCategory.TRANSFORMED_QUESTION, data_type=dtype, description=desc, human_label=label))

        # ------------------------------------------------------------------
        # NLP_DERIVED: Unstructured text fields for classical NLP extraction
        # ------------------------------------------------------------------
        nlp_derived_fields = [
            ("overview", "string", "Plot synopsis / summary overview", "Overview"),
            ("keywords", "list[string]", "Associated metadata keywords", "Keywords"),
        ]
        for name, dtype, desc, label in nlp_derived_fields:
            self.register(FeatureDefinition(name=name, category=FeatureCategory.NLP_DERIVED, data_type=dtype, description=desc, human_label=label))

        # ------------------------------------------------------------------
        # LEARNING_ONLY: Internal statistical & popularity metrics
        # ------------------------------------------------------------------
        learning_only_fields = [
            ("vote_count", "float", "Total user vote count"),
            ("popularity", "float", "Source popularity score"),
            ("rank", "float", "Global popularity or quality rank"),
            ("favorites", "float", "Number of user favorite saves"),
            ("num_list_users", "float", "Number of user list inclusions"),
            ("source_media_type", "string", "Source-specific media format"),
            ("relations", "list[dict]", "Cross-entity franchise relations"),
        ]
        for name, dtype, desc in learning_only_fields:
            self.register(FeatureDefinition(name=name, category=FeatureCategory.LEARNING_ONLY, data_type=dtype, description=desc))


# Default global instance for convenience
DEFAULT_FEATURE_REGISTRY = FeatureRegistry()
