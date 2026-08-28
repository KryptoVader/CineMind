"""
Question Generators for CineMind.

Generates candidate structured questions across Media, Genre, Language, Country, Time,
Runtime, Episode Count, and Rating families.
"""

from abc import ABC, abstractmethod
from collections import Counter
from typing import Any, Dict, List, Optional, Set
import numpy as np

from cinemind.data.entity import Entity
from cinemind.data.feature_registry import FeatureRegistry, DEFAULT_FEATURE_REGISTRY
from cinemind.questions.schema import Operator, Question


# Mapping codes to clear human readable names
LANGUAGE_NAMES: Dict[str, str] = {
    "ja": "Japanese",
    "en": "English",
    "fr": "French",
    "ko": "Korean",
    "zh": "Chinese",
    "hi": "Hindi",
    "es": "Spanish",
    "de": "German",
    "it": "Italian",
    "ru": "Russian",
    "pt": "Portuguese",
    "th": "Thai",
    "tl": "Tagalog",
    "id": "Indonesian",
    "ta": "Tamil",
    "te": "Telugu",
    "ml": "Malayalam",
    "bn": "Bengali",
}

COUNTRY_NAMES: Dict[str, str] = {
    "JP": "Japan",
    "US": "the United States",
    "GB": "the United Kingdom",
    "FR": "France",
    "KR": "South Korea",
    "CN": "China",
    "IN": "India",
    "DE": "Germany",
    "CA": "Canada",
    "IT": "Italy",
    "ES": "Spain",
    "AU": "Australia",
    "RU": "Russia",
    "HK": "Hong Kong",
}


class BaseQuestionGenerator(ABC):
    """Abstract base class for all question generators."""

    def __init__(self, registry: Optional[FeatureRegistry] = None) -> None:
        self.registry = registry or DEFAULT_FEATURE_REGISTRY

    @abstractmethod
    def generate(self, entities: List[Entity]) -> List[Question]:
        """Generate a list of candidate questions based on the entity population."""
        pass


class MediaGenerator(BaseQuestionGenerator):
    """Generates media format questions (Movie, TV Series, Anime)."""

    def generate(self, entities: List[Entity]) -> List[Question]:
        questions: List[Question] = []
        media_types = set()
        for e in entities:
            m = e.get_feature("media_type")
            if m:
                media_types.add(str(m).lower())

        if "movie" in media_types:
            questions.append(
                Question.create(
                    text="Is it a movie?",
                    feature="media_type",
                    operator=Operator.EQUALS,
                    value="movie",
                    question_family="media",
                    reliability=0.98,
                    answerability=1.0,
                )
            )
        if "tv" in media_types:
            questions.append(
                Question.create(
                    text="Is it a TV series?",
                    feature="media_type",
                    operator=Operator.EQUALS,
                    value="tv",
                    question_family="media",
                    reliability=0.98,
                    answerability=1.0,
                )
            )
        if "anime" in media_types:
            questions.append(
                Question.create(
                    text="Is it an anime?",
                    feature="media_type",
                    operator=Operator.EQUALS,
                    value="anime",
                    question_family="media",
                    reliability=0.98,
                    answerability=1.0,
                )
            )

        return questions


class GenreGenerator(BaseQuestionGenerator):
    """Generates categorical genre questions based on actual dataset frequencies."""

    def __init__(
        self,
        min_occurrences: int = 10,
        registry: Optional[FeatureRegistry] = None,
    ) -> None:
        super().__init__(registry=registry)
        self.min_occurrences = min_occurrences

    def generate(self, entities: List[Entity]) -> List[Question]:
        genre_counts: Counter = Counter()
        for e in entities:
            genres = e.get_feature("genres")
            if isinstance(genres, list):
                for g in genres:
                    if isinstance(g, str) and g.strip():
                        genre_counts[g.strip()] += 1

        questions: List[Question] = []
        for genre, count in genre_counts.items():
            if count >= self.min_occurrences:
                # Humanize genre text
                text_label = genre.lower()
                if not text_label.startswith("an ") and not text_label.startswith("a "):
                    article = "an" if text_label[0] in "aeiou" else "a"
                    text = f"Is it {article} {genre} title?"
                else:
                    text = f"Is it {genre}?"

                questions.append(
                    Question.create(
                        text=text,
                        feature="genres",
                        operator=Operator.CONTAINS,
                        value=genre,
                        question_family="genre",
                        reliability=0.90,
                        answerability=0.95,
                    )
                )

        return questions


class LanguageGenerator(BaseQuestionGenerator):
    """Generates primary audio language questions."""

    def __init__(
        self,
        top_n: int = 12,
        registry: Optional[FeatureRegistry] = None,
    ) -> None:
        super().__init__(registry=registry)
        self.top_n = top_n

    def generate(self, entities: List[Entity]) -> List[Question]:
        lang_counts: Counter = Counter()
        for e in entities:
            lang = e.get_feature("original_language")
            if lang and isinstance(lang, str):
                lang_counts[lang.strip().lower()] += 1

        questions: List[Question] = []
        for code, count in lang_counts.most_common(self.top_n):
            if count < 5:
                continue
            lang_name = LANGUAGE_NAMES.get(code, code.upper())
            questions.append(
                Question.create(
                    text=f"Is the primary language {lang_name}?",
                    feature="original_language",
                    operator=Operator.EQUALS,
                    value=code,
                    question_family="language",
                    reliability=0.95,
                    answerability=0.90,
                )
            )

        return questions


class CountryGenerator(BaseQuestionGenerator):
    """Generates origin country questions."""

    def __init__(
        self,
        top_n: int = 10,
        registry: Optional[FeatureRegistry] = None,
    ) -> None:
        super().__init__(registry=registry)
        self.top_n = top_n

    def generate(self, entities: List[Entity]) -> List[Question]:
        country_counts: Counter = Counter()
        for e in entities:
            countries = e.get_feature("origin_country")
            if isinstance(countries, list):
                for c in countries:
                    if isinstance(c, str) and c.strip():
                        country_counts[c.strip().upper()] += 1

        questions: List[Question] = []
        for code, count in country_counts.most_common(self.top_n):
            if count < 5:
                continue
            country_name = COUNTRY_NAMES.get(code, code)
            text = f"Is it produced in or associated with {country_name}?"
            questions.append(
                Question.create(
                    text=text,
                    feature="origin_country",
                    operator=Operator.CONTAINS,
                    value=code,
                    question_family="country",
                    reliability=0.92,
                    answerability=0.85,
                )
            )

        return questions


class TimeGenerator(BaseQuestionGenerator):
    """Generates release time threshold and decade questions."""

    def generate(self, entities: List[Entity]) -> List[Question]:
        years: List[float] = []
        for e in entities:
            yr = e.get_feature("release_year")
            if yr and not np.isnan(yr) and 1900 <= float(yr) <= 2030:
                years.append(float(yr))

        if not years:
            return []

        questions: List[Question] = []

        # Decade questions
        decades = [1970, 1980, 1990, 2000, 2010, 2020]
        for d in decades:
            questions.append(
                Question.create(
                    text=f"Was it released in the {d}s ({d}–{d+9})?",
                    feature="release_year",
                    operator=Operator.BETWEEN,
                    value=[d, d + 9],
                    question_family="time",
                    reliability=0.95,
                    answerability=0.85,
                )
            )

        # Useful threshold questions
        thresholds = [1980, 1990, 2000, 2010, 2018, 2020]
        for t in thresholds:
            questions.append(
                Question.create(
                    text=f"Was it released after {t}?",
                    feature="release_year",
                    operator=Operator.GREATER_THAN,
                    value=t,
                    question_family="time",
                    reliability=0.95,
                    answerability=0.90,
                )
            )
            questions.append(
                Question.create(
                    text=f"Was it released before {t}?",
                    feature="release_year",
                    operator=Operator.LESS_THAN,
                    value=t,
                    question_family="time",
                    reliability=0.95,
                    answerability=0.90,
                )
            )

        return questions


class RuntimeGenerator(BaseQuestionGenerator):
    """Generates duration/runtime threshold questions."""

    def generate(self, entities: List[Entity]) -> List[Question]:
        runtimes = [
            float(e.get_feature("runtime"))
            for e in entities
            if e.get_feature("runtime") is not None and not np.isnan(float(e.get_feature("runtime")))
        ]

        if not runtimes:
            return []

        questions: List[Question] = []

        # Useful thresholds: <60m, >=90m, >=120m (2 hours)
        questions.append(
            Question.create(
                text="Is the runtime under 60 minutes?",
                feature="runtime",
                operator=Operator.LESS_THAN,
                value=60.0,
                question_family="runtime",
                reliability=0.90,
                answerability=0.80,
            )
        )
        questions.append(
            Question.create(
                text="Is the runtime 90 minutes (1.5 hours) or longer?",
                feature="runtime",
                operator=Operator.GREATER_EQUAL,
                value=90.0,
                question_family="runtime",
                reliability=0.90,
                answerability=0.85,
            )
        )
        questions.append(
            Question.create(
                text="Is the runtime 2 hours (120 minutes) or longer?",
                feature="runtime",
                operator=Operator.GREATER_EQUAL,
                value=120.0,
                question_family="runtime",
                reliability=0.90,
                answerability=0.85,
            )
        )

        return questions


class EpisodeCountGenerator(BaseQuestionGenerator):
    """Generates episode count threshold questions for TV series and anime."""

    def generate(self, entities: List[Entity]) -> List[Question]:
        episodes = [
            float(e.get_feature("num_episodes"))
            for e in entities
            if e.get_feature("num_episodes") is not None and not np.isnan(float(e.get_feature("num_episodes")))
        ]

        if not episodes:
            return []

        questions: List[Question] = []

        questions.append(
            Question.create(
                text="Is it a single episode or feature special?",
                feature="num_episodes",
                operator=Operator.EQUALS,
                value=1.0,
                question_family="episodes",
                reliability=0.92,
                answerability=0.90,
            )
        )
        questions.append(
            Question.create(
                text="Does it have more than 12 episodes (more than 1 cour)?",
                feature="num_episodes",
                operator=Operator.GREATER_THAN,
                value=12.0,
                question_family="episodes",
                reliability=0.92,
                answerability=0.85,
            )
        )
        questions.append(
            Question.create(
                text="Does it have more than 24 episodes (2+ seasons or 2+ cours)?",
                feature="num_episodes",
                operator=Operator.GREATER_THAN,
                value=24.0,
                question_family="episodes",
                reliability=0.92,
                answerability=0.85,
            )
        )
        questions.append(
            Question.create(
                text="Does it have more than 50 episodes?",
                feature="num_episodes",
                operator=Operator.GREATER_THAN,
                value=50.0,
                question_family="episodes",
                reliability=0.92,
                answerability=0.85,
            )
        )

        return questions


class RatingGenerator(BaseQuestionGenerator):
    """Generates user rating threshold questions (>= 7.0, >= 8.0, >= 8.5)."""

    def generate(self, entities: List[Entity]) -> List[Question]:
        ratings = [
            float(e.get_feature("rating"))
            for e in entities
            if e.get_feature("rating") is not None and not np.isnan(float(e.get_feature("rating")))
        ]

        if not ratings:
            return []

        questions: List[Question] = []

        questions.append(
            Question.create(
                text="Is it highly rated (rating 7.0 out of 10 or higher)?",
                feature="rating",
                operator=Operator.GREATER_EQUAL,
                value=7.0,
                question_family="rating",
                reliability=0.85,
                answerability=0.75,
            )
        )
        questions.append(
            Question.create(
                text="Is it critically acclaimed (rating 8.0 out of 10 or higher)?",
                feature="rating",
                operator=Operator.GREATER_EQUAL,
                value=8.0,
                question_family="rating",
                reliability=0.85,
                answerability=0.75,
            )
        )

        return questions


class CompositeQuestionGenerator(BaseQuestionGenerator):
    """Aggregates all structured question generators to generate full candidate pool."""

    def __init__(self, registry: Optional[FeatureRegistry] = None) -> None:
        super().__init__(registry=registry)
        self.generators: List[BaseQuestionGenerator] = [
            MediaGenerator(registry=self.registry),
            GenreGenerator(registry=self.registry),
            LanguageGenerator(registry=self.registry),
            CountryGenerator(registry=self.registry),
            TimeGenerator(registry=self.registry),
            RuntimeGenerator(registry=self.registry),
            EpisodeCountGenerator(registry=self.registry),
            RatingGenerator(registry=self.registry),
        ]

    def generate(self, entities: List[Entity]) -> List[Question]:
        all_questions: List[Question] = []
        for gen in self.generators:
            all_questions.extend(gen.generate(entities))
        return all_questions
