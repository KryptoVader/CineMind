"""
CineMind Knowledge Layer (Phase 1).

Builds entity representations across 100,000 entities:
1. Structured Metadata & Supervised Genre Classifiers (Logistic Regression)
2. Sparse TF-IDF Matrix over plot overviews
3. Dense 100-dim SVD/LSA Entity Vectors & Semantic Nearest Neighbors
4. 200 Concept Clusters with NLTK POS + NER Proper-Noun Filtering
"""

from pathlib import Path
import re
import logging
from typing import Optional, Any
import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from pipeline.config import DATA_DIR, CANONICAL_DIR

ANALYTICS_DIR = DATA_DIR / "analytics"
MODEL_DIR = DATA_DIR / "models"

logger = logging.getLogger(__name__)


class MetadataFeature:
    """A metadata or semantic feature definition."""
    def __init__(self, feature_id: str, question_text: str, category: str, eval_fn: Any):
        self.feature_id = feature_id
        self.question_text = question_text
        self.category = category
        self.eval_fn = eval_fn


GENRES_TO_LEARN = [
    ("Drama", "m_genre_drama", ["drama"]),
    ("Comedy", "m_genre_comedy", ["comedy"]),
    ("Action", "m_genre_action", ["action"]),
    ("Romance", "m_genre_romance", ["romance", "romantic"]),
    ("Crime", "m_genre_crime", ["crime"]),
    ("Thriller", "m_genre_thriller", ["thriller", "suspense"]),
    ("Adventure", "m_genre_adventure", ["adventure"]),
    ("Fantasy", "m_genre_fantasy", ["fantasy"]),
    ("Mystery", "m_genre_mystery", ["mystery"]),
    ("Horror", "m_genre_horror", ["horror"]),
    ("Family", "m_genre_family", ["family", "kids", "children"]),
    ("Science Fiction", "m_genre_science_fiction", ["science fiction", "sci-fi", "scifi"]),
    ("War", "m_genre_war", ["war", "military"]),
    ("Supernatural", "m_genre_supernatural", ["supernatural"]),
    ("Mecha", "m_genre_mecha", ["mecha"]),
    ("School", "m_genre_school", ["school"]),
    ("Slice of Life", "m_genre_slice_of_life", ["slice of life"]),
    ("Sports", "m_genre_sports", ["sports", "sport"]),
    ("Shounen", "m_genre_shounen", ["shounen"]),
    ("Seinen", "m_genre_seinen", ["seinen"]),
    ("Documentary", "m_genre_documentary", ["documentary"]),
    ("Western", "m_genre_western", ["western"]),
    ("History", "m_genre_history", ["historical", "history"]),
    ("Music", "m_genre_music", ["music"]),
    ("Animation", "m_genre_animation", ["animation", "animated", "anime"]),
]


def train_or_load_genre_classifiers(df: pd.DataFrame, tfidf_sparse: Any) -> dict[str, Any]:
    """
    Requirement 1: Train One-vs-Rest Logistic Regression per genre on TMDB ground-truth labels.
    Cache trained classifiers to disk (DATA_DIR / 'models' / 'genre_classifiers.joblib').
    Report Precision/Recall on held-out test split.
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = MODEL_DIR / "genre_classifiers.joblib"

    if cache_path.exists():
        print("  [KnowledgeBase] Loading cached supervised genre classifiers from disk...")
        try:
            classifiers = joblib.load(cache_path)
            return classifiers
        except Exception as e:
            logger.warning(f"Failed to load cached classifiers: {e}. Retraining...")

    print("  [KnowledgeBase] Training One-vs-Rest Supervised Genre Classifiers (80/20 train/test split)...")
    classifiers = {}
    reports = []

    for g_name, feat_id, tags in GENRES_TO_LEARN:
        # Ground-truth binary label from TMDB genres tag list
        y_true = df["genres"].apply(
            lambda x: any(any(gt in str(i).lower() for gt in tags) for i in x)
            if isinstance(x, (list, np.ndarray, tuple)) else False
        ).values.astype(int)

        if np.sum(y_true) < 10:
            continue

        # Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            tfidf_sparse, y_true, test_size=0.20, random_state=42, stratify=y_true
        )

        clf = LogisticRegression(C=2.0, max_iter=200, class_weight="balanced", random_state=42)
        clf.fit(X_train, y_train)
        classifiers[g_name] = clf

        # Evaluate on held-out split
        y_pred = clf.predict(X_test)
        pos_indices = np.where(y_test == 1)[0]
        neg_indices = np.where(y_test == 0)[0]
        tp = np.sum(y_pred[pos_indices] == 1)
        fp = np.sum(y_pred[neg_indices] == 1)
        fn = np.sum(y_pred[pos_indices] == 0)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        reports.append((g_name, precision, recall, np.sum(y_true)))

    print("  ================================================================")
    print("  LEARNED GENRE CLASSIFIER EVALUATION (20% HELD-OUT TEST SPLIT)")
    print("  ================================================================")
    print(f"  {'Genre':<20s} | {'Precision':<10s} | {'Recall':<10s} | {'Total Positives':<15s}")
    print("  ----------------------------------------------------------------")
    for g_name, prec, rec, tot in reports:
        print(f"  {g_name:<20s} | {prec*100:6.1f}%    | {rec*100:6.1f}%    | {tot:<15,d}")
    print("  ================================================================")

    joblib.dump(classifiers, cache_path)
    print(f"  [KnowledgeBase] Cached {len(classifiers)} trained classifiers to {cache_path}")
    return classifiers


def build_metadata_definitions(df: pd.DataFrame, tfidf_sparse: Any) -> list[MetadataFeature]:
    """Define structured metadata and supervised genre features."""
    defs: list[MetadataFeature] = []

    # 1. Media Types
    defs.append(MetadataFeature("m_is_movie", "Is it a movie (not a TV series)?", "media_type",
        lambda d: d["media_type"].str.contains("movie", case=False).fillna(False).values))
    defs.append(MetadataFeature("m_is_tv", "Is it a TV series?", "media_type",
        lambda d: d["media_type"].str.contains("tv", case=False).fillna(False).values))
    defs.append(MetadataFeature("m_is_anime", "Is it an anime (Japanese animation)?", "media_type",
        lambda d: d["media_type"].str.contains("anime|ova|ona", case=False).fillna(False).values))

    # 2. Decades
    defs.append(MetadataFeature("m_2020s", "Was it released in the 2020s (2020 or later)?", "decade",
        lambda d: (d["release_year"] >= 2020).fillna(False).values))
    defs.append(MetadataFeature("m_2010s", "Was it released in the 2010s (2010–2019)?", "decade",
        lambda d: ((d["release_year"] >= 2010) & (d["release_year"] <= 2019)).fillna(False).values))
    defs.append(MetadataFeature("m_2000s", "Was it released in the 2000s (2000–2009)?", "decade",
        lambda d: ((d["release_year"] >= 2000) & (d["release_year"] <= 2009)).fillna(False).values))
    defs.append(MetadataFeature("m_1990s", "Was it released in the 1990s (1990–1999)?", "decade",
        lambda d: ((d["release_year"] >= 1990) & (d["release_year"] <= 1999)).fillna(False).values))
    defs.append(MetadataFeature("m_pre1990", "Was it released before 1990?", "decade",
        lambda d: (d["release_year"] < 1990).fillna(False).values))

    # 3. Languages
    for code, name in [("ja", "Japanese"), ("en", "English"), ("ko", "Korean"),
                       ("zh", "Chinese"), ("fr", "French"), ("es", "Spanish"),
                       ("de", "German"), ("hi", "Hindi"), ("it", "Italian")]:
        defs.append(MetadataFeature(f"m_lang_{code}", f"Is the original language {name}?", "language",
            (lambda c: lambda d: (d["original_language"] == c).values)(code)))

    # 4. Origin Countries
    for code, name in [("US", "United States"), ("JP", "Japan"), ("KR", "South Korea"),
                       ("IN", "India"), ("GB", "United Kingdom"), ("FR", "France"), ("CN", "China")]:
        defs.append(MetadataFeature(f"m_country_{code.lower()}", f"Is it a {name} production?", "origin_country",
            (lambda c: lambda d: d["origin_country"].apply(
                lambda x: c in x if isinstance(x, (list, np.ndarray, tuple)) else False
            ).values)(code)))

    # 5. Rating Tiers
    defs.append(MetadataFeature("m_rating_masterpiece", "Is it critically acclaimed (rated 8.5 or higher)?", "rating",
        lambda d: (d["rating"] >= 8.5).fillna(False).values))
    defs.append(MetadataFeature("m_rating_good", "Is it well-regarded (rated 7.5 or higher)?", "rating",
        lambda d: (d["rating"] >= 7.5).fillna(False).values))
    defs.append(MetadataFeature("m_rating_poor", "Is it considered low-rated or poor (rated below 5.0)?", "rating",
        lambda d: ((d["rating"].notna()) & (d["rating"] < 5.0)).values))

    # 6. Source Material
    for mat_id, mat_name in [("manga", "manga"), ("original", "original story"), ("light_novel", "light novel"), ("game", "video game")]:
        defs.append(MetadataFeature(f"m_source_{mat_id}", f"Is it based on an {mat_name}?", "source_material",
            (lambda m: lambda d: (d["source_material"].str.lower() == m).fillna(False).values)(mat_id)))

    # 7. Episode Count Buckets
    defs.append(MetadataFeature("m_ep_movie", "Is it a standalone feature / single episode?", "episodes",
        lambda d: (d["num_episodes"] == 1).fillna(False).values))
    defs.append(MetadataFeature("m_ep_long", "Is it a long-running series (100+ episodes)?", "episodes",
        lambda d: (d["num_episodes"] >= 100).fillna(False).values))

    # 8. Requirement 1: Supervised Learned Genre Classifiers
    classifiers = train_or_load_genre_classifiers(df, tfidf_sparse)

    for g_name, feat_id, tags in GENRES_TO_LEARN:
        if g_name in classifiers:
            clf = classifiers[g_name]
            # Predict match vector using trained classifier
            preds = clf.predict(tfidf_sparse).astype(bool)
            defs.append(MetadataFeature(
                feat_id,
                f"Does it belong to or involve {g_name}?",
                "genre",
                (lambda p: lambda d: p)(preds)
            ))

    return defs


class KnowledgeBase:
    """Loaded knowledge base holding metadata vectors, TF-IDF matrix, dense LSA vectors, and concept clusters."""

    NOISE_WORDS = {
        "man", "woman", "life", "world", "story", "young", "new", "years",
        "finds", "day", "way", "based", "takes", "tries", "help", "later",
        "soon", "featuring", "episodes", "recap", "begins", "wants", "comes",
        "lives", "home", "time", "makes", "gets", "goes", "back", "must"
    }

    def __init__(self, df: pd.DataFrame, num_concepts: int = 200):
        self.df = df
        self.num_entities = len(df)
        self.num_concepts = num_concepts

        # Build TF-IDF
        print("  [KnowledgeBase] Fitting TF-IDF vectorizer over overviews...")
        overviews = df["overview"].fillna("").values
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            min_df=5,
            max_df=0.15,
            stop_words="english",
            ngram_range=(1, 2),
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
            sublinear_tf=True,
        )
        self.tfidf_sparse = self.vectorizer.fit_transform(overviews)  # shape: (100K, N_vocab)
        self.feature_names = self.vectorizer.get_feature_names_out()
        self.word_to_idx = {w: idx for idx, w in enumerate(self.feature_names)}

        print(f"  [KnowledgeBase] TF-IDF sparse matrix: {self.tfidf_sparse.shape}")

        # Requirement 3: Project entity TF-IDF rows into 100-dim SVD component space
        print("  [KnowledgeBase] Computing 100-dim Entity LSA Dense Embeddings...")
        self.svd = TruncatedSVD(n_components=100, random_state=42)
        self.entity_lsa_vectors = self.svd.fit_transform(self.tfidf_sparse)  # shape: (100K, 100)

        # Pre-normalize entity LSA vectors for fast cosine similarity matrix multiplication
        lsa_norms = np.linalg.norm(self.entity_lsa_vectors, axis=1, keepdims=True)
        lsa_norms = np.where(lsa_norms > 1e-6, lsa_norms, 1.0)
        self.entity_lsa_normalized = self.entity_lsa_vectors / lsa_norms

        # Load metadata features and trained genre classifiers
        self.meta_defs = build_metadata_definitions(df, self.tfidf_sparse)
        self.meta_matrix = np.zeros((len(self.meta_defs), self.num_entities), dtype=bool)

        for i, m_def in enumerate(self.meta_defs):
            self.meta_matrix[i] = m_def.eval_fn(df)

        print(f"  [KnowledgeBase] Metadata matrix: {self.meta_matrix.shape}")

        # Build Concept Clusters using MiniBatchKMeans over TruncatedSVD word representations
        print(f"  [KnowledgeBase] Fitting MiniBatchKMeans ({num_concepts} concept clusters)...")
        word_vectors = self.svd.components_.T  # shape: (5000, 100)

        kmeans = MiniBatchKMeans(n_clusters=num_concepts, random_state=42, batch_size=1000)
        word_labels = kmeans.fit_predict(word_vectors)

        # Import local NLTK tools for classical POS & NER proper-noun filtering
        import nltk
        from nltk.corpus import wordnet as wn
        from nltk.stem import WordNetLemmatizer
        from nltk import pos_tag, ne_chunk, tree2conlltags

        lemmatizer = WordNetLemmatizer()

        # Store clusters as list of word index arrays, clean display words, templates, and low_quality flags
        self.concept_clusters: list[np.ndarray] = []
        self.concept_clean_words: list[list[str]] = []
        self.concept_questions: list[str] = []
        self.concept_is_low_quality: list[bool] = []

        # Pre-computed concept matrix: shape (num_concepts, num_entities)
        self.concept_sums_matrix = np.zeros((num_concepts, self.num_entities), dtype=np.float32)

        for c in range(num_concepts):
            w_indices = np.where(word_labels == c)[0]
            self.concept_clusters.append(w_indices)

            if len(w_indices) == 0:
                self.concept_clean_words.append([])
                self.concept_questions.append("")
                self.concept_is_low_quality.append(True)
                continue

            # Calculate cluster centroid in SVD space
            c_vecs = word_vectors[w_indices]  # shape: (|cluster|, 100)
            centroid = c_vecs.mean(axis=0)    # shape: (100,)
            c_norm = np.linalg.norm(centroid)

            # Compute cosine similarity of each word to cluster centroid
            if c_norm > 1e-6:
                word_norms = np.linalg.norm(c_vecs, axis=1)
                word_norms = np.where(word_norms > 1e-6, word_norms, 1.0)
                cos_sims = (c_vecs @ centroid) / (word_norms * c_norm)
            else:
                cos_sims = np.zeros(len(w_indices))

            # Rank word indices by cosine similarity to centroid (descending)
            ranked_order = np.argsort(cos_sims)[::-1]
            ranked_w_indices = w_indices[ranked_order]

            # Filter words using POS tagging & NER proper-noun filtering
            words_to_tag = [self.feature_names[idx] for idx in ranked_w_indices if self.feature_names[idx] not in self.NOISE_WORDS]
            pos_results = dict(pos_tag(words_to_tag)) if words_to_tag else {}

            # Requirement 2: NLTK Named Entity Recognition proper-noun detection
            ner_results: dict[str, str] = {}
            if words_to_tag:
                try:
                    tagged_pos = pos_tag(words_to_tag)
                    ne_tree = ne_chunk(tagged_pos)
                    conll = tree2conlltags(ne_tree)
                    ner_results = {w: iob for w, ptag, iob in conll}
                except Exception:
                    ner_results = {}

            seen_lemmas: set[str] = set()
            valid_words: list[str] = []
            word_types: list[str] = []
            proper_noun_count = 0

            for w_idx in ranked_w_indices:
                word = self.feature_names[w_idx]
                if word in self.NOISE_WORDS or len(word) <= 2:
                    continue

                # Requirement 2: Detect & exclude Proper Nouns (NNP / NNPS / NER tags PERSON / GPE / ORG)
                ptag = pos_results.get(word, "")
                ner_tag = ner_results.get(word, "O")

                is_proper_noun = (
                    ptag in ["NNP", "NNPS"] or
                    ner_tag != "O" or
                    word[0].isupper()
                )

                if is_proper_noun:
                    proper_noun_count += 1
                    continue

                # Get noun lemma
                lemma = lemmatizer.lemmatize(word, pos="n")
                if lemma == word:
                    lemma = lemmatizer.lemmatize(word, pos="v")

                if lemma in seen_lemmas:
                    continue

                # POS tagging check: prefer nouns/noun-phrases over verb stems or fragments
                synsets = wn.synsets(word)
                noun_synsets = [s for s in synsets if s.pos() == "n"]

                # Accept if POS tag is noun (NN/NNS) OR has WordNet noun synsets
                is_noun = ptag.startswith("NN") or len(noun_synsets) > 0
                if not is_noun:
                    continue

                seen_lemmas.add(lemma)
                valid_words.append(word)

                # Classify type for template selection
                lexnames = [s.lexname() for s in noun_synsets[:3]] if noun_synsets else []
                if any("person" in lex or "group" in lex or "relative" in lex for lex in lexnames):
                    word_types.append("person")
                elif any("location" in lex or "artifact" in lex or "state" in lex for lex in lexnames):
                    word_types.append("place")
                else:
                    word_types.append("concept")

                if len(valid_words) >= 3:
                    break

            # Requirement 2: Mark low_quality if cluster is majority proper-nouns or has < 2 valid common nouns
            is_majority_proper = proper_noun_count > len(valid_words)
            if len(valid_words) < 2 or is_majority_proper:
                self.concept_clean_words.append(valid_words)
                self.concept_questions.append("")
                self.concept_is_low_quality.append(True)
            else:
                self.concept_clean_words.append(valid_words)
                self.concept_is_low_quality.append(False)

                # Select question template from template bank
                words_str = ", ".join(valid_words)
                person_cnt = sum(1 for t in word_types if t == "person")
                place_cnt = sum(1 for t in word_types if t == "place")

                if person_cnt >= 1:
                    q_text = f"Does the story involve a character or role like {words_str}?"
                elif place_cnt >= 1:
                    q_text = f"Does the story take place in or involve a {words_str}?"
                else:
                    q_text = f"Does the story involve concepts like {words_str}?"

                self.concept_questions.append(q_text)

            # Compute TF-IDF sums for entity matrix
            cluster_sparse = self.tfidf_sparse[:, w_indices]
            self.concept_sums_matrix[c] = np.array(cluster_sparse.sum(axis=1)).ravel()

        print(f"  [KnowledgeBase] Concept sums matrix shape: {self.concept_sums_matrix.shape}")
        valid_cnt = sum(1 for lq in self.concept_is_low_quality if not lq)
        print(f"  [KnowledgeBase] Quality concept clusters: {valid_cnt}/{num_concepts} (Low quality skipped: {num_concepts - valid_cnt})")

    def get_lsa_similarity(self, idx_a: int, idx_b: int) -> float:
        """Requirement 3: Compute LSA cosine similarity between two entity indices."""
        return float(self.entity_lsa_normalized[idx_a] @ self.entity_lsa_normalized[idx_b])

    def get_lsa_neighbors(self, idx: int, k: int = 5) -> list[int]:
        """Requirement 3: Find top-K nearest neighbor entity indices in dense LSA space."""
        sims = self.entity_lsa_normalized @ self.entity_lsa_normalized[idx]
        top_k = np.argpartition(sims, -(k + 1))[-(k + 1):]
        top_k = top_k[np.argsort(sims[top_k])[::-1]]
        return [int(i) for i in top_k if i != idx][:k]

    def get_concept_top_words(self, concept_idx: int, top_n: int = 3) -> list[str]:
        """Get top clean display words for a concept cluster."""
        words = self.concept_clean_words[concept_idx]
        if words:
            return words[:top_n]
        w_indices = self.concept_clusters[concept_idx]
        return [self.feature_names[i] for i in w_indices[:top_n]]

    def get_concept_question_text(self, concept_idx: int) -> str:
        """Get template-generated question text for a concept cluster."""
        return self.concept_questions[concept_idx]


def build_and_save_knowledge() -> KnowledgeBase:
    """Build knowledge base over development_entities.parquet."""
    dev_path = ANALYTICS_DIR / "development_entities.parquet"
    if not dev_path.exists():
        dev_path = CANONICAL_DIR / "canonical_entities.parquet"

    df = pd.read_parquet(dev_path)
    print(f"Building Knowledge Base over {len(df):,} entities...")

    kb = KnowledgeBase(df, num_concepts=200)
    return kb


if __name__ == "__main__":
    build_and_save_knowledge()
