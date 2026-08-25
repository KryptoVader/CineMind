"""
CineMind Hierarchical Question Generators (Phase 3).

Three generators competing every turn:
1. MetadataGenerator: Broad media_type, decade, language, genre splitters (55+ definitions)
2. ConceptClusterGenerator: Atomic semantic topic questions from 200 TF-IDF concept clusters
3. ContrastiveGenerator: Precise contrastive word extraction from raw overviews of top candidates
"""

from typing import Any, Callable, Optional
import numpy as np

from guesser.knowledge import KnowledgeBase
from guesser.belief import BeliefTracker


class CandidateQuestion:
    """A candidate question proposed by any generator."""
    def __init__(self, q_id: str, text: str, generator_type: str, get_p_yes_fn: Callable[[], np.ndarray]):
        self.q_id = q_id
        self.text = text
        self.generator_type = generator_type
        self.get_p_yes_fn = get_p_yes_fn


class MetadataGenerator:
    """Generates candidate metadata & semantic genre questions."""

    def __init__(self, kb: KnowledgeBase, tracker: BeliefTracker):
        self.kb = kb
        self.tracker = tracker

    def generate_questions(self, asked_q_ids: set[str]) -> list[CandidateQuestion]:
        candidates: list[CandidateQuestion] = []
        for idx, m_def in enumerate(self.kb.meta_defs):
            if m_def.feature_id in asked_q_ids:
                continue

            q_id = m_def.feature_id
            text = m_def.question_text
            meta_idx = idx

            # Empirical likelihood closure via BeliefTracker (per-category reliability + feedback recalibration)
            def make_p_yes_fn(m_i: int):
                return lambda: self.tracker.get_empirical_p_yes_metadata(m_i)

            candidates.append(CandidateQuestion(
                q_id=q_id,
                text=text,
                generator_type="metadata",
                get_p_yes_fn=make_p_yes_fn(meta_idx)
            ))
        return candidates


class ConceptClusterGenerator:
    """Generates atomic semantic topic questions from clean TF-IDF concept clusters."""

    def __init__(self, kb: KnowledgeBase, tracker: BeliefTracker):
        self.kb = kb
        self.tracker = tracker

    def generate_questions(self, asked_q_ids: set[str]) -> list[CandidateQuestion]:
        candidates: list[CandidateQuestion] = []

        for c_idx in range(self.kb.num_concepts):
            q_id = f"concept_cluster_{c_idx}"
            if q_id in asked_q_ids:
                continue

            # Skip low_quality clusters identified by classical NLP filtering
            if self.kb.concept_is_low_quality[c_idx]:
                continue

            text = self.kb.get_concept_question_text(c_idx)
            if not text:
                continue

            def make_p_yes_fn(c_i: int):
                return lambda: self.tracker.get_empirical_p_yes_concept(c_i)

            candidates.append(CandidateQuestion(
                q_id=q_id,
                text=text,
                generator_type="concept_cluster",
                get_p_yes_fn=make_p_yes_fn(c_idx)
            ))

        return candidates


class ContrastiveGenerator:
    """Generates precise contrastive keyword questions targeting top candidates and their dense LSA nearest neighbors."""

    def __init__(self, kb: KnowledgeBase, tracker: BeliefTracker, lam: float = 0.5):
        self.kb = kb
        self.tracker = tracker
        self.lam = lam

    def generate_questions(self, asked_q_ids: set[str], top_candidates_k: int = 5) -> list[CandidateQuestion]:
        candidates: list[CandidateQuestion] = []
        top_cands = self.tracker.get_top_candidates(k=top_candidates_k)

        if len(top_cands) < 2:
            return candidates

        # Requirement 3: Include top candidates BY posterior AND their dense LSA nearest neighbors
        candidate_entity_indices: set[int] = set()
        for cand, prob in top_cands:
            cid = cand["cinemind_id"]
            matched_idx = self.kb.df.index[self.kb.df["cinemind_id"] == cid]
            if len(matched_idx) > 0:
                ent_idx = matched_idx[0]
                candidate_entity_indices.add(ent_idx)
                # Add 3 nearest neighbors in 100-dim dense LSA space
                lsa_neighbors = self.kb.get_lsa_neighbors(ent_idx, k=3)
                candidate_entity_indices.update(lsa_neighbors)

        top_indices = list(candidate_entity_indices)
        if len(top_indices) < 2:
            return candidates

        proposed_words: set[str] = set()

        for target_ent_idx in top_indices:
            target_sparse = self.kb.tfidf_sparse[target_ent_idx]
            competitor_indices = [idx for idx in top_indices if idx != target_ent_idx]
            comp_mean_sparse = self.kb.tfidf_sparse[competitor_indices].mean(axis=0)

            # Fast sparse contrast score
            contrast_sparse = target_sparse - self.lam * comp_mean_sparse
            contrast_arr = np.asarray(contrast_sparse).ravel()

            best_w_indices = np.argpartition(contrast_arr, -3)[-3:]
            best_w_indices = best_w_indices[np.argsort(contrast_arr[best_w_indices])[::-1]]

            for w_idx in best_w_indices:
                score = contrast_arr[w_idx]
                if score <= 0.02:
                    continue

                word = self.kb.feature_names[w_idx]
                q_id = f"contrastive_{word}"

                if q_id in asked_q_ids or word in proposed_words:
                    continue

                proposed_words.add(word)
                text = f"Does the plot or story involve '{word}'?"

                def make_p_yes_fn(w_name: str):
                    return lambda: self.tracker.get_empirical_p_yes_word(w_name)

                candidates.append(CandidateQuestion(
                    q_id=q_id,
                    text=text,
                    generator_type="contrastive",
                    get_p_yes_fn=make_p_yes_fn(word)
                ))

        return candidates

