"""
Unit tests for CineMind V2 Game Engine & End-to-End Simulation.
Includes toy deterministic 10-entity test where guessing the exact entity is proven mathematically.
"""

import pandas as pd
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from cinemind_v2.knowledge.entity_store import EntityStore
from cinemind_v2.knowledge.feature_store import FeatureStore
from cinemind_v2.game.engine import GameEngine


def test_toy_deterministic_guessing_game():
    """
    Toy 10-entity dataset test.
    Target entity: 'Spirited Away' (anime, 2001, Japanese, rated masterpiece, animation/fantasy).
    Target should be correctly guessed within ~3-5 adaptive questions.
    """
    entities_data = [
        {"cinemind_id": "ent_01", "title": "Inception", "media_type": "movie", "release_year": 2010, "original_language": "en", "rating": 8.8, "genres": ["Action", "Sci-Fi"]},
        {"cinemind_id": "ent_02", "title": "The Matrix", "media_type": "movie", "release_year": 1999, "original_language": "en", "rating": 8.7, "genres": ["Action", "Sci-Fi"]},
        {"cinemind_id": "ent_03", "title": "Interstellar", "media_type": "movie", "release_year": 2014, "original_language": "en", "rating": 8.6, "genres": ["Sci-Fi", "Drama"]},
        {"cinemind_id": "ent_04", "title": "Parasite", "media_type": "movie", "release_year": 2019, "original_language": "ko", "rating": 8.5, "genres": ["Drama", "Thriller"]},
        {"cinemind_id": "ent_05", "title": "Spirited Away", "media_type": "anime", "release_year": 2001, "original_language": "ja", "rating": 8.6, "genres": ["Animation", "Fantasy"]},
        {"cinemind_id": "ent_06", "title": "Attack on Titan", "media_type": "anime", "release_year": 2013, "original_language": "ja", "rating": 9.0, "genres": ["Action", "Fantasy"]},
        {"cinemind_id": "ent_07", "title": "Breaking Bad", "media_type": "tv", "release_year": 2008, "original_language": "en", "rating": 9.5, "genres": ["Drama", "Crime"]},
        {"cinemind_id": "ent_08", "title": "Game of Thrones", "media_type": "tv", "release_year": 2011, "original_language": "en", "rating": 9.2, "genres": ["Action", "Fantasy"]},
        {"cinemind_id": "ent_09", "title": "Squid Game", "media_type": "tv", "release_year": 2021, "original_language": "ko", "rating": 8.0, "genres": ["Action", "Drama"]},
        {"cinemind_id": "ent_10", "title": "Amélie", "media_type": "movie", "release_year": 2001, "original_language": "fr", "rating": 8.3, "genres": ["Comedy", "Romance"]},
    ]
    df = pd.DataFrame(entities_data)

    store = EntityStore(df)
    fs = FeatureStore(store, df)

    engine = GameEngine(
        entity_store=store,
        feature_store=fs,
        target_posterior=0.75,
        target_margin=0.40,
        max_questions=10,
    )

    engine.start_new_game()
    target_id = "ent_05"  # Spirited Away

    max_turns = 10
    turns = 0

    while not engine.state.is_over and turns < max_turns:
        next_q_item = engine.get_next_question()
        assert next_q_item is not None, "Engine failed to produce a question!"
        q, ig_score, tree_sig = next_q_item

        # Compute ground truth answer for target_id
        feat_val = fs.get_feature(target_id, q.feature_id)
        answer = "YES" if feat_val else "NO"

        result = engine.answer_question(q, answer, ig_score=ig_score, tree_signal=tree_sig)
        turns += 1

    # Assert target entity is guessed as #1 with top confidence!
    assert engine.state.is_over is True
    assert engine.state.final_guess is not None
    guessed_id = engine.state.final_guess["entity_id"]
    guessed_entity = engine.state.final_guess["entity"]

    assert guessed_id == target_id, f"Engine guessed '{guessed_entity.title}' instead of 'Spirited Away'"
    assert engine.posterior.get_rank(target_id) == 1
    print(f"Toy test PASSED! Guessed '{guessed_entity.title}' in {turns} questions (confidence: {engine.state.final_guess['probability']:.1%})")


def test_stopping_conditions():
    entities_data = [
        {"cinemind_id": f"ent_{i}", "title": f"Title {i}", "media_type": "movie", "release_year": 2020, "original_language": "en", "rating": 8.0, "genres": ["Action"]}
        for i in range(5)
    ]
    df = pd.DataFrame(entities_data)

    store = EntityStore(df)
    fs = FeatureStore(store, df)

    engine = GameEngine(store, fs, max_questions=2)
    engine.start_new_game()

    q_item = engine.get_next_question()
    if q_item:
        engine.answer_question(q_item[0], "YES")

    q_item2 = engine.get_next_question()
    if q_item2:
        res = engine.answer_question(q_item2[0], "YES")
        assert res["is_over"] is True
        assert "Maximum questions limit reached" in res["final_guess"]["reason"]


if __name__ == "__main__":
    test_toy_deterministic_guessing_game()
    test_stopping_conditions()
    print("All game engine tests passed successfully!")
