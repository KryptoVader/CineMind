"""
Integration test running CineMind V2 baseline engine over real development dataset slices (1,000 and 10,000 entities).
"""

from pathlib import Path
import sys
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from cinemind_v2.knowledge.entity_store import EntityStore
from cinemind_v2.knowledge.feature_store import FeatureStore
from cinemind_v2.game.engine import GameEngine


def test_v2_baseline_on_dev_dataset_1k():
    data_path = Path(__file__).resolve().parent.parent.parent / "src" / "data" / "analytics" / "development_entities.parquet"
    if not data_path.exists():
        print(f"Skipping dev dataset test (file not found at {data_path})")
        return

    # Load 1,000 entities
    store = EntityStore.from_parquet(data_path, limit=1000)
    fs = FeatureStore(store)

    assert store.num_entities == 1000
    assert fs.validate_alignment() is True

    engine = GameEngine(
        entity_store=store,
        feature_store=fs,
        target_posterior=0.75,
        max_questions=15,
    )

    # Pick an entity from the 1,000 subset to guess
    target_idx = 42
    target_entity = store.get_entity_by_index(target_idx)
    target_id = target_entity.cinemind_id

    print(f"\n[1K Test] Simulating game for target: '{target_entity.title}' ({target_id})")

    engine.start_new_game()
    turns = 0
    while not engine.state.is_over and turns < 15:
        q_item = engine.get_next_question()
        if not q_item:
            break
        q, ig_score = q_item
        feat_val = fs.get_feature(target_id, q.feature_id)
        answer = "YES" if feat_val else "NO"
        res = engine.answer_question(q, answer, ig_score=ig_score)
        turns += 1

    assert engine.state.is_over is True
    rank = engine.posterior.get_rank(target_id)
    print(f"[1K Test] Finished in {turns} questions. Target '{target_entity.title}' Rank: #{rank}")
    assert rank <= 5, f"Expected target in top 5, but got rank #{rank}"


def test_v2_baseline_on_dev_dataset_10k():
    data_path = Path(__file__).resolve().parent.parent.parent / "src" / "data" / "analytics" / "development_entities.parquet"
    if not data_path.exists():
        print(f"Skipping dev dataset test (file not found at {data_path})")
        return

    # Load 10,000 entities
    store = EntityStore.from_parquet(data_path, limit=10000)
    fs = FeatureStore(store)

    assert store.num_entities == 10000
    assert fs.validate_alignment() is True

    engine = GameEngine(
        entity_store=store,
        feature_store=fs,
        target_posterior=0.70,
        max_questions=20,
    )

    target_idx = 100
    target_entity = store.get_entity_by_index(target_idx)
    target_id = target_entity.cinemind_id

    print(f"\n[10K Test] Simulating game for target: '{target_entity.title}' ({target_id})")

    engine.start_new_game()
    turns = 0
    while not engine.state.is_over and turns < 20:
        q_item = engine.get_next_question()
        if not q_item:
            break
        q, ig_score = q_item
        feat_val = fs.get_feature(target_id, q.feature_id)
        answer = "YES" if feat_val else "NO"
        res = engine.answer_question(q, answer, ig_score=ig_score)
        turns += 1

    rank = engine.posterior.get_rank(target_id)
    print(f"[10K Test] Finished in {turns} questions. Target '{target_entity.title}' Rank: #{rank}")
    assert rank <= 10, f"Expected target in top 10, but got rank #{rank}"


if __name__ == "__main__":
    test_v2_baseline_on_dev_dataset_1k()
    test_v2_baseline_on_dev_dataset_10k()
    print("All real dataset integration tests passed successfully!")
