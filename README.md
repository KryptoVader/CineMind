# CineMind 🎬 — Classical ML Guesser Engine

> [!IMPORTANT]
> **Clean-Restart Status**: The previous experimental prototype codebase has been intentionally discarded. CineMind is being built from scratch as a modular, testable, classical Machine Learning system.
> 
> **Current Milestone Completed**: **Milestone 1 — Feature & Question Foundation** (`src/cinemind/`). The inference, game engine, evaluation, and UI components will be implemented in subsequent milestones.

---

## 🎯 Project Vision & Goals

**CineMind** is an Akinator-style guessing game for **movies**, **TV series**, and **anime**.

Given any entity in the user's mind, CineMind asks a sequence of structured, information-dense questions to split the entity space and guess the entity in as few turns as possible.

### Core Differentiator: Learning Human Perception
CineMind aims to learn from actual human gameplay:
1. How real players interpret and answer questions about media.
2. How human perception differs from raw, noisy dataset labels/metadata.
3. Which question predicates produce the most reliable and informative entity splits.
4. How to dynamically optimize guessing efficiency from recorded gameplay.

### Strict Classical Machine Learning Philosophy
CineMind is explicitly a **classical ML & statistical project**. 

**Strictly Excluded**:
- No Large Language Models (LLMs)
- No Transformers / BERT / sentence-transformers
- No Neural Networks or Deep Learning
- No Neural Embeddings or Generative AI

**Allowed & Planned Classical ML Methods**:
- Decision Trees & Information Theory (Shannon Entropy, Information Gain)
- Probabilistic Models & Recursive Bayesian Inference
- Logistic Regression & Naive Bayes
- Classical NLP (TF-IDF, SVD / Latent Semantic Analysis, N-grams)
- Clustering & Statistical Calibration

---

## 🎮 User Interaction Model

Player interaction is strictly structured around three canonical choices:
- **`YES`**
- **`NO`**
- **`UNKNOWN` / `DON'T KNOW`**

Free-text descriptions or unconstrained natural language queries are not used.

---

## 🏛️ System Architecture (`src/cinemind/`)

```text
src/cinemind/
├── data/
│   ├── feature_registry.py   # Formal Feature Registry & strict zero-leakage protection
│   ├── schemas.py            # Missingness semantics (KNOWN_TRUE, KNOWN_FALSE, UNKNOWN, NOT_APPLICABLE)
│   ├── entity.py             # Entity domain abstraction (decoupled from pandas columns)
│   └── loader.py             # Parquet dataset loader and entity instantiator
│
├── questions/
│   ├── schema.py             # Question representation & predicate evaluation engine
│   ├── generators.py         # 8 Structured Question Generators (Media, Genre, Language, Country, Time, Runtime, Episode, Rating)
│   ├── quality.py            # Offline Question Quality Profiler (Coverage, Balance, Missingness, Reliability, Answerability, Redundancy)
│   └── catalog.py            # Catalog manager & serialization (Parquet + JSON)
│
├── inference/                # Bayesian inference & belief tracking (Future Milestone)
├── engine/                   # Decision-tree splitting & question selection engine (Future Milestone)
├── evaluation/               # Simulation & benchmark harness (Future Milestone)
└── app/                      # User interface (Future Milestone)
```

---

## 🔒 Feature Classification & Zero-Leakage Guarantee

Every canonical entity attribute is formally registered in `FeatureRegistry` under one of 5 categories:

| Category | Description | Example Attributes |
| :--- | :--- | :--- |
| `DIRECT_QUESTION` | Direct categorical metadata suitable for human questions | `media_type`, `genres`, `original_language`, `origin_country`, `status`, `source_material`, `themes`, `demographics` |
| `TRANSFORMED_QUESTION` | Continuous/numeric data transformed into threshold predicates | `release_year`, `runtime`, `rating`, `num_episodes` |
| `NLP_DERIVED` | Unstructured text metadata reserved for classical NLP | `overview`, `keywords` |
| `LEARNING_ONLY` | Popularity/statistical metrics used for ML priors, never asked | `vote_count`, `popularity`, `rank`, `favorites`, `num_list_users` |
| `NEVER_EXPOSE` | **Strict identity & leakage fields (NEVER usable as questions)** | `title`, `original_title`, `alternative_titles`, `tmdb_id`, `mal_id`, `cinemind_id`, `source_id` |

> [!CAUTION]
> The `FeatureRegistry` explicitly raises a `ValueError` if any code attempts to construct a question from a `NEVER_EXPOSE` or `LEARNING_ONLY` attribute.

---

## ❓ Missingness Semantics

Missing metadata is **never collapsed into `False`**. CineMind preserves strict missingness distinctions:

- **`KNOWN_TRUE`**: Verified positive match for predicate.
- **`KNOWN_FALSE`**: Verified negative match for predicate.
- **`UNKNOWN`**: Missing, null, or empty metadata in source provider.
- **`NOT_APPLICABLE`**: Domain-inapplicable feature (e.g. `num_episodes` for standalone movies).

When evaluating a question against an entity with `UNKNOWN` or `NOT_APPLICABLE` state, the question returns `PlayerAnswer.UNKNOWN`, ensuring missing metadata is never treated as negative evidence.

---

## 📦 Data Sources & Warehouse

The canonical dataset combines data harvested from:
- **TMDB API v2/v3**
- **Official MyAnimeList API v2** *(Not Jikan)*

| Warehouse Table | Description | Path |
| :--- | :--- | :--- |
| **Canonical Entities** | 461k Merged Knowledge Universe | `data/canonical/canonical_entities.parquet` |
| **Diverse 100k Sample** | Stratified Sampler | `data/canonical/diverse_100k.parquet` |
| **Question Catalog** | Serialized Reproducible Question Pool | `data/model/question_catalog.parquet` / `.json` |

---

## 🚀 Running Tests & Catalog Generation

### Run Unit Tests
```bash
.venv\Scripts\python.exe -m pytest -v
```

### Build Reproducible Question Catalog
```bash
$env:PYTHONPATH="src"
.venv\Scripts\python.exe -m cinemind.questions.catalog
```

---

## 🗺️ Roadmap & Progression Plan

```text
1. Feature/question foundation      ← [COMPLETED - Milestone 1]
2. Simple deterministic baseline    ← [NEXT UP]
3. Belief/inference model
4. Adaptive question selection
5. Human-response learning
6. Classical NLP concept enrichment
7. Gameplay evaluation
8. Product / UI
```
