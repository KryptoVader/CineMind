# CineMind

A guesser system for movies, TV series, and anime.

## Data Pipeline & Guesser Analytics Layer

The data pipeline lives in `src/pipeline/` and is responsible for acquiring, normalizing, deduplicating, canonicalizing, auditing, and building the ML/EDA analytics layer from multiple sources.

### Architecture

```
src/pipeline/
├── config.py              # Centralized configuration
├── http_client.py         # Resilient HTTP with retry/backoff
├── checkpoint.py          # Checkpoint/resume + graceful shutdown
├── tmdb/
│   ├── client.py          # TMDB API v3 client
│   ├── discovery.py       # Multi-strategy TMDB discovery
│   └── normalizer.py      # TMDB → normalized schema
├── mal/
│   ├── client.py          # Official MAL API v2 client
│   ├── discovery.py       # Multi-strategy MAL discovery
│   └── normalizer.py      # MAL → normalized schema
├── canonical/
│   ├── entity.py          # Canonical entity builder
│   ├── matcher.py         # Candidate-blocked cross-source entity resolution
│   └── sampler.py         # Deterministic stratified diversity sampler
├── analytics/
│   ├── builder.py         # Analytics views builder (development_*)
│   ├── validator.py       # Automated analytics dataset validation
│   └── report.py          # Master 30-section DEVELOPMENT_DATASET_REPORT generator
├── audit/
│   └── reports.py         # Master pipeline audit & quality reports
└── cli.py                 # CLI entry point
```

### Prerequisites

```bash
# Environment variables (in .env file)
TMDB_API_KEY=your_tmdb_api_key
MAL_CLIENT_ID=your_mal_client_id
```

### Commands

All commands run from the `src/` directory:

```bash
# --- DISCOVERY ---
# Full discovery (may run for hours — safe to interrupt and resume)
python -m pipeline.cli discover all

# TMDB only
python -m pipeline.cli discover tmdb

# MAL only
python -m pipeline.cli discover mal

# --- PROCESSING ---
# Normalize + deduplicate + build canonical dataset
python -m pipeline.cli process

# --- ENTITY RESOLUTION ---
# Cross-source matching (TMDB ↔ MAL)
python -m pipeline.cli match

# --- DIVERSITY SAMPLING ---
# Run stratified diversity sampling into diverse_100k.parquet
python -m pipeline.cli sample

# --- AUDITING ---
# Generate master pipeline audit reports (CINEMIND_DATA_AUDIT.md)
python -m pipeline.cli audit

# --- ANALYTICS & EDA DATASETS ---
# Build source-separated analytical datasets & DEVELOPMENT_DATASET_REPORT.md
python -m pipeline.cli analytics

# --- STATUS & EXPORT ---
python -m pipeline.cli status
python -m pipeline.cli export
```

### Data Directory Structure

```
src/data/
├── raw/                      # Raw TMDB & MAL API JSONL discovery dumps
├── staging/                  # Normalized TMDB & MAL Parquet files
├── canonical/
│   ├── canonical_entities.parquet    # Full canonical universe (~461k entities)
│   ├── diverse_100k.parquet          # Stratified development sample (~98k entities)
│   ├── entity_links.parquet          # Verified 1:1 cross-source links (6,445 links)
│   └── match_candidates.parquet      # Low-confidence match candidates
├── analytics/
│   ├── development_entities.parquet  # Primary development dataset (~98k records)
│   ├── development_tmdb.parquet      # TMDB entity view (81,426 records)
│   ├── development_mal.parquet       # MAL entity view (23,503 records)
│   ├── development_shared.parquet    # Verified shared cross-source view (6,453 records)
│   └── DEVELOPMENT_DATASET_REPORT.md # Master 30-section guesser readiness audit
├── audit/
│   └── CINEMIND_DATA_AUDIT.md        # Master 14-safeguard pipeline audit report
└── checkpoints/               # Discovery progress state files
```

### Guesser Architecture & Discrimination Attributes

CineMind is an **Akinator-style guesser system**. It progressive eliminates or re-ranks candidate entities based on user answers to binary/multi-choice questions.

Key attributes preserved for question discovery:
- `media_type`: High entropy initial splitting question
- `release_year` / `decade`: Temporal binary/decade questions
- `original_language`: Language filtering questions
- `genres`: Multi-label genre presence questions
- `rating`, `runtime`, `num_episodes`: Threshold questions
