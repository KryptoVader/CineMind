# CineMind

Preference-learning recommendation system for movies, TV series, and anime.

## Data Pipeline

The data pipeline lives in `src/pipeline/` and is responsible for acquiring, normalizing, deduplicating, and auditing the candidate title universe from multiple sources.

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
│   └── matcher.py         # Cross-source entity resolution
├── audit/
│   └── reports.py         # Population/quality/discovery reports
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

# Limited test runs
python -m pipeline.cli discover tmdb --years 2023 2024
python -m pipeline.cli discover mal --ranking-types all tv

# --- PROCESSING ---
# Normalize + deduplicate + build canonical dataset
python -m pipeline.cli process

# --- ENTITY RESOLUTION ---
# Cross-source matching (TMDB ↔ MAL)
python -m pipeline.cli match

# --- AUDITING ---
# Generate population, quality, and discovery reports
python -m pipeline.cli audit

# --- STATUS ---
# Check pipeline progress
python -m pipeline.cli status

# --- EXPORT ---
# Show candidate dataset summary
python -m pipeline.cli export
```

### Data Directory Structure

```
src/data/
├── raw/
│   ├── tmdb/
│   │   ├── discovery_movies.jsonl    # Raw TMDB movie records
│   │   ├── discovery_tv.jsonl        # Raw TMDB TV records
│   │   ├── genre_map.json            # Cached genre ID→name mapping
│   │   ├── movies/                   # Existing detailed JSON (preserved)
│   │   └── tv/                       # Existing detailed JSON (preserved)
│   └── mal/
│       └── discovery.jsonl           # Raw MAL anime records
├── staging/
│   ├── tmdb_normalized.parquet       # Normalized TMDB records
│   └── mal_normalized.parquet        # Normalized MAL records
├── canonical/
│   ├── candidates.parquet            # Full candidate universe
│   ├── cross_source_matches.parquet  # Confirmed cross-source matches
│   └── unresolved_matches.parquet    # Uncertain match candidates
├── checkpoints/
│   ├── tmdb_discovery_state.json     # TMDB discovery progress
│   └── mal_discovery_state.json      # MAL discovery progress
└── reports/
    ├── population_report.md          # Decade/language/genre distributions
    ├── data_quality_report.md        # Missing data, duplicates, validation
    └── discovery_contribution.md     # Per-strategy effectiveness
```

### Discovery Strategies

#### TMDB (4 strategies)

1. **Year-by-year**: `discover/movie` and `discover/tv` for every year 1900–current. Gets up to 10,000 results per year (500 pages × 20 results).

2. **Language segmentation**: For high-volume years (≥400 pages), repeats discovery with `with_original_language` filter for 26 languages (ja, ko, hi, zh, fr, de, es, etc.). Surfaces non-English titles buried beyond page 500.

3. **Genre × decade**: Crosses all TMDB genres with 10-year bins. Finds genre-specific titles that popularity sorting may bury.

4. **Low-popularity sweep**: Sorts by `vote_count.asc` with `vote_count.gte=1`. Surfaces obscure titles with minimal votes.

#### MAL (2 strategies)

1. **Ranking discovery**: Paginates 9 ranking types (all, airing, upcoming, tv, movie, ova, special, bypopularity, favorite). Follows `paging.next` until exhaustion.

2. **Systematic search**: ~200 search queries covering Japanese syllables, anime title words, genre terms, English words, and romanized concepts. Tracks marginal contribution per query.

### Resume / Checkpoint

The pipeline is designed for long-running acquisition (6–12+ hours for full discovery). Interrupting with Ctrl+C triggers a graceful shutdown:

1. Current API page finishes
2. Checkpoint saved to disk (atomic write)
3. Process exits cleanly

Restarting skips all completed tasks and resumes from the last checkpoint.

### Canonicalization

Each record gets a `cinemind_id`:
- TMDB records: `tmdb_{tmdb_id}`
- MAL records: `mal_{mal_id}`

Cross-source matching identifies when TMDB and MAL records refer to the same work (e.g., "Attack on Titan" ↔ "Shingeki no Kyojin"). Matches are stored with confidence levels (high/medium/low) and never auto-merged for uncertain cases.

### Deduplication

1. **Within-source**: Deduplicated by source ID during normalization
2. **Cross-source**: Conservative matching pipeline with title normalization, year compatibility, and media-type compatibility checks

### Known API Limitations

- **TMDB**: Hard 500-page limit per query (max 10,000 results). Language segmentation and genre×decade strategies work around this.
- **MAL**: Search queries require ≥3 characters. Ranking populations are finite (typically 10,000–30,000 per type).
- **TMDB discover**: Returns lightweight records (no credits, keywords, or production details). Full enrichment requires separate detail API calls.
- **MAL**: All entries assumed `original_language=ja`. Non-Japanese anime (Chinese donghua, Korean manhwa adaptations) will have incorrect language.

### Known Data Quality Limitations

- TMDB discover endpoint does not provide `production_countries` — only available via detail endpoint
- TMDB `origin_country` only available for TV shows, not movies
- Some TMDB entries have empty release dates or titles
- MAL `mean` (rating) is null for entries with insufficient votes
- Cross-source matching cannot use explicit external IDs without TMDB detail enrichment
- Fuzzy title matching may miss matches with very different romanizations
