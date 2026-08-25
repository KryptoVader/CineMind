# CineMind 🎬 `[PROTOTYPE / WORK IN PROGRESS]`

> [!WARNING]
> **PROTOTYPE STATUS**: CineMind is currently an **experimental prototype under active development**. While the core Bayesian engine, supervised genre models, SVD embeddings, and Two-Tier Information Gain ranking are functional, the feature engineering pipeline and data warehouse normalization are undergoing major active redesigns.

**CineMind** is an **Active Learning Bayesian Guesser Engine** for movies, TV series, and anime built over **461,188 entities**. 

Given any movie, TV series, or anime in a user's mind, CineMind asks a sequence of adaptive, information-dense questions to rapidly narrow down and guess the exact entity within 15–30 questions.

All ML/NLP models run **100% locally and offline** using pure classical statistics, information theory, matrix factorization, and supervised machine learning (zero LLM/API dependencies).

---

## 🏛️ System Architecture

```
c:\cinemind\src\
├── guesser/                      # Core Active Learning Guesser Engine
│   ├── knowledge.py              # Knowledge Base, Supervised Genre Models, SVD Embeddings, Concept Clusters
│   ├── belief.py                 # Bayesian Log-Posterior Tracker & Log-Popularity Priors
│   ├── generators.py             # 3 Hierarchical Question Generators (Metadata, Concept, Contrastive)
│   ├── engine.py                 # Vectorized Two-Tier Information Gain & Adaptive Tier Gating
│   ├── feedback.py               # Persistent Feedback Logger & Count-Weighted Bayesian Recalibration
│   ├── simulator.py              # Interactive CLI Guesser Game & Automated Testing Suite
│   └── experiments.py            # 100-Sample Incremental Benchmark Suite
│
├── pipeline/                     # Data Acquisition, Staging & Canonical Pipeline
│   ├── config.py                 # Centralized Configuration & File Paths
│   ├── http_client.py            # Resilient HTTP Client with Exponential Backoff
│   ├── checkpoint.py             # Checkpoint/Resume & State Management
│   ├── tmdb/                     # TMDB Discovery (363K Movies, 112K TV Shows)
│   ├── mal/                      # MAL Anime Discovery (23.5K Anime)
│   ├── canonical/                # Entity Resolution, Cross-Source Linking & Stratified Sampler
│   ├── analytics/                # Analytical View Builders & Readiness Audits
│   └── cli.py                    # Master Pipeline CLI Entrypoint
```

---

## 🚀 Quickstart & Usage

### 1. Requirements & Environment Setup

Ensure Python 3.10+ is installed and sync dependencies using `uv`:

```bash
# Clone and enter directory
cd c:\cinemind

# Install dependencies via uv
uv add nltk pandas numpy scikit-learn joblib pyarrow
```

### 2. Play Interactive Guesser Game

Think of any movie, TV show, or anime, then run:

```bash
cd src
python -m guesser.simulator
```

**Example Game Session**:
```text
============================================================
WELCOME TO CINEMIND — FULLY GENERATIVE AKINATOR GUESSER
============================================================
Think of any movie, TV series, or anime in your mind!
Answer: yes (y) / no (n) / dont know (k) / quit (q)
============================================================

Question #1 [METADATA] Top Guess: 'Shingeki no Kyojin' (0.0%)
  -> Is it a movie (not a TV series)?
Your answer (y/n/k/q): n

Question #2 [METADATA] Top Guess: 'Kimetsu no Yaiba' (0.0%)
  -> Is the original language English?
Your answer (y/n/k/q): y

Question #3 [CONCEPT_CLUSTER] Top Guess: 'Interstellar' (0.0%)
  -> Does the story take place in or involve a space, astronaut, planet?
Your answer (y/n/k/q): y

...
============================================================
CINEMIND GUESS RESULT
============================================================
  Are you thinking of:
  ★ INTERSTELLAR (2014)
    Media: movie | Language: en
    Confidence: 89.4% (in 18 questions)
============================================================
```

### 3. Run Automated Testing Benchmark

To run automated 20-sample accuracy evaluation against synthetic targets:

```bash
cd src
python -m guesser.simulator --test-auto --samples 20
```

### 4. Run Incremental Validation Suite

To evaluate accuracy across all 3 generator tiers over 100 randomized samples:

```bash
cd src
python -m guesser.experiments --samples 100
```

---

## 📐 Formal Mathematical Framework

### 1. Popularity Log-Prior Probability Distribution
For an entity universe $\mathcal{E} = \{e_1, e_2, \dots, e_N\}$ ($N = 461,188$), the prior probability $P(e_i)$ is defined via logarithmic normalization:

$$P(e_i) = \frac{\ln\left(1 + v_i + u_i + p_i\right)}{\sum_{j=1}^{N} \ln\left(1 + v_j + u_j + p_j\right)}$$

where $v_i$ is vote count, $u_i$ is user count, and $p_i$ is popularity score.

### 2. Recursive Bayesian Log-Posterior Update
Given turn history $\mathcal{H}_t = \{(q_1, a_1), \dots, (q_t, a_t)\}$, log-posterior probabilities are updated recursively:

$$\ln P(e_i \mid \mathcal{H}_t) = \ln P(e_i \mid \mathcal{H}_{t-1}) + \ln P(a_t \mid e_i, q_t) - C_t$$

where $C_t = \ln \sum_{j=1}^{N} \exp\left(\ln P(e_j \mid \mathcal{H}_{t-1}) + \ln P(a_t \mid e_j, q_t)\right)$.

### 3. Empirical Likelihood Models $P(\text{YES} \mid e_i, q)$

- **Structured Metadata Queries**:
  $$P(\text{YES} \mid e_i, q_{\text{meta}}) = \begin{cases} \eta & \text{if } e_i \text{ satisfies } q_{\text{meta}} \\ 1 - \eta & \text{otherwise} \end{cases}$$
  where $\eta = 0.90$ for structural attributes (media type, language, decade) and $\eta = 0.80$ for supervised genre classifications.

- **TF-IDF Concept Cluster Queries**:
  $$P(\text{YES} \mid e_i, q_{\text{concept}}) = \sigma\left(\lambda \cdot (s_i - \tau_{75})\right) = \frac{1}{1 + \exp\left(-\lambda (s_i - \tau_{75})\right)}$$
  where $\sigma$ is the logistic sigmoid function, $\lambda = 8.0$, and $\tau_{75}$ is the 75th percentile of non-zero TF-IDF overview sums.

### 4. Count-Weighted Bayesian Recalibration
To blend synthetic priors with real user answer distributions logged to disk (`data/feedback/game_feedback_logs.parquet`):

$$P_{\text{calibrated}}(\text{YES} \mid e_i, q) = \frac{k \cdot P_{\text{prior}}(\text{YES} \mid e_i, q) + n_{\text{YES}}}{k + N_{\text{obs}}}$$

where $n_{\text{YES}}$ is observed affirmative count, $N_{\text{obs}}$ is total observed answers, and $k = 5.0$.

### 5. Vectorized Information Gain ($IG$) Selection Criteria
The question $q^*$ selected at turn $t+1$ maximizes expected Shannon Information Gain:

$$q^* = \arg\max_{q \in \mathcal{Q}} \Big( H(E) - \left[ P(\text{YES} \mid q) H(E \mid \text{YES}, q) + P(\text{NO} \mid q) H(E \mid \text{NO}, q) \right] \Big)$$

where Shannon Entropy $H(E) = -\sum_{i=1}^{N} P(e_i \mid \mathcal{H}_t) \log_2 P(e_i \mid \mathcal{H}_t)$ in bits.

---

## 🧠 Machine Learning & NLP Component Architecture

### 1. Supervised One-vs-Rest Genre Classifiers (`knowledge.py`)
- **Model**: 25 independent One-vs-Rest `LogisticRegression(C=2.0, class_weight='balanced')` models trained on TF-IDF overview features on an **80/20 train/test split**.
- **Disk Caching**: Models are cached to `data/models/genre_classifiers.joblib` for instant startup.
- **Precision / Recall**: Drama (66.1% / 72.9%), Action (43.8% / 75.2%), Sci-Fi (35.0% / 70.6%).

### 2. NLTK POS & NER Proper-Noun Filtering (`knowledge.py`)
- **Classical NLTK Pipeline**: Integrates `nltk.pos_tag` and `nltk.ne_chunk` into concept cluster generation.
- **Filtering**: Excludes proper nouns (`NNP`/`NNPS`) and named entities (`PERSON`, `GPE`, `ORGANIZATION`) to prevent unanswerable character-name questions.

### 3. Dense 100-dim SVD Entity Embeddings (`knowledge.py`, `generators.py`)
- **LSA Embedding Matrix**: Projects 5,000-dim TF-IDF overview space into **100-dim dense SVD component space** (`entity_lsa_normalized`).
- **Semantic Contrastive Generation**: `ContrastiveGenerator` uses LSA cosine nearest-neighbors (`get_lsa_neighbors()`) to extract discriminating keywords contrasting top candidates and their dense semantic neighbors.

---

## 📊 Data Warehouse & Data Pipeline (`pipeline/`)

CineMind includes a resilient data ingestion pipeline that harvests, normalizes, deduplicates, and canonicalizes over **460,000 entities**:

| Data Warehouse Layer | File Path | Record Count ($N$) | File Size |
| :--- | :--- | :--- | :--- |
| **Raw TMDB Movies** | `data/raw/tmdb/discovery_movies.jsonl` | 363,264 | 236.94 MB |
| **Raw TMDB TV** | `data/raw/tmdb/discovery_tv.jsonl` | 112,563 | 69.47 MB |
| **Raw MAL Anime** | `data/raw/mal/discovery.jsonl` | 23,503 | 28.48 MB |
| **Staged TMDB Normalized** | `data/staging/tmdb_normalized.parquet` | 444,138 | 81.63 MB |
| **Staged MAL Normalized** | `data/staging/mal_normalized.parquet` | 23,503 | 7.84 MB |
| **Canonical Merged Table** | `data/canonical/canonical_entities.parquet` | **461,188** | 91.90 MB |

---

## ⚡ Performance Summary

- **Question Selection Latency**: **~12 ms / turn**
- **Memory Footprint**: ~350 MB (TF-IDF sparse matrix + SVD embeddings + metadata matrices)
- **Offline Local Execution**: 100% self-contained, 0 external API calls during gameplay
- **Entity Coverage**: 461,188 titles (360,208 movies, 77,477 TV shows, 23,503 anime)

---

## 🚧 Prototype Development Roadmap

1. **Feature Engineering Redesign**: Rebuilding feature stores directly on top of the full **461,188 scraped data warehouse entities**.
2. **Dedicated TMDB Keywords Fetch**: Implementing a secondary API crawler targeting `GET /movie/{id}/keywords` and `GET /tv/{id}/keywords` (currently 0% present in initial discovery dumps).
3. **Canonical Genre Normalization**: Unifying compound genre tags (`Sci-Fi & Fantasy` $\rightarrow$ `Science Fiction`, `Fantasy`; `Action & Adventure` $\rightarrow$ `Action`, `Adventure`).
4. **Source-Aware Percentile Prior Scaling**: Normalizing TMDB `popularity` and MAL `popularity` (rank index vs continuous score) using source-specific percentile ranks for Bayesian priors $P(e_i)$.
5. **Anime-Specific Metadata Features**: Adding dedicated question generators for `source_material` (`manga`, `light_novel`, `original`), `studios`, and content certifications (`pg_13`, `tv_ma`).
