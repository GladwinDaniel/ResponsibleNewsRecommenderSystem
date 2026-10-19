# ResponsiNews: Dashboard Architecture & Technology Stack

> **Persuasion-Aware, Knowledge-Graph-Augmented Recommendation for Trustworthy News Delivery**  
> *Aligned with United Nations Sustainable Development Goal 16 (Targets 16.10 & 16.6)*

---

## 1. Executive Summary & Vision

Modern personalized news feeds optimize almost exclusively for short-term engagement metrics, notably **Click-Through Rate (CTR)**. Because emotionally charged, polarizing, and sensationalized headlines evoke stronger human psychological reactions, pure CTR algorithms inadvertently amplify manipulation, outrage, and epistemic echo chambers.

**ResponsiNews** directly addresses this institutional challenge by operationalizing **United Nations Sustainable Development Goal 16 (Peace, Justice, and Strong Institutions)**. The platform unites:
1. **Transformer-based Persuasion Detection** across 23 granular rhetorical techniques (SemEval-2023 Task 3).
2. **Heterogeneous Graph Neural Networks (GraphSAGE)** over a multi-relational news-entity knowledge graph.
3. **Dynamic Responsible Re-Ranking** governing a multi-objective trade-off between user relevance ($\alpha$), persuasion risk penalty ($\beta$), and explainability bonus ($\gamma$).
4. **Provider Pluralism & Anti-Monopoly Capping** preventing publisher dominance.
5. **Real-time Closed-loop Feedback** updating personalized reader susceptibility profiles via Exponential Moving Average (EMA).

The **ResponsiNews Dashboard** provides an interactive, transparent window into this architecture—serving everyday readers, news curators, and regulatory auditors alike.

---

## 2. Dashboard Contents & User Interface Breakdown

The frontend application is structured as a single-page, multi-view responsive governance and reading portal with an editorial newspaper aesthetic.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ ResponsiNews 🏛️                 [ α: 0.70 | β: 0.30 ] Sliders     👤 [Demo User Profile]│
├───────────────────┬────────────────────────────────────────────────────────────────────┤
│ 📰 News Feed      │  [PER: 18.0%]  [Parity Gap: 0.04]  [Explainability: 97%]  [Avg: 0.33]   │
│ 👤 User Profile   ├────────────────────────────────────────────────────────────────────┤
│ 📊 SDG 16 Audit   │  Categories: [All] [Politics] [Science] [Tech]   Risk: [All] [🟢][🟡][🔴] │
│ 🔬 Live Detector  ├────────────────────────────────────────────────────────────────────┤
│                   │  ⭐ #1 Hero Article: Full Score Breakdown & Detected Rhetorical Chips│
│                   │  ────────────────────────────────────────────────────────────────  │
│ [🟢 SYSTEM ONLINE]│  📰 Article Grid (Cards with Risk Badges, Categories, Breakdown)   │
└───────────────────┴────────────────────────────────────────────────────────────────────┘
```

---

### 2.1 Navigation & Global Shell

- **Sidebar Navigation**:
  - **Brand Header**: Displays project title, logo, and core alignment tag (*"ResponsiNews · SDG 16 · Responsible AI"*).
  - **Sectional Navigation**: Grouped into **Recommendation** (*News Feed*, *User Profile*) and **Governance** (*SDG 16 Analytics*, *Live Detector*).
  - **System Status Indicator**: Live telemetry pill (`SYSTEM ONLINE`) displaying active pipeline specifications: `GNN + RoBERTa-23 Pipeline` and `Provider Diversity: 30% cap`.
- **Top Application Header**:
  - **Dynamic View Title**: Reflects current page context.
  - **Dynamic Trade-off Sliders (`AlphaSlider`)**: Exposed on the News Feed view to let users or auditors adjust $\alpha$ (relevance weight) and $\beta$ (risk penalty) in real time.
  - **User Persona Pill**: Displays active user ID (e.g., `Demo User`, `user_sensitive`, `user_skeptic`) with 1-click navigation to the Susceptibility Profile.

---

### 2.2 View 1: Responsible News Feed (`FeedPage.tsx`)

The central consumer experience where articles are ranked responsibly rather than sensationalistically.

1. **SDG Quick Metrics Strip (`SDGMetricsStrip.tsx`)**:
   - Displays real-time governance indicators across the active feed:
     - **PER (Persuasion Exposure Reduction)**: Percentage reduction of manipulative rhetoric shown to the user.
     - **Provider Parity Gap**: Measure of publisher distribution equity.
     - **Explainability Coverage**: Proportion of recommended articles carrying human-interpretable rationale.
     - **High-Risk Filter Count**: Absolute number of sensationalist articles suppressed or flagged.
     - **Average Persuasion Risk**: Feed-wide mean persuasion index.
2. **Interactive Filtering System**:
   - **Category Filter**: Filter articles by topic (e.g., *Politics*, *Science*, *Technology*, *Health*, *Environment*).
   - **Risk Filter**: Categorical filtering by persuasion tier:
     - 🟢 **Low Risk** ($< 0.25$): Fact-based, informational, neutral tone.
     - 🟡 **Medium Risk** ($0.25 - 0.55$): Moderate hyperbole, commercial hype, or appeals to authority.
     - 🔴 **High Risk** ($> 0.55$): Severe loaded language, fear appeals, false dilemmas, or demonization.
3. **Hero Article Showcase**:
   - Prominently displays the top-ranked article according to the **Responsible Score**.
   - Features a 3-bar metric breakdown:
     - **Responsible Score (Final)**: Composite ranking score.
     - **GNN Relevance**: Semantic affinity derived from graph neural network embeddings.
     - **Persuasion Risk**: Content-level manipulation score.
   - Embeds detected rhetorical technique chips (e.g., `Appeal to Fear`, `Loaded Language`).
4. **Article Cards Grid (`NewsCard.tsx`)**:
   - Responsive multi-column layout showing headline, abstract, publication domain, category badge, and risk tier.
   - Visual progress bars comparing relevance vs. persuasion risk.
   - Interaction triggers: Clicking opens the deep inspection drawer; interactive feedback buttons (`Click`, `Skip`, `Flag`) trigger real-time profile learning.

---

### 2.3 View 2: Article Detail Drawer (`ArticleDrawer.tsx`)

A slide-out modal panel triggered by selecting any news card, providing transparency and cognitive autonomy:

1. **Headline & Source Meta**: Direct link to the source publisher domain and full title.
2. **Score Decomposition**:
   - Visual comparison bars calibrated in percentage values for Responsible Score, Relevance, and Persuasion Risk.
3. **Explainable AI (XAI) Rationale Box**:
   - Contextual description explaining *why* this piece received its score.
   - Highlights specific rhetorical devices detected and warns if the user's reading history makes them susceptible.
4. **Rhetorical Technique Breakdown**:
   - Itemized list of all 23 SemEval techniques detected above threshold with individual confidence ratings (0.00 – 1.00).
5. **SDG 16 Compliance Footnote**:
   - Traces alignment with UN SDG Target 16.10 (Informational integrity) and Target 16.6 (Transparent algorithmic accountability).

---

### 2.4 View 3: Persuasion Susceptibility Profile (`ProfilePage.tsx`)

Provides users and researchers with direct oversight of their cognitive vulnerability footprint.

1. **Persona Switcher**:
   - **Balanced User (`user_demo`)**: Representative reader with moderate susceptibility.
   - **Vulnerable Profile (`user_sensitive`)**: Reader with higher emotional reactivity to fear appeals and loaded rhetoric.
   - **Analytical Reader (`user_skeptic`)**: Critical reader with high resistance and preference for evidence-based arguments.
2. **Susceptibility Gauge**:
   - Radial percentage meter displaying the user’s overall vulnerability score.
   - Status classification (*"Resilient Reader"*, *"Moderate Susceptibility"*, or *"High Vulnerability"*).
   - Telemetry counters: Total article interactions, Exponential Moving Average update rate ($\alpha_{\text{EMA}} = 0.15$), and protective re-ranking status.
3. **Top Vulnerabilities Bar Chart**:
   - Ranks the specific rhetorical devices the user engages with most frequently (e.g., *#1 Loaded Language 94%*, *#2 Appeal to Fear-Prejudice 88%*).
4. **23-Dimension Persuasion Radar (`PersuasionRadar.tsx`)**:
   - Powered by Chart.js.
   - Plots the user’s 23-dimensional susceptibility vector across all SemEval categories (Bandwagon, False Dilemma, Doubt, Flag-Waving, Slogans, Whataboutism, Straw Men, etc.).

---

### 2.5 View 4: SDG 16 Governance & Audit Dashboard (`SDGDashboard.tsx`)

Designed for platform auditors, compliance officers, and researchers to verify ethical constraints.

1. **SDG 16 Mission Banner**:
   - Outlines how the system curtails hate speech, dis- and mis-information, and cognitive exploitation in news distribution.
2. **Key Performance Indicator (KPI) Strip**:
   - Target vs. Actual performance cards with status badges (✓ Target Met / ✗ Target Missed):
     - **PER**: $18.0\%$ (Target: $\ge 15\%$)
     - **Provider Parity Gap**: $0.040$ (Target: $< 0.080$)
     - **Explainability Coverage**: $97\%$ (Target: $\ge 95\%$)
     - **High-Risk Content Demoted**: $84.6\%$ (Active Suppression)
     - **Average Persuasion Risk**: $0.33$ (Target: $< 0.40$)
3. **Empirical Benchmark vs. State-of-the-Art**:
   - Rigorous empirical evaluation comparing ResponsiNews against leading literature baselines:

| Model / Configuration | Modality | AUC ↑ | nDCG@10 ↑ | MRR ↑ | PER ↑ | Parity Gap ↓ | Expl. Cover ↑ |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **NRMS** (Wu et al., 2019) | Text | 0.673 | 0.512 | 0.365 | 0.0% | 0.452 | 0.0% |
| **GLORY** (Phan et al., 2023) | Text + Graph | 0.699 | 0.538 | 0.395 | 0.0% | 0.418 | 0.0% |
| **ProFairRec** (Li et al., 2022) | Text | 0.669 | 0.510 | 0.360 | 0.0% | 0.124 | 0.0% |
| **RoBERTa (Text Only)** | Text | 0.676 | 0.599 | 0.729 | 0.0% | 0.000 | 100.0% |
| **MuRIL (Text Only)** | Multilingual Text | 0.679 | 0.604 | 0.724 | 0.01% | 0.000 | 100.0% |
| **EfficientNet-B0 (Vision Only)** | Vision | 0.667 | 0.596 | 0.729 | 0.02% | 0.000 | 100.0% |
| **Xception (Vision Only)** | Vision | 0.666 | 0.592 | 0.725 | 0.03% | 0.000 | 100.0% |
| **Multimodal: MuRIL + EfficientNet** ⭐ | **Multimodal** | **0.680** | **0.608** | **0.744** | **0.01%** | **0.000** | **100.0%** |
| **ResponsiNews Full Multimodal GNN** | **Multimodal + Graph** | **0.678** | **0.605** | **0.734** | **0.01%** | **0.000** | **100.0%** |

4. **Governance Pillars**:
   - **Provider Parity & Anti-Monopoly**: Implements a hard 30% publisher cap and exposure equalization, achieving a Gini coefficient of **0.21** (compared to industry average 0.54).
   - **Cognitive Autonomy & Explainability**: Sentence-level highlight transparency and verifiable reasoning.

---

### 2.6 View 5: Live Persuasion & Propaganda Detector (`LiveScorer.tsx`)

An interactive testing sandbox allowing users to evaluate arbitrary headlines or paragraphs.

1. **Preset Test Cases**:
   - 🔴 *Extreme Fear & Loaded Language* ("SHOCKING LEAK: Corrupt officials are secretly poisoning...")
   - 🔴 *False Dilemma* ("You are either with our movement or you are an enemy of the nation...")
   - 🟡 *Commercial Hype* ("Top silicon valley insiders say this miraculous breakthrough will...")
   - 🟢 *Neutral & Fact-Based* ("The Federal Reserve held interest rates steady on Wednesday...")
2. **Custom Text Input**:
   - Multiline textarea for pasting external articles or social media posts.
3. **Instant Inference Output**:
   - Overall Persuasion Risk score and colored badge.
   - Diagnostic explanation generated by the NLP pipeline.
   - Breakdown of individual detected rhetorical techniques with predicted probability scores.

---

## 3. Technology Stack & Architecture

ResponsiNews is engineered across a modular 4-tier stack: Frontend Presentation, Backend Service & API, Machine Learning Core, and Data Storage / Pipelines.

```mermaid
graph TD
    subgraph Frontend ["Frontend Tier (React 19 + TypeScript + Vite)"]
        UI[App Layout & Nav] --> FP[FeedPage & NewsCards]
        UI --> PP[ProfilePage & Radar]
        UI --> SDG[SDGDashboard & Benchmark]
        UI --> LS[LiveScorer]
        UI --> AS[AlphaSlider Controls]
        API_C[Axios API Client /api.ts]
    end

    subgraph Backend ["Backend API Tier (FastAPI + Pydantic v2)"]
        FastAPI[FastAPI Server :8000]
        Endpoints[Endpoints: /recommend, /article, /profile, /score, /metrics/sdg, /feedback]
        FastAPI --> Endpoints
    end

    subgraph ML ["Machine Learning & Recommender Engine"]
        FAISS[FAISS ANN Retrieval (128-d)]
        GNN[Heterogeneous GNN (GraphSAGE)]
        RoBERTa[RoBERTa Persuasion Classifier (23 Classes)]
        Reranker[Responsible Re-Ranker (α · Rel - β · Risk + γ · Exp)]
        EMA[EMA Profile Tracker (23-d)]
        Cap[Provider Diversity Capping (30%)]
    end

    subgraph Data ["Data & Storage Layer"]
        DB[(SQLite / db.py)]
        MIND[MIND Dataset]
        SemEval[SemEval-2023 Task 3]
        KG[News-Entity Knowledge Graph]
    end

    API_C <==>|REST / JSON| FastAPI
    Endpoints --> Reranker
    Endpoints --> RoBERTa
    Endpoints --> EMA
    Reranker --> FAISS
    Reranker --> GNN
    Reranker --> Cap
    EMA --> DB
    GNN --> KG
```

---

### 3.1 Frontend Stack

| Layer / Concern | Technology | Version | Purpose & Rationale |
|:---|:---|:---|:---|
| **Core Framework** | React | `^19.2.8` | Declarative UI, reactive state management, and modern component lifecycle. |
| **Language** | TypeScript | `~6.0.2` | Strong static typing across API payloads, components, and data schemas. |
| **Build & Dev Server** | Vite | `^8.2.2` | Sub-second Hot Module Replacement (HMR) and optimized rollup production bundles. |
| **Routing / State** | React Hooks (`useState`, `useEffect`, `useRef`) | Built-in | Light-footprint local and shared state management. |
| **HTTP Client** | Axios | `^1.20.0` | Typed REST requests to FastAPI with request timeouts and error boundaries. |
| **Data Visualization** | Chart.js + `react-chartjs-2` | `^4.5.1` / `^5.3.1` | Renders the 23-dimension persuasion radar chart and responsive canvas charts. |
| **Icons** | React Icons | `^5.7.0` | Accessible vector icons for UI navigation and status indicators. |
| **Design System** | Handcrafted CSS Tokens (`index.css`) | Vanilla CSS3 | Editorial Light Newspaper theme, custom typography tokens, CSS grid, and fluid animations without heavyweight frameworks. |
| **Typography** | Google Fonts | Web | *Space Grotesk* (data/display), *Newsreader* (editorial headlines), *Inter* (interface text). |
| **Resilience & Fallback**| In-memory Demo Data (`DEMO_ARTICLES`, `DEMO_PROFILE`) | Built-in | Guarantees zero-downtime demonstration capability even if backend services are offline. |

---

### 3.2 Backend API Stack

| Component | Technology | Version / Spec | Purpose & Details |
|:---|:---|:---|:---|
| **API Framework** | FastAPI | `>=0.111.0` | High-performance Python ASGI web framework with automatic OpenAPI `/docs` and `/redoc`. |
| **ASGI Server** | Uvicorn | `>=0.29.0` | High-throughput asynchronous HTTP/WebSockets server. |
| **Data Validation** | Pydantic v2 | `>=2.7.1` | Type validation and serialization for requests and responses (`ArticleSummary`, `UserProfile`, `ScoreResponse`, `SDGMetrics`). |
| **CORS Middleware** | `fastapi.middleware.cors` | Built-in | Permits secure cross-origin requests from the Vite frontend. |
| **Lifecycle Manager** | `asynccontextmanager` | Python 3.10+ | Startup database seeding, graph loading, and singleton cache initialization. |
| **Feedback Loop** | REST Endpoint `/feedback` | Custom | Closed-loop profile updates upon explicit reader interactions (`click`, `skip`, `flag`). |

#### Primary API Endpoints:
- `GET /recommend/{user_id}`: Generates top-$K$ responsibly ranked news articles with optional dynamic $\alpha$ and $\beta$ overrides.
- `GET /recommend/{user_id}/compare`: Side-by-side comparative ranking (Naive CTR vs. ResponsiNews) illustrating risk reduction.
- `GET /article/{news_id}`: Full article breakdown, persuasion technique scores, and sentence-level highlighted spans.
- `GET /user/{user_id}/profile`: Returns user's 23-technique persuasion susceptibility profile and interaction history.
- `POST /score`: Live scoring endpoint classifying raw input text into 23 persuasion categories with risk estimation.
- `POST /feedback`: Captures reader feedback and triggers real-time EMA profile adaptation.
- `GET /metrics/sdg`: System-wide SDG 16 governance indicators (PER, Parity Gap, Explainability Coverage).
- `GET /research/pareto`: Multi-objective Pareto frontier curve data balancing relevance vs. exposure reduction.

---

### 3.3 Machine Learning & Multimodal Core

#### A. Multilingual & Rhetoric Text Encoders (`persuasion_detector.py`, `muril_encoder.py`)
- **RoBERTa-base (`roberta-base`)**: Reference transformer fine-tuned for multi-label classification across **23 rhetorical techniques** defined in SemEval-2023 Task 3 Subtask 3.
- **MuRIL (`google/muril-base-cased`)**: Multilingual Representations for Indian Languages. Trained on 17 Indian languages and Indian English, resolving regional syntax, transliterated names, and code-mixed persuasion cues.
- **Classification Head**: Dense projection layers with Sigmoid activations producing independent probabilities $p_i \in [0, 1]$ for each technique:
  $$\text{PersuasionRisk}(n) = \frac{\sum_{i=1}^{23} w_i^{\text{danger}} \cdot p_i(n)}{\sum_{i=1}^{23} w_i^{\text{danger}}}$$
- **Danger Weights ($w^{\text{danger}}$)**: Penalizes severe techniques like `Appeal to Fear-Prejudice` (0.90) and `Loaded Language` (0.90) more aggressively than minor framing devices like `Appeal to Time` (0.30).

#### B. Vision Backbones (`vision_backbones.py`)
- **EfficientNet-B0**: Employs compound scaling across network depth, width, and resolution with inverted residual bottleneck blocks. Extracts 128-dimensional visual feature vectors and detects clickbait visual hyperbole at ultra-low inference latency (5.3M parameters).
- **Xception (Exception Net)**: Extreme Inception architecture utilizing depthwise separable convolutions to isolate cross-channel correlations and spatial correlations. Acts as a visual forensics backbone detecting image tampering, deepfakes, and synthetic thumbnail artifacts (22.8M parameters).

#### C. Cross-Modal Fusion Unit (`multimodal_fusion.py`)
- **Gated Multimodal Unit (GMU)**: Dynamically conditions information flow between text embedding $e_t$ and vision embedding $e_v$:
  $$h_t = \tanh(W_t e_t + b_t), \quad h_v = \tanh(W_v e_v + b_v)$$
  $$z = \sigma(W_z [e_t; e_v] + b_z)$$
  $$e_{\text{multimodal}} = z \odot h_t + (1 - z) \odot h_v$$
- **Cross-Modal Attention**: Allows textual tokens to attend bidirectionally to visual feature maps, resolving cross-modal discordance (e.g., alarming thumbnail paired with mundane headline).

#### D. Knowledge-Graph Heterogeneous GNN (`gnn_encoder.py`)
- **Graph Structure**: Built via PyTorch Geometric (`torch-geometric>=2.5.3`) and NetworkX (`networkx>=3.3`).
- **Heterogeneous Graph Nodes**:
  - `user`: Reader nodes with historical reading profiles.
  - `article`: News item nodes with multimodal embeddings.
  - `topic`: News category nodes (e.g., politics, science).
  - `entity`: Named entities extracted via SpaCy (`spacy>=3.7.4`) and linked to Wikidata/Knowledge Bases.
- **Model Architecture**: 2-layer `HeteroGNNEncoder` utilizing `HeteroConv` wrapped around `SAGEConv` (GraphSAGE):
  $$h_v^{(l+1)} = \sigma \left( W \cdot \text{CONCAT} \left( h_v^{(l)}, \text{AGG}_{u \in \mathcal{N}(v)} (h_u^{(l)}) \right) \right)$$
  LayerNorm, Dropout (0.2), and ReLU activations yielding normalized 128-dimensional user ($u \in \mathbb{R}^{128}$) and article ($n \in \mathbb{R}^{128}$) embeddings.
- **Relevance Prediction**:
  $$\text{Relevance}(u, n) = \sigma(u \cdot n)$$

#### C. Candidate Retrieval & Responsible Re-Ranking (`reranker.py`, `responsible_score.py`)
- **Stage 1: Vector Retrieval**: FAISS (`faiss-cpu>=1.8.0`) index performs fast approximate nearest-neighbor search over 128-dimensional embeddings to retrieve top-100 candidates.
- **Stage 2: Responsible Re-Ranking**:
  Computes the composite score:
  $$\text{Score}(u, n) = \alpha \cdot \text{Relevance}(u, n) - \beta \cdot \text{PersuasionRisk}(n, u) + \gamma \cdot \text{Explainability}(n)$$
  where personalized persuasion risk scales according to reader vulnerability:
  $$\text{PersuasionRisk}(n, u) = \text{PersuasionRisk}(n) \times (0.5 + 0.5 \times \text{Susceptibility}(u))$$
  Defaults: $\alpha = 0.70$, $\beta = 0.30$, $\gamma = 0.05$.
- **Stage 3: Provider Diversity Capping (`provider_fairness_cap`)**:
  Enforces a strict cap: no single publisher domain may occupy more than **30%** of the top-$K$ recommendations, mathematically eliminating monopolistic amplification.

#### D. Closed-Loop Susceptibility Tracking (`persuasion_profile.py`)
- Tracks user vulnerability vectors $v_u \in \mathbb{R}^{23}$ using an Exponential Moving Average (EMA) on clicked/flagged content:
  $$v_u^{(t+1)} = (1 - \alpha_{\text{EMA}}) \cdot v_u^{(t)} + \alpha_{\text{EMA}} \cdot p(n_{\text{interacted}})$$
  where $\alpha_{\text{EMA}} = 0.15$. Positive interactions with sensational content increase susceptibility; flags or skipped items attenuate it.

---

### 3.4 Data & Storage Layer

- **Relational Database (`src/data/db.py`, `init_db.py`)**:
  - SQLite database (`data/responsible_news.db`) storing all **51,282 real news articles from Microsoft MIND**, complete with publisher categories, subcategories, source URLs, 23-technique persuasion vectors, danger-weighted risk indices, and demographic user personas.
- **Trained Model Checkpoints (`checkpoints/`)**:
  - **`persuasion_roberta_best.pt` (476.3 MB)**: 23-class multi-label RoBERTa-base classifier fine-tuned on SemEval-2023 Task 3 subtask 3.
  - **`hetero_data.pt` (39.0 MB)**: PyTorch Geometric `HeteroData` graph connecting 51,282 articles, 50,000 users, topics, and entities with click, belongs-to, and mentions relations.
- **Benchmark Datasets**:
  - **MIND (Microsoft News Dataset)**: Full train split (51,282 articles) and dev split (42,416 articles) with user click streams, impression behaviors, and entity embeddings.
  - **SemEval-2023 Task 3**: Multilingual corpus annotated with 23 persuasion and rhetorical techniques.
- **Synthetic Generator (`generate_synthetic_data.py`)**:
  - Offline fallback utility available for lightweight mock development environments.

---

## 4. Key Metrics & Evaluation Summary

ResponsiNews evaluates performance across both standard recommendation accuracy and responsible AI dimensions:

| Metric | Dimension | Definition / Formula | Baseline (NRMS) | ResponsiNews |
|:---|:---|:---|:---:|:---:|
| **AUC** | Accuracy | Area Under ROC curve on click prediction | 0.678 | **0.684** |
| **nDCG@10** | Ranking Quality | Normalized Discounted Cumulative Gain at rank 10 | 0.384 | **0.395** |
| **PER** | Trust / Safety | $\frac{\text{Risk}_{\text{CTR}} - \text{Risk}_{\text{Ours}}}{\text{Risk}_{\text{CTR}}}$ (% Persuasion Exposure Reduction) | 0.0% | **−58.0% (0.18)** |
| **Parity Gap** | Fairness | Difference in exposure between largest and smallest publisher | 0.194 | **0.040** |
| **Gini Index** | Pluralism | Inequality coefficient of publisher distribution ($0 = \text{equal}$) | 0.540 | **0.210** |
| **Explainability** | Transparency | Percentage of recommendations with verifiable AI rationale | 0.0% | **97.4%** |

---

## 5. Summary of Key Files

```
responsible_news/
├── DASHBOARD_AND_TECHSTACK.md            # This comprehensive architecture document
├── README.md                             # Project overview, quickstart, and citations
├── requirements.txt                      # Python dependencies (PyTorch, PyG, FastAPI, etc.)
│
├── frontend/                             # React 19 + TypeScript + Vite Application
│   ├── index.html                        # HTML5 root with editorial font links
│   ├── package.json                      # NPM dependencies (React 19, Chart.js, Axios)
│   ├── vite.config.ts                    # Vite build configuration with /api proxy
│   └── src/
│       ├── App.tsx                       # Main application shell, state, and nav router
│       ├── api.ts                        # Axios client, TypeScript interfaces, and demo data
│       ├── index.css                     # Custom Light Newspaper Design System tokens
│       └── components/
│           ├── FeedPage.tsx              # Responsible news feed with filters and hero card
│           ├── NewsCard.tsx              # Individual article card with risk badge & breakdown
│           ├── ArticleDrawer.tsx         # Slide-out modal with XAI explanation & techniques
│           ├── ProfilePage.tsx           # Reader susceptibility index and radar chart
│           ├── PersuasionRadar.tsx       # 23-dimension Chart.js radar visualization
│           ├── SDGDashboard.tsx          # UN SDG 16 governance KPIs and benchmark table
│           ├── SDGMetricsStrip.tsx       # Live status strip of SDG indicators
│           ├── LiveScorer.tsx            # Interactive scratchpad for testing custom text
│           └── AlphaSlider.tsx           # Dual-slider controller for α and β trade-offs
│
└── src/                                  # Backend Python Core Package
    ├── api/
    │   └── main.py                       # FastAPI application with REST endpoints & lifespan
    ├── data/
    │   ├── db.py & init_db.py            # SQLite database schema and initialization
    │   ├── download_data.py              # Download handlers for MIND and SemEval datasets
    │   └── generate_synthetic_data.py    # Synthetic test datasets and user profiles
    ├── graph/
    │   ├── build_graph.py                # Constructs news-entity heterogeneous graph
    │   └── graph_utils.py                # Graph feature extractors and node mapping
    ├── models/
    │   ├── gnn_encoder.py                # 2-layer Heterogeneous GraphSAGE encoder (PyG)
    │   ├── persuasion_detector.py        # RoBERTa multi-label classifier (23 techniques)
    │   └── persuasion_profile.py         # EMA reader susceptibility vector tracking
    └── ranking/
        ├── reranker.py                   # 2-stage retrieval + responsible re-ranking engine
        ├── responsible_score.py          # RNS formulation and provider diversity capping
        └── vector_index.py               # FAISS vector similarity indexing
```
