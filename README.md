# Intelligent-Recruitment-Assistant-Using-Retrieval-Augmented-Generation

 **MSc Data Science Portfolio Project**  
> An end-to-end AI system leveraging Retrieval-Augmented Generation (RAG), vector similarity search, structured NLP skill extraction, and hybrid ranking algorithms to automate resume screening, candidate-job matching, and recruiter decision support.

---

## 📌 Executive Summary

Recruiters spend countless hours manually evaluating hundreds of resumes per job opening. Traditional Applicant Tracking Systems (ATS) rely on brittle keyword search, frequently missing highly qualified candidates due to vocabulary mismatches (e.g., "Deep Learning" vs. "Neural Networks").

This project implements an **Intelligent Recruitment Assistant** that combines **Retrieval-Augmented Generation (RAG)** with **Dense Semantic Embeddings (ChromaDB)**, **Structured NLP Skill Extraction**, and a mathematical **Candidate Suitability Index (CSI)**. 

### Key Capabilities
1. **Multi-Format Resume Ingestion**: Extracts text and metadata from PDF, DOCX, and TXT resumes.
2. **Dense Vector Embeddings**: Embeds resume text into 384-dimensional dense vectors using `SentenceTransformers` (`all-MiniLM-L6-v2`) or Google Gemini / OpenAI embeddings.
3. **Hybrid Candidate Suitability Index (CSI)**: Blends vector cosine similarity, Jaccard skill overlap ratio, and experience alignment metrics.
4. **Explainable AI Matching**: Provides detailed rationale for why a candidate is a strong fit, highlighting matched skills and gap areas.
5. **Side-by-Side Candidate Comparison**: Comparative analysis of candidates for specific job descriptions.
6. **Conversational Recruiter Chatbot**: Natural language Q&A interface for queries such as *"Find candidates with Python, SQL, and AWS"* or *"Compare Candidate A and Candidate B"*.
7. **Interactive Streamlit UI & FastAPI REST API**: Production-ready microservice backend and executive recruiter dashboard.

---

## 📐 Mathematical & Algorithmic Formulation

### 1. Candidate Suitability Index (CSI)
The composite suitability score $\text{CSI} \in [0, 100\%]$ for candidate $C$ given Job Description $J$ is calculated as:

$$\text{CSI}(C, J) = 100 \times \left( w_1 \cdot \text{Sim}_{\text{semantic}}(V_C, V_J) + w_2 \cdot \text{Match}_{\text{skill}}(\mathcal{S}_C, \mathcal{S}_J) + w_3 \cdot \text{Score}_{\text{exp}}(E_C, E_J) \right)$$

Where default weights are $w_1 = 0.50$, $w_2 = 0.35$, $w_3 = 0.15$ subject to $\sum w_i = 1.0$.

### 2. Dense Semantic Vector Similarity
Using normalized embedding vectors $V_C, V_J \in \mathbb{R}^d$:

$$\text{Sim}_{\text{semantic}}(V_C, V_J) = \frac{V_C \cdot V_J}{\|V_C\|_2 \|V_J\|_2}$$

### 3. Jaccard Skill Overlap Score
Given extracted technical skill sets $\mathcal{S}_C$ (candidate) and $\mathcal{S}_J$ (job description):

$$\text{Match}_{\text{skill}}(\mathcal{S}_C, \mathcal{S}_J) = \frac{|\mathcal{S}_C \cap \mathcal{S}_J|}{|\mathcal{S}_J|}$$

### 4. Normalized Experience Score

$$\text{Score}_{\text{exp}}(E_C, E_J) = \min\left(1.0, \frac{E_C}{\max(1, E_J)}\right)$$

---

## 🏗️ System Architecture

```
                                  +---------------------------------------+
                                  |    Recruiter Web Dashboard            |
                                  |         (Streamlit UI)                |
                                  +-------------------+-------------------+
                                                      |
                                                      v
                                  +-------------------+-------------------+
                                  |       FastAPI REST Service            |
                                  +---------+-----------------+-----------+
                                            |                 |
                     +----------------------+                 +----------------------+
                     |                                                               |
                     v                                                               v
+--------------------+-------------------+                         +-----------------+-------------------+
|      Document Processing Engine        |                         |         RAG & Reasoning Engine      |
|  - PyPDF / python-docx Parsing         |                         |  - LangChain RAG Orchestration    |
|  - 150+ Skill Taxonomy Extraction      |                         |  - Explainable Match Rationale    |
|  - Experience & Education Parsing      |                         |  - Candidate Comparison Agent     |
+--------------------+-------------------+                         +-----------------+-------------------+
                     |                                                               |
                     v                                                               v
+--------------------+-------------------+                         +-----------------+-------------------+
|    Vector Store & Embedding Manager    |                         |   Candidate Suitability Index     |
|  - ChromaDB Vector Store Persistence   |                         |   - Hybrid CSI Scoring Ranker     |
|  - SentenceTransformers (384-d)        |                         |   - Jaccard Skill Matcher         |
+----------------------------------------+                         +-----------------------------------+
```

---

## 📁 Repository Structure

```
recruitment_rag/
├── app/
│   ├── __init__.py
│   ├── config.py                 # System configurations & settings
│   ├── core/
│   │   ├── __init__.py
│   │   ├── document_loader.py    # PDF/DOCX parser & skill taxonomy extractor
│   │   ├── embeddings.py         # SentenceTransformers / Gemini / OpenAI wrapper
│   │   ├── vector_store.py       # ChromaDB persistent vector manager
│   │   ├── ranker.py             # Candidate Suitability Index (CSI) hybrid scoring
│   │   └── rag_engine.py         # LangChain RAG pipeline (QA, explanation, comparison)
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py               # FastAPI backend application entry point
│   │   └── routes/
│   │       ├── resumes.py        # Upload, list, retrieve, delete resume endpoints
│   │       ├── matching.py       # Candidate ranking & comparative analysis routes
│   │       └── chat.py           # Recruiter Q&A chatbot route
│   └── dashboard/
│       └── app.py                # Interactive Streamlit Web Dashboard
├── data/
│   ├── raw_resumes/              # Ingested PDF/DOCX/TXT resume files
│   ├── job_descriptions/         # Sample Job Descriptions
│   └── chroma_db/                # ChromaDB vector store directory
├── tests/
│   ├── test_loader.py            # Unit tests for text & skill parsing
│   ├── test_vector_store.py      # Unit tests for ChromaDB indexing & vector search
│   ├── test_ranker.py            # Unit tests for CSI hybrid scoring formulas
│   └── test_api.py               # Integration tests for FastAPI REST endpoints
├── scripts/
│   ├── generate_sample_data.py   # Dataset generator (creates sample resumes & indexes them)
│   ├── evaluate_rag.py           # RAG retrieval metrics evaluation (Precision@K, MRR, Hit Rate)
│   └── deploy_aws.sh             # AWS EC2 & S3 deployment shell script
├── Dockerfile                    # Containerization build setup
├── docker-compose.yml            # Multi-container orchestration (FastAPI + Streamlit)
├── requirements.txt              # Project dependencies
├── .env.example                  # Environment configuration template
└── README.md                     # Comprehensive project documentation
```

---

## 🛠️ Step-by-Step Execution & Usage Guide

### 1. Installation & Environment Setup
Clone the repository and install requirements:

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Sample Resumes & Populate Vector Database
Run the pre-built dataset generator to automatically create 5 realistic candidate resumes across Data Science, NLP, DevOps, Full Stack, and Machine Learning engineering roles, and index them into ChromaDB:

```bash
python scripts/generate_sample_data.py
```

### 3. Run Unit Tests & Verify Setup
Validate all core modules, parsers, vector store operations, ranking formulas, and FastAPI routes:

```bash
pytest tests/
```

### 4. Run RAG Retrieval Evaluation Script
Evaluate system performance metrics (Hit Rate@K, Precision@K, Mean Reciprocal Rank):

```bash
python scripts/evaluate_rag.py
```

### 5. Launch FastAPI REST Backend
Start the FastAPI API service:

```bash
uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload
```
- Interactive API Docs (Swagger UI): `http://localhost:8000/docs`
- Health Check: `http://localhost:8000/health`

### 6. Launch Streamlit Recruiter Dashboard
Open a separate terminal window and launch the Streamlit interface:

```bash
streamlit run app/dashboard/app.py
```
- Dashboard Access: `http://localhost:8501`

---

## 💬 Answering Recruiter Natural Language Queries

The system addresses the 4 key recruiter questions:

### 1. "Find candidates with Python, SQL, and AWS experience."
- **Retrieved Top Candidate**: **Alex Morgan** (5 yrs exp, Master's Degree) & **Marcus Vance** (6 yrs exp).
- **Extracted Skills**: `python`, `sql`, `aws`, `fastapi`, `docker`.
- **Match Rationale**: Both candidates possess hands-on AWS infrastructure deployment experience with Python REST microservices.

### 2. "Which applicants have experience in Machine Learning and NLP?"
- **Retrieved Top Candidates**: **Dr. Priya Sharma** (PhD in NLP, 4 yrs exp) & **Alex Morgan** (Master's in Data Science, 5 yrs exp).
- **Extracted Skills**: `nlp`, `machine learning`, `transformers`, `huggingface`, `pytorch`, `langchain`, `rag`.

### 3. "Why is Candidate A suitable for this Data Scientist role?"
- **Candidate Evaluated**: Alex Morgan.
- **Suitability Breakdown**:
  - **CSI Overall Score**: `92.4%`
  - **Matched Skills**: `python`, `sql`, `machine learning`, `deep learning`, `pytorch`, `langchain`, `rag`, `fastapi`, `aws`.
  - **Strengths**: Extensive hands-on experience building RAG architectures and vector search applications.

### 4. "Compare Candidate A and Candidate B."
- **Side-by-Side Comparison**: Side-by-side metric matrix comparing Alex Morgan vs. Dr. Priya Sharma in technical specializations (Production RAG & MLOps vs. Academic Transformers & NLP Research).

---

## ☁️ AWS Cloud & Docker Deployment

### Docker Containerization
To launch both FastAPI backend and Streamlit dashboard using Docker Compose:

```bash
docker-compose up -d --build
```

### AWS EC2 & S3 Setup
Run the automated deployment script on an AWS EC2 instance (Ubuntu 22.04 LTS):

```bash
chmod +x scripts/deploy_aws.sh
./scripts/deploy_aws.sh
```

---

## 🎓 MSc Data Science Project Information

- **Degree**: Master of Science in Data Science
- **Core Methodology**: Retrieval-Augmented Generation (RAG), Hybrid Rank Fusion, Vector Embeddings (ChromaDB), Fast REST APIs, Streamlit AI UI.
- **License**: MIT
