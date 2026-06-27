# IBM watsonx Finance Agent — AI-Powered Investment Recommendation System

<div align="center">

```
╔══════════════════════════════════════════════════════════════════╗
║   IBM watsonx.ai · Finance Agent · Multi-Agent Orchestration    ║
║   Powered by IBM Granite-4.0-8B-Instruct · IBM BOB Platform     ║
╚══════════════════════════════════════════════════════════════════╝
```

**4 Specialist AI Agents · RAG Pipeline · Real-time Dashboard · IBM Cloud Ready**

</div>

---

## 📐 System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         IBM BOB Orchestration Platform                  │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                     Agent Orchestrator                           │   │
│  │            IBM Granite-4.0-8B-Instruct (Synthesis)              │   │
│  └──────┬──────────────────────────┬───────────────────────────────┘   │
│         │                          │                                     │
│    Stage 1                    Stage 2 ─────────┐                       │
│         │                          │            │                       │
│  ┌──────▼──────────┐   ┌───────────▼────────┐  │                       │
│  │  Investment     │   │  Portfolio         │  │                       │
│  │  Knowledge      │──▶│  Recommendation    │  │                       │
│  │  Agent (RAG)    │   │  Agent             │  │                       │
│  └─────────────────┘   └───────────┬────────┘  │                       │
│                                    │            │                       │
│                              Stage 3 ──────────┘                       │
│                                    │                                     │
│                    ┌───────────────┴──────────────────┐                │
│                    │                                  │                 │
│           ┌────────▼──────┐               ┌──────────▼───────┐        │
│           │  Risk         │               │  Market          │        │
│           │  Analysis     │               │  Monitoring      │        │
│           │  Agent        │               │  Agent           │        │
│           └────────┬──────┘               └──────────┬───────┘        │
│                    │                                  │                 │
│                    └──────────────┬───────────────────┘                │
│                                   │                                     │
│                          ┌────────▼────────┐                           │
│                          │  Final Synthesis │                           │
│                          │  + Dashboard     │                           │
│                          └─────────────────┘                           │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐    ┌──────────────────┐    ┌────────────────────────┐
│  Vector Index   │    │  File Component  │    │  IBM watsonx.ai        │
│  ChromaDB       │    │  IBM COS / Local │    │  Granite-4.0-8B        │
│  RAG Pipeline   │    │  Portfolios      │    │  Foundation Model      │
│  Embeddings     │    │  Reports/Docs    │    │  Agent Execution       │
└─────────────────┘    └──────────────────┘    └────────────────────────┘
```

---

## 🤖 Multi-Agent System

### Agent 1 — Investment Knowledge Agent
Retrieves and synthesises financial information using RAG over uploaded documents, company reports, and market news.
- **Tools**: Vector search, document retrieval, news fetch
- **RAG**: Enabled — queries the ChromaDB vector index
- **Output**: Knowledge summary, key insights, recommended sectors, risk considerations

### Agent 2 — Portfolio Recommendation Agent
Generates personalised investment recommendations and asset allocation.
- **Inputs**: User goals, budget, risk tolerance, investment horizon, knowledge context
- **Model**: Modern Portfolio Theory + Granite LLM synthesis
- **Output**: Asset allocation, specific instruments, expected returns, projected value, top picks

### Agent 3 — Risk Analysis Agent
Evaluates portfolio risk with quantitative metrics and mitigation strategies.
- **Metrics**: Beta, volatility, Sharpe ratio, max drawdown, diversification score
- **Output**: Risk score (0–10), concentration warnings, stress scenarios, mitigation strategies

### Agent 4 — Market Monitoring Agent
Monitors financial news, stock trends, and fires real-time investment alerts.
- **Data**: Market indices, sector performance, macro indicators
- **Output**: Market sentiment, sector heatmap, investment alerts (INFO/WARNING/CRITICAL)

---

## 🗂️ Project Structure

```
Finance_Agent/
│
├── agents/                          # Agent modules
│   ├── __init__.py
│   ├── base_agent.py                # Abstract base class
│   ├── watsonx_client.py            # IBM Granite-4.0 client
│   ├── models.py                    # Pydantic data models
│   ├── orchestrator.py              # IBM BOB orchestrator + memory
│   ├── investment_knowledge_agent.py
│   ├── portfolio_recommendation_agent.py
│   ├── risk_analysis_agent.py
│   └── market_monitoring_agent.py
│
├── rag/                             # RAG pipeline
│   ├── __init__.py
│   └── pipeline.py                  # VectorStore, RAGRetriever, Ingestion
│
├── storage/                         # File component
│   ├── __init__.py
│   └── file_component.py            # IBM COS / local file storage
│
├── api/                             # FastAPI backend
│   ├── __init__.py
│   └── main.py                      # REST API + WebSocket endpoints
│
├── config/                          # Configuration
│   ├── __init__.py
│   ├── settings.py                  # Pydantic settings (all env vars)
│   └── bob_workflow.py              # IBM BOB workflow specification
│
├── data/
│   ├── knowledge/                   # Pre-loaded financial knowledge
│   │   └── seed_knowledge.py
│   ├── vector_store/                # ChromaDB persistent store
│   ├── uploads/                     # User document uploads
│   └── reports/                     # Generated reports
│
├── frontend/
│   └── index.html                   # Single-page dashboard + chat UI
│
├── deploy/
│   └── ibm_cloud_deploy.py          # IBM Code Engine deployment script
│
├── tests/
│   └── test_agents.py               # Pytest test suite
│
├── main.py                          # Application entry point
├── requirements.txt                 # Python dependencies
├── Dockerfile                       # Container image
├── docker-compose.yml               # Local development stack
├── manifest.yml                     # IBM Cloud Foundry manifest
└── .env.example                     # Environment variable template
```

---

## ⚡ Quick Start

### 1. Clone and set up environment

```bash
git clone <repository-url>
cd Finance_Agent
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```env
# Required — IBM watsonx.ai
WATSONX_API_KEY=your_ibm_cloud_api_key
WATSONX_PROJECT_ID=your_watsonx_project_id
WATSONX_URL=https://us-south.ml.cloud.ibm.com

# Required — App security
SECRET_KEY=generate-a-secure-random-string

# Optional — IBM COS (falls back to local disk)
IBM_COS_API_KEY=your_cos_api_key
IBM_COS_INSTANCE_CRN=your_cos_crn
IBM_COS_BUCKET_NAME=finance-agent-files

# Optional — Market data APIs
ALPHA_VANTAGE_API_KEY=your_key
NEWS_API_KEY=your_key
FINNHUB_API_KEY=your_key
```

### 3. Run the application

```bash
python main.py
# Or with hot reload for development:
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Access the dashboard

Open your browser: **http://localhost:8000** (if static files are served) or open `frontend/index.html` directly.

---

## 🐳 Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or build and run standalone
docker build -t finance-agent .
docker run -p 8000:8000 --env-file .env finance-agent
```

---

## ☁️ IBM Cloud Deployment

### IBM Code Engine (Recommended)

```bash
# 1. Install IBM Cloud CLI
curl -fsSL https://clis.cloud.ibm.com/install/linux | sh
ibmcloud plugin install code-engine

# 2. Login
ibmcloud login --apikey $IBM_CLOUD_API_KEY -r us-south

# 3. Build and push image to IBM Container Registry
ibmcloud cr login
docker build -t us.icr.io/$NAMESPACE/finance-agent:latest .
docker push us.icr.io/$NAMESPACE/finance-agent:latest

# 4. Create Code Engine project and deploy
ibmcloud ce project create --name finance-agent-project
ibmcloud ce application create \
  --name finance-agent \
  --image us.icr.io/$NAMESPACE/finance-agent:latest \
  --cpu 2 --memory 4G \
  --min-scale 1 --max-scale 5 \
  --port 8000 \
  --env-from-secret finance-agent-secrets

# 5. Get URL
ibmcloud ce application get --name finance-agent --output url
```

### IBM Cloud Foundry

```bash
ibmcloud cf push finance-agent -f manifest.yml
```

---

## 🔌 API Reference

### Chat — Full Multi-Agent Pipeline

```http
POST /api/chat
Content-Type: application/json

{
  "message": "Build me a diversified portfolio for retirement",
  "user_id": "user_001",
  "investment_goal": "Retirement Planning",
  "budget": 100000,
  "risk_level": "Medium",
  "investment_horizon_years": 20,
  "monthly_contribution": 1000,
  "age": 40,
  "existing_portfolio": [
    {
      "symbol": "SPY",
      "name": "S&P 500 ETF",
      "asset_class": "Equity",
      "quantity": 10,
      "purchase_price": 450.0,
      "current_price": 520.0
    }
  ]
}
```

**Response includes**:
- `message` — Full AI narrative recommendation
- `recommendation` — Allocations, top picks, projected value, expected return
- `risk_metrics` — Beta, volatility, Sharpe, diversification score, stress scenarios
- `market_insight` — Market sentiment, alerts, sector performance
- `sources` — Knowledge sources used by RAG

### Document Upload (RAG Ingestion)

```http
POST /api/upload?user_id=user_001
Content-Type: multipart/form-data
file: [financial_report.pdf]
```

### Portfolio Management

```http
POST /api/portfolio/save          # Save portfolio
GET  /api/portfolio/{user_id}     # Load portfolio
GET  /api/portfolio/{user_id}/files  # List portfolio files
```

### Market Data

```http
GET /api/market/snapshot          # Live market snapshot
GET /api/rag/stats                # Vector store statistics
GET /health                        # Service health check
```

### WebSocket — Streaming Chat

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/chat/{session_id}');
ws.send(JSON.stringify({
  message: "What stocks should I buy?",
  budget: 50000,
  risk_level: "Medium",
  investment_horizon_years: 10
}));

// Receive events:
// {"event": "progress", "stage": "knowledge_agent", "message": "Searching..."}
// {"event": "progress", "stage": "portfolio_agent", ...}
// {"event": "complete", "message": "...", "recommendation": {...}}
```

---

## 🧠 RAG Pipeline

The Retrieval-Augmented Generation pipeline processes financial documents:

```
Document Upload (PDF/DOCX/TXT/CSV/XLSX)
           ↓
    Text Extraction
           ↓
    Chunking (512 chars, 64 overlap)
           ↓
    Embedding (all-MiniLM-L6-v2)
           ↓
    ChromaDB Vector Storage
           ↓
    Semantic Search (cosine similarity)
           ↓
    Context Assembly → Granite Prompt
           ↓
    Grounded AI Response
```

**Supported document types**: PDF, DOCX, TXT, CSV, XLSX, Markdown

---

## 🧪 Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run all tests
pytest tests/ -v

# Run specific test class
pytest tests/test_agents.py::TestRiskAnalysisAgent -v

# Run with coverage
pytest tests/ --cov=agents --cov=rag --cov=storage --cov-report=term-missing
```

---

## 🔒 Security

- **JWT Authentication**: Stateless token-based auth
- **Secret Management**: IBM Cloud Secrets Manager / environment variables
- **Encryption at Rest**: IBM COS with AES-256
- **Encryption in Transit**: TLS 1.3 for all connections
- **Input Validation**: Pydantic strict schema validation on all inputs
- **Audit Logging**: Structured logging for all agent executions
- **Non-root Container**: Docker image runs as non-privileged user

---

## 📊 Dashboard Features

| Feature | Description |
|---------|-------------|
| Portfolio Allocation | Doughnut chart with asset class breakdown |
| Investment Growth | Multi-year projection with DCA contributions |
| Risk Metrics | Radar chart (beta, volatility, Sharpe, drawdown) |
| Market Heatmap | Sector performance colour-coded grid |
| Risk Score | Dynamic gauge needle (0–10) |
| Stress Scenarios | Bear / Base / Bull case projections |
| Market Alerts | Real-time INFO / WARNING / CRITICAL alerts |
| Market Indices | Bar chart of major index changes |
| Top Investment Picks | AI-selected instruments with rationale |
| Knowledge Sources | RAG document citations |

---

## 🏗️ IBM BOB Workflow

The `config/bob_workflow.py` defines the complete IBM BOB multi-agent topology:

- **5 agents**: Orchestrator + 4 specialists
- **Data flows**: Explicit input/output mappings between agents
- **Tools**: RAG search, market data, risk computation, alerting
- **Memory**: Session-scoped conversational memory
- **Execution**: Sequential pipeline with parallel-safe stages
- **Deployment**: IBM Code Engine with autoscaling

---

## 📋 Environment Variables Reference

| Variable | Required | Description |
|----------|----------|-------------|
| `WATSONX_API_KEY` | ✅ | IBM Cloud API key |
| `WATSONX_PROJECT_ID` | ✅ | watsonx.ai project ID |
| `WATSONX_URL` | ✅ | watsonx.ai endpoint URL |
| `SECRET_KEY` | ✅ | App security secret key |
| `GRANITE_MODEL_ID` | | Default: `ibm/granite-4-8b-instruct` |
| `IBM_COS_API_KEY` | | IBM Cloud Object Storage key |
| `IBM_COS_INSTANCE_CRN` | | COS instance CRN |
| `VECTOR_DB_PATH` | | ChromaDB path (default: `./data/vector_store`) |
| `EMBEDDING_MODEL` | | Sentence transformer (default: `all-MiniLM-L6-v2`) |
| `REDIS_URL` | | Redis for caching (default: `redis://localhost:6379`) |
| `NEWS_API_KEY` | | NewsAPI.org key for market news |
| `FINNHUB_API_KEY` | | Finnhub key for real-time market data |
| `ALPHA_VANTAGE_API_KEY` | | Alpha Vantage key for financial data |
| `DEBUG` | | Enable debug logging (default: `false`) |

---

## 📜 Disclaimer

This system is for **educational and informational purposes only**. It does not constitute financial, investment, tax, or legal advice. All investment decisions should be made in consultation with a licensed financial advisor. Past performance does not guarantee future results. Investments carry risk including possible loss of principal.

---

<div align="center">

Built with **IBM BOB** · **IBM watsonx.ai** · **IBM Granite-4.0-8B-Instruct** · **IBM Cloud**

</div>
