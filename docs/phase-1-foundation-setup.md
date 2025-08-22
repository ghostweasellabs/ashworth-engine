# Phase 1: Foundation & Environment Setup

## Duration: 2 days
## Goal: Establish project infrastructure and confirm requirements

### 1.1 Project Structure & Repository Setup

**Create complete folder structure:**
```
ashworth-engine/
├── src/
│   ├── agents/                    # LangGraph agent implementations
│   │   ├── __init__.py
│   │   ├── data_fetcher.py       # @task decorated functions
│   │   ├── data_cleaner.py       # Data cleaning and standardization
│   │   ├── data_processor.py
│   │   ├── tax_categorizer.py
│   │   ├── report_generator.py
│   │   └── orchestrator.py
│   ├── workflows/                 # StateGraph definitions
│   │   ├── __init__.py
│   │   ├── financial_analysis.py  # Main workflow definition
│   │   └── state_schemas.py      # TypedDict state definitions
│   ├── utils/                     # Utility functions and helpers
│   │   ├── __init__.py
│   │   ├── file_processing.py
│   │   ├── data_validation.py
│   │   ├── decimal_utils.py
│   │   ├── llm_integration.py
│   │   ├── supabase_client.py    # Supabase integration
│   │   └── vector_operations.py  # RAG operations
│   ├── config/                    # Configuration management
│   │   ├── __init__.py
│   │   ├── settings.py
│   │   ├── prompts.py
│   │   └── personas.py
│   ├── api/                       # FastAPI endpoints
│   │   ├── __init__.py
│   │   ├── routes.py
│   │   └── dependencies.py
│   ├── stores/                    # Memory store implementations
│   │   ├── __init__.py
│   │   ├── checkpointers.py
│   │   └── memory_stores.py
│   └── tests/                     # Test files co-located
│       ├── __init__.py
│       ├── test_agents/
│       ├── test_workflows/
│       └── test_utils/
├── supabase/                      # Supabase configuration
│   ├── config.toml               # Supabase CLI config
│   ├── migrations/               # Database migrations
│   │   ├── 001_setup_extensions.sql
│   │   ├── 002_setup_vector_tables.sql
│   │   ├── 003_setup_analytics_tables.sql
│   │   ├── 004_setup_storage.sql
│   │   ├── 005_setup_rls_policies.sql
│   │   └── 006_setup_data_cleaning.sql  # Data cleaning logs
│   ├── functions/                # Edge functions (if needed)
│   └── seed.sql                  # Initial data seeding
├── volumes/                       # Docker volumes configuration
│   ├── api/
│   ├── db/
│   ├── storage/
│   └── logs/
├── langgraph.json                 # LangGraph deployment configuration
├── pyproject.toml                # Python project and dependencies
├── uv.lock                       # Dependency lock file
├── Dockerfile                    # Container definition
├── docker-compose.yml            # Local Supabase stack
├── .env.example                  # Environment variables template
└── README.md                     # Project documentation
```

### 1.2 Development Environment Setup

**Install uv and setup Python environment:**

```bash
# Install uv (modern Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
# Or on macOS: brew install uv
# Or on Windows: powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Initialize Python project with uv
cd ashworth-engine
uv init .

# Setup virtual environment with Python 3.10+
uv venv

# Activate the virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Add core LangGraph dependencies
uv add langgraph
uv add langchain
uv add langchain-openai
uv add langchain-anthropic

# Add FastAPI and API dependencies
uv add fastapi
uv add "uvicorn[standard]"
uv add pydantic
uv add python-multipart

# Add data processing libraries
uv add pandas
uv add numpy
uv add openpyxl

# Add document processing
uv add pypdf2
uv add pdfplumber
uv add pytesseract
uv add pillow

# Add chart generation
uv add pyecharts
uv add seaborn
uv add plotly

# Add Supabase and vector database integration
uv add supabase
uv add vecs
uv add psycopg2-binary
uv add sqlalchemy

# Add PDF generation
uv add weasyprint
uv add markdown

# Add development tools
uv add --dev pytest
uv add --dev pytest-cov
uv add --dev pytest-asyncio
uv add --dev black
uv add --dev isort
uv add --dev mypy

# Add additional utilities
uv add requests
uv add aiofiles
uv add python-dotenv
uv add structlog
```

**Update pyproject.toml with installed versions:**

After adding all dependencies, update the `pyproject.toml` file to lock the specific versions that were installed:

```bash
# View currently installed versions
uv pip list

# Update pyproject.toml to pin specific versions (example)
# Edit pyproject.toml to specify exact versions from uv pip list output
```

**Note**: The `pyproject.toml` file will be automatically created by uv and should be manually updated with specific version constraints after installation to ensure reproducible builds.

Create `langgraph.json`:

**Configure uv project (`pyproject.toml`):**

Ensure the `pyproject.toml` file contains proper project configuration:

```toml
[project]
name = "ashworth-engine"
version = "2.0.0"
description = "AI-powered financial intelligence platform with Supabase backend"
readme = "README.md"
requires-python = ">=3.10"
classifiers = [
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]

dependencies = [
    # Core dependencies will be populated by uv add commands
    # Pin specific versions after installation
]

[project.optional-dependencies]
dev = [
    "pytest",
    "pytest-cov", 
    "pytest-asyncio",
    "black",
    "isort",
    "mypy",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.uv]
dev-dependencies = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "pytest-asyncio>=0.21.0",
    "black>=23.0.0",
    "isort>=5.12.0",
    "mypy>=1.7.0",
]

[tool.black]
line-length = 88
target-version = ['py310']

[tool.isort]
profile = "black"

[tool.pytest.ini_options]
testpaths = ["src/tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = "-v --cov=src --cov-report=html --cov-report=term"
```
```json
{
  "dependencies": ["./pyproject.toml"],
  "graphs": {
    "financial_analysis": "./src/workflows/financial_analysis.py:app"
  },
  "env": ".env",
  "python_executable": "./.venv/bin/python"
}
```

Verify LangGraph setup:
- Test `uv run langgraph dev` command
- Verify LangGraph Studio connectivity
- Ensure virtual environment is properly recognized

### 1.4 Supabase Project Initialization

**Install and setup Supabase CLI using uv:**
```bash
# Install Supabase CLI globally (npm still required for Supabase CLI)
npm install -g supabase

# Login to Supabase (for remote project access)
supabase login

# Initialize Supabase project
supabase init

# Start local Supabase stack with PostgreSQL + pgvector
supabase start
```

**Configure Supabase project (`supabase/config.toml`):**
```toml
project_id = "ashworth-engine"

[api]
port = 54321
schemas = ["public", "storage", "graphql_public"]
extra_search_path = ["public", "extensions"]
max_rows = 1000

[db]
port = 54322
shadow_port = 54320
major_version = 15

[studio]
port = 54323

[inbucket]
port = 54324
smtp_port = 54325
pop3_port = 54326

[storage]
file_size_limit = "50MiB"

[auth]
site_url = "http://localhost:3000"
additional_redirect_urls = ["https://localhost:3000"]
jwt_expiry = 3600
enable_signup = true
enable_email_confirmations = false

[edge_runtime]
port = 54321
```

### 1.5 Environment Variables Template

Create `.env.example`:
```
# LLM Configuration
OPENAI_API_KEY=sk-your-key-here
ANTHROPIC_API_KEY=your-key-here
LLM_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-ada-002

# Local Supabase Configuration (default for development)
SUPABASE_URL=http://127.0.0.1:54321
SUPABASE_ANON_KEY=your-local-anon-key
SUPABASE_SERVICE_KEY=your-local-service-key
SUPABASE_JWT_SECRET=your-local-jwt-secret

# Local Database Configuration (PostgreSQL with pgvector)
DATABASE_URL=postgresql://postgres:postgres@127.0.0.1:54322/postgres
VECS_CONNECTION_STRING=postgresql://postgres:postgres@127.0.0.1:54322/postgres

# Local Storage Configuration
STORAGE_PROVIDER=supabase
STORAGE_BUCKET=reports
CHARTS_BUCKET=charts
SUPABASE_STORAGE_URL=http://127.0.0.1:54321/storage/v1

# Vector Database Configuration
VECTOR_COLLECTION_NAME=documents
VECTOR_DIMENSION=1536
VECTOR_SIMILARITY_THRESHOLD=0.8

# API Configuration
API_AUTH_KEY=your-auth-key
AE_ENV=development

# Processing Configuration
MAX_UPLOAD_SIZE=52428800
REPORT_RETENTION_DAYS=90
OCR_LANGUAGE=eng

# Performance
MAX_CONCURRENT_REQUESTS=5
LLM_TIMEOUT_SECONDS=300

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json

# Production Supabase (uncomment when deploying to production)
# SUPABASE_URL=https://your-project-ref.supabase.co
# DATABASE_URL=postgresql://postgres.[PROJECT_REF]:[PASSWORD]@aws-0-[REGION].pooler.supabase.com:6543/postgres
# VECS_CONNECTION_STRING=postgresql://postgres.[PROJECT_REF]:[PASSWORD]@aws-0-[REGION].pooler.supabase.com:5432/postgres
```

### 1.6 Core Data Models Definition

Create `src/workflows/state_schemas.py`:
```python
from typing import Annotated, List, Dict, Any, Optional
from typing_extensions import TypedDict
from decimal import Decimal
from datetime import datetime
from pydantic import BaseModel
import operator

class Transaction(BaseModel):
    date: str  # ISO format
    description: str
    amount: Decimal
    category: Optional[str] = None
    tax_category: Optional[str] = None
    is_deductible: Optional[bool] = None
    account: Optional[str] = None
    currency: str = "USD"
    metadata: Optional[Dict[str, Any]] = None  # Additional context for LLM processing
    
class FinancialMetrics(BaseModel):
    total_revenue: Decimal
    total_expenses: Decimal
    gross_profit: Decimal
    gross_margin_pct: float
    cash_balance: Optional[Decimal] = None
    expense_by_category: Dict[str, Decimal]
    anomalies: List[Transaction]
    pattern_matches: Dict[str, int]
    detected_business_types: List[str]
    
class TaxSummary(BaseModel):
    deductible_total: Decimal
    non_deductible_total: Decimal
    potential_savings: Decimal
    flags: List[str]
    categorization_accuracy: float = 100.0
    
class OverallState(TypedDict):
    # Input data
    client_id: str
    analysis_type: str
    file_content: bytes
    
    # Processing state
    raw_extracted_data: Annotated[List[Dict[str, Any]], operator.add]  # Raw OCR/extraction output
    transactions: Annotated[List[Transaction], operator.add]  # Cleaned and structured data
    financial_metrics: Optional[FinancialMetrics]
    tax_summary: Optional[TaxSummary]
    
    # Data quality tracking
    data_quality_score: Optional[float]
    cleaning_summary: Optional[Dict[str, Any]]
    
    # Output state
    final_report_md: Optional[str]
    final_report_pdf_path: Optional[str]
    charts: List[str]
    
    # Error handling
    error_messages: Annotated[List[str], operator.add]
    warnings: Annotated[List[str], operator.add]
    
    # Metadata
    workflow_phase: str
    processing_start_time: datetime
    processing_end_time: Optional[datetime]
    trace_id: str

class InputState(TypedDict):
    client_id: str
    analysis_type: str
    file_content: bytes

class OutputState(TypedDict):
    final_report_md: str
    final_report_pdf_path: str
    charts: List[str]
    error_messages: List[str]
```

### 1.7 Persona Configuration

Create `src/config/personas.py`:
```python
PERSONAS = {
    "orchestrator": {
        "name": "Dr. Victoria Ashworth",
        "title": "Chief Financial Operations Orchestrator",
        "achievement": "coordinated analysis with {completion_rate}% on-time completion",
        "expertise": "Strategic oversight and quality assurance"
    },
    "data_cleaner": {
        "name": "Alexandra Sterling",
        "title": "Chief Data Transformation Specialist",
        "achievement": "achieved {data_quality_score}% data quality with precision cleaning",
        "expertise": "Data standardization, quality assurance, and LLM optimization"
    },
    "data_processor": {
        "name": "Dexter Blackwood", 
        "title": "Quantitative Data Integrity Analyst",
        "achievement": "delivered {data_validation_accuracy}% validation accuracy",
        "expertise": "Data quality and financial calculations"
    },
    "categorizer": {
        "name": "Clarke Pemberton",
        "title": "Corporate Tax Compliance Strategist", 
        "achievement": "achieved {tax_categorization_accuracy}% error-free categorization",
        "expertise": "Tax compliance and optimization"
    },
    "data_fetcher": {
        "name": "Dr. Marcus Thornfield",
        "title": "Senior Market Intelligence Analyst",
        "achievement": "comprehensive data collection in record time",
        "expertise": "Multi-format data extraction and market context"
    },
    "report_generator": {
        "name": "Professor Elena Castellanos", 
        "title": "Executive Financial Storytelling Director",
        "achievement": "synthesized insights into compelling strategy",
        "expertise": "C-suite narrative and strategic recommendations"
    }
}

def fill_persona_placeholders(state: dict) -> dict:
    """Fill persona achievement placeholders with actual metrics"""
    return {
        "orchestrator_achievement": PERSONAS["orchestrator"]["achievement"].format(
            completion_rate=99.9
        ),
        "data_fetcher_achievement": PERSONAS["data_fetcher"]["achievement"],
        "data_cleaner_achievement": PERSONAS["data_cleaner"]["achievement"].format(
            data_quality_score=state.get("data_quality_score", 98.5)
        ),
        "data_processor_achievement": PERSONAS["data_processor"]["achievement"].format(
            data_validation_accuracy=state.get("data_validation_accuracy", 99.99)
        ),
        "tax_categorizer_achievement": PERSONAS["categorizer"]["achievement"].format(
            tax_categorization_accuracy=state.get("tax_categorization_accuracy", 100.0)
        ),
        "report_generator_achievement": PERSONAS["report_generator"]["achievement"]
    }
```

### 1.8 Architecture Decision Records Setup

Create `docs/adr/` directory with initial ADRs:

**ADR-001: Adopt LangGraph for Orchestration**
- Decision: Use LangGraph's StateGraph for agent workflow management
- Rationale: Provides robust stateful multi-agent orchestration

**ADR-002: Use uv for Python Dependency Management**
- Decision: Adopt uv as the exclusive Python package manager
- Rationale: Faster installs, better dependency resolution, modern tooling
- Impact: Replaces pip/requirements.txt with pyproject.toml and uv.lock

**ADR-003: Supabase as Backend Infrastructure**
- Decision: Use Supabase for database, storage, auth, and real-time features
- Rationale: Comprehensive backend-as-a-service with PostgreSQL and pgvector
- Impact: Eliminates need for separate MinIO, Redis, or custom auth solutions

**ADR-004: API-First, No Dedicated UI**
- Decision: Provide functionality via REST API only initially
- Rationale: SMB clients may integrate into existing tools
- Consequences: Relies on external UI development

**ADR-005: Use OpenAI GPT-4 for Report Generation**
- Decision: Leverage GPT-4 as primary narrative engine
- Rationale: Highest quality output for consulting-grade reports
- Consequences: External dependency and cost per report

**ADR-006: Containerize with Docker Compose**
- Decision: Deploy with Docker Compose including full Supabase stack
- Rationale: Simplicity for solo dev and reproducible local-first deployment
- Consequences: Limited per-container scalability but easier management

**ADR-007: PostgreSQL with pgvector for Analytics**
- Decision: Use PostgreSQL with pgvector extension for vector operations and analytics
- Rationale: Provides both relational data storage and vector similarity search
- Consequences: Can query across analyses and provide RAG capabilities

**ADR-008: Dedicated Data Cleaning Agent**
- Decision: Implement separate data_cleaner_agent between data extraction and processing
- Rationale: Ensures data is standardized and optimized for LLM processing
- Impact: Improves data quality, provides audit trail, enables LLM-specific formatting
- Consequences: Additional workflow step but significantly better data quality and analysis

### 1.9 Basic Configuration Files

Create `src/config/settings.py`:
```python
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # LLM Configuration
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    llm_provider: str = "openai"
    embedding_model: str = "text-embedding-ada-002"
    
    # Supabase Configuration
    supabase_url: str = "http://127.0.0.1:54321"
    supabase_anon_key: Optional[str] = None
    supabase_service_key: Optional[str] = None
    supabase_jwt_secret: Optional[str] = None
    
    # Database Configuration (PostgreSQL with pgvector)
    database_url: str = "postgresql://postgres:postgres@127.0.0.1:54322/postgres"
    vecs_connection_string: str = "postgresql://postgres:postgres@127.0.0.1:54322/postgres"
    
    # Storage Configuration
    storage_provider: str = "supabase"
    storage_bucket: str = "reports"
    charts_bucket: str = "charts"
    supabase_storage_url: str = "http://127.0.0.1:54321/storage/v1"
    
    # Vector Database Configuration
    vector_collection_name: str = "documents"
    vector_dimension: int = 1536
    vector_similarity_threshold: float = 0.8
    
    # API Configuration
    api_auth_key: Optional[str] = None
    ae_env: str = "development"
    
    # Processing Configuration
    max_upload_size: int = 52428800  # 50MB
    report_retention_days: int = 90
    ocr_language: str = "eng"
    
    # Performance
    max_concurrent_requests: int = 5
    llm_timeout_seconds: int = 300
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "json"
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
```

**Create Supabase utilities (`src/utils/supabase_client.py`):**
```python
from supabase import create_client, Client
import vecs
from src.config.settings import settings

def create_supabase_client() -> Client:
    """Create Supabase client with proper configuration"""
    return create_client(
        supabase_url=settings.supabase_url,
        supabase_key=settings.supabase_service_key
    )

def create_vecs_client() -> vecs.Client:
    """Create vecs client for vector operations"""
    return vecs.create_client(settings.vecs_connection_string)

def get_vector_collection(dimension: int = 1536, collection_name: str = "documents"):
    """Get or create vector collection for RAG"""
    vx = create_vecs_client()
    return vx.get_or_create_collection(
        name=collection_name,
        dimension=dimension
    )

# Initialize clients
supabase_client = create_supabase_client()
vecs_client = create_vecs_client()
```

## Phase 1 Acceptance Criteria

- [ ] **uv environment operational** (uv --version works, virtual environment created)
- [ ] **Development environment operational** (FastAPI hello world runs with uv run)
- [ ] **LangGraph imports correctly** and `uv run langgraph dev` works
- [ ] **Supabase CLI installed** and `supabase start` works
- [ ] **Project structure created** with all placeholder files
- [ ] **All dependencies installed via uv** and importable
- [ ] **pyproject.toml configured** with proper project metadata and dependencies
- [ ] **uv.lock file generated** for reproducible builds
- [ ] Environment variables template created
- [ ] Core data models defined and validated
- [ ] Persona configurations documented
- [ ] ADR index established with uv and Supabase decisions
- [ ] Settings management implemented with Supabase configuration
- [ ] Supabase client utilities created
- [ ] Vector database connection established
- [ ] Stakeholder sign-off on architecture

## Next Steps

After Phase 1 completion, proceed to Phase 2: Core Development - Modular Workflow Implementation.

## RACI Matrix

**Responsible:** Solo Developer
**Accountable:** Solo Developer  
**Consulted:** AI assistant for bootstrapping, Project Owner for requirements
**Informed:** Team lead/client sponsor on completion