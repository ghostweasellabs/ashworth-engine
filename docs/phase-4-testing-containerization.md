# Phase 4: Testing, Hardening, and Containerization

## Duration: 5-6 days
## Goal: Rigorously test the system, fix issues, and prepare deployment package

### 4.1 Comprehensive Test Suite Development

**Create test structure following best practices:**

```
tests/
├── __init__.py
├── conftest.py                    # Pytest configuration and fixtures
├── test_agents/
│   ├── __init__.py
│   ├── test_data_fetcher.py       # Unit tests for data extraction
│   ├── test_data_processor.py     # Unit tests for analytics
│   ├── test_tax_categorizer.py    # Unit tests for tax logic
│   └── test_report_generator.py   # Unit tests for report generation
├── test_workflows/
│   ├── __init__.py
│   ├── test_financial_analysis.py # Integration tests for full workflow
│   └── test_workflow_routing.py   # Test analysis_type routing
├── test_utils/
│   ├── __init__.py
│   ├── test_file_processing.py    # Test file parsing logic
│   ├── test_data_validation.py    # Test validation functions
│   └── test_chart_generation.py   # Test visualization generation
├── test_api/
│   ├── __init__.py
│   ├── test_routes.py             # API endpoint testing
│   └── test_error_handling.py     # API error scenarios
├── fixtures/
│   ├── sample_data.csv            # Test data files
│   ├── sample_data.xlsx
│   ├── malformed_data.csv
│   └── large_dataset.csv
└── integration/
    ├── __init__.py
    ├── test_end_to_end.py         # Full system tests
    └── test_performance.py        # Performance benchmarks
```

**Core test fixtures in `conftest.py`:**

```python
import pytest
from decimal import Decimal
from datetime import datetime
from src.workflows.state_schemas import Transaction, FinancialMetrics, TaxSummary

@pytest.fixture
def sample_transactions():
    """Sample transactions for testing"""
    return [
        Transaction(
            date="2024-01-01",
            description="Office supplies from Staples",
            amount=Decimal('-150.00'),
            currency="USD"
        ),
        Transaction(
            date="2024-01-02", 
            description="Client payment received",
            amount=Decimal('5000.00'),
            currency="USD"
        ),
        Transaction(
            date="2024-01-03",
            description="Business lunch at restaurant",
            amount=Decimal('-85.50'),
            currency="USD"
        ),
        Transaction(
            date="2024-01-04",
            description="Software license renewal",
            amount=Decimal('-299.99'),
            currency="USD"
        )
    ]

@pytest.fixture
def sample_financial_metrics():
    """Sample financial metrics for testing"""
    return FinancialMetrics(
        total_revenue=Decimal('5000.00'),
        total_expenses=Decimal('535.49'),
        gross_profit=Decimal('4464.51'),
        gross_margin_pct=89.3,
        expense_by_category={
            "office_supplies": Decimal('150.00'),
            "meals": Decimal('85.50'),
            "equipment": Decimal('299.99')
        },
        anomalies=[],
        pattern_matches={"vendor_count": 3},
        detected_business_types=["consulting"]
    )

@pytest.fixture
def mock_llm_response():
    """Mock LLM response for testing"""
    return """
# Executive Summary

Based on analysis of 4 transactions totaling $5,535.49 in activity, the business shows strong profitability with an 89.3% gross margin.

## Key Findings
- Revenue: $5,000.00
- Expenses: $535.49
- Gross Profit: $4,464.51

## Strategic Recommendations
1. Continue current client engagement model
2. Monitor expense categorization for tax optimization
3. Consider scaling operations given strong margins
"""
```

### 4.2 Unit Tests for Agents

**Test data fetcher functionality (`test_agents/test_data_fetcher.py`):**

```python
import pytest
from unittest.mock import Mock, patch
from src.agents.data_fetcher import data_fetcher_agent
from src.utils.file_processing import parse_excel, parse_csv

class TestDataFetcher:
    
    def test_data_fetcher_with_csv(self, sample_csv_content):
        """Test data fetcher with CSV input"""
        state = {
            "file_content": sample_csv_content,
            "trace_id": "test-123",
            "analysis_type": "financial_analysis"
        }
        
        result = data_fetcher_agent(state)
        
        assert result["workflow_phase"] == "data_extraction_complete"
        assert len(result["transactions"]) > 0
        assert result["error_messages"] == []
    
    def test_data_fetcher_with_excel(self, sample_excel_content):
        """Test data fetcher with Excel input"""
        state = {
            "file_content": sample_excel_content,
            "trace_id": "test-124",
            "analysis_type": "financial_analysis"
        }
        
        result = data_fetcher_agent(state)
        
        assert result["workflow_phase"] == "data_extraction_complete"
        assert len(result["transactions"]) > 0
    
    def test_data_fetcher_no_content(self):
        """Test data fetcher with no file content"""
        state = {
            "file_content": None,
            "trace_id": "test-125"
        }
        
        result = data_fetcher_agent(state)
        
        assert result["workflow_phase"] == "data_extraction_failed"
        assert "No file content provided" in result["error_messages"][0]
    
    def test_data_fetcher_invalid_format(self):
        """Test data fetcher with unsupported format"""
        state = {
            "file_content": b"invalid content",
            "trace_id": "test-126"
        }
        
        with patch('src.utils.file_processing.detect_file_type') as mock_detect:
            mock_detect.return_value = "unknown"
            
            result = data_fetcher_agent(state)
            
            assert result["workflow_phase"] == "data_extraction_failed"
            assert "Unsupported file type" in result["error_messages"][0]
```

**Test financial calculations (`test_agents/test_data_processor.py`):**

```python
import pytest
from decimal import Decimal
from src.agents.data_processor import data_processor_agent
from src.utils.financial_calculations import calculate_financial_metrics

class TestDataProcessor:
    
    def test_financial_metrics_calculation(self, sample_transactions):
        """Test basic financial metrics calculation"""
        metrics = calculate_financial_metrics(sample_transactions)
        
        assert metrics.total_revenue == Decimal('5000.00')
        assert metrics.total_expenses == Decimal('535.49')
        assert metrics.gross_profit == Decimal('4464.51')
        assert abs(metrics.gross_margin_pct - 89.3) < 0.1
    
    def test_data_processor_with_valid_transactions(self, sample_transactions):
        """Test data processor with valid transaction data"""
        state = {
            "transactions": sample_transactions,
            "trace_id": "test-200"
        }
        
        result = data_processor_agent(state)
        
        assert result["workflow_phase"] == "data_processing_complete"
        assert result["financial_metrics"] is not None
        assert result["error_messages"] == []
    
    def test_data_processor_no_transactions(self):
        """Test data processor with no transactions"""
        state = {
            "transactions": [],
            "trace_id": "test-201"
        }
        
        result = data_processor_agent(state)
        
        assert "No transactions to process" in result["error_messages"]
    
    def test_anomaly_detection(self, sample_transactions):
        """Test anomaly detection in transactions"""
        # Add an anomalous transaction
        anomalous_transaction = Transaction(
            date="2024-01-05",
            description="Extremely large expense",
            amount=Decimal('-50000.00'),  # Much larger than others
            currency="USD"
        )
        
        transactions_with_anomaly = sample_transactions + [anomalous_transaction]
        metrics = calculate_financial_metrics(transactions_with_anomaly)
        
        assert len(metrics.anomalies) > 0
        assert any(abs(float(t.amount)) > 10000 for t in metrics.anomalies)
    
    def test_precision_calculations(self):
        """Test decimal precision in financial calculations"""
        transactions = [
            Transaction(date="2024-01-01", description="Test", amount=Decimal('100.33')),
            Transaction(date="2024-01-02", description="Test", amount=Decimal('-50.17')),
            Transaction(date="2024-01-03", description="Test", amount=Decimal('-25.16'))
        ]
        
        metrics = calculate_financial_metrics(transactions)
        
        # Test precision - should be exact
        assert metrics.total_revenue == Decimal('100.33')
        assert metrics.total_expenses == Decimal('75.33')
        assert metrics.gross_profit == Decimal('25.00')
```

**Test tax categorization (`test_agents/test_tax_categorizer.py`):**

```python
import pytest
from decimal import Decimal
from src.agents.tax_categorizer import categorize_transaction, calculate_tax_summary
from src.workflows.state_schemas import Transaction

class TestTaxCategorizer:
    
    def test_travel_categorization(self):
        """Test travel expense categorization"""
        transaction = Transaction(
            date="2024-01-01",
            description="Hotel booking for business trip",
            amount=Decimal('-250.00')
        )
        
        category = categorize_transaction(transaction)
        assert category == "travel"
    
    def test_meals_categorization(self):
        """Test meals expense categorization"""
        transaction = Transaction(
            date="2024-01-01", 
            description="Business lunch at restaurant",
            amount=Decimal('-85.50')
        )
        
        category = categorize_transaction(transaction)
        assert category == "meals"
    
    def test_meals_50_percent_deduction(self):
        """Test that meals are 50% deductible"""
        transactions = [
            Transaction(
                date="2024-01-01",
                description="Business lunch",
                amount=Decimal('-100.00'),
                tax_category="meals"
            )
        ]
        
        tax_summary = calculate_tax_summary(transactions)
        
        # Should be 50% deductible = $50
        assert tax_summary.deductible_total == Decimal('50.00')
    
    def test_office_supplies_categorization(self):
        """Test office supplies categorization"""
        transaction = Transaction(
            date="2024-01-01",
            description="Staples office supplies",
            amount=Decimal('-150.00')
        )
        
        category = categorize_transaction(transaction)
        assert category == "office_supplies"
    
    def test_unknown_categorization(self):
        """Test unknown transaction categorization"""
        transaction = Transaction(
            date="2024-01-01",
            description="Some random expense",
            amount=Decimal('-100.00')
        )
        
        category = categorize_transaction(transaction)
        assert category == "other"
    
    def test_tax_summary_calculation(self, sample_transactions):
        """Test comprehensive tax summary calculation"""
        # Categorize transactions first
        for transaction in sample_transactions:
            transaction.tax_category = categorize_transaction(transaction)
        
        tax_summary = calculate_tax_summary(sample_transactions)
        
        assert tax_summary.deductible_total > 0
        assert tax_summary.potential_savings > 0
        assert isinstance(tax_summary.flags, list)
```

### 4.3 Integration Tests

**End-to-end workflow testing (`integration/test_end_to_end.py`):**

```python
import pytest
from src.workflows.financial_analysis import app as workflow_app
import tempfile
import os

class TestEndToEnd:
    
    @pytest.mark.asyncio
    async def test_complete_workflow_csv(self, sample_csv_content):
        """Test complete workflow with CSV input"""
        initial_state = {
            "client_id": "test-client",
            "analysis_type": "financial_analysis",
            "file_content": sample_csv_content,
            "trace_id": "e2e-test-001",
            "processing_start_time": "2024-01-01T00:00:00",
            "workflow_phase": "initialized",
            "transactions": [],
            "error_messages": [],
            "warnings": [],
            "charts": []
        }
        
        result = await workflow_app.ainvoke(initial_state)
        
        # Verify successful completion
        assert result["workflow_phase"] == "report_generation_complete"
        assert len(result["error_messages"]) == 0
        assert result["final_report_md"] is not None
        assert len(result["final_report_md"]) > 100  # Substantial content
    
    @pytest.mark.asyncio
    async def test_partial_workflow_data_collection(self, sample_csv_content):
        """Test partial workflow - data collection only"""
        initial_state = {
            "client_id": "test-client", 
            "analysis_type": "data_collection",
            "file_content": sample_csv_content,
            "trace_id": "e2e-test-002",
            "processing_start_time": "2024-01-01T00:00:00",
            "workflow_phase": "initialized",
            "transactions": [],
            "error_messages": [],
            "warnings": [],
            "charts": []
        }
        
        result = await workflow_app.ainvoke(initial_state)
        
        # Should stop after data fetcher
        assert result["workflow_phase"] == "data_extraction_complete"
        assert len(result["transactions"]) > 0
        assert result["final_report_md"] is None
    
    @pytest.mark.asyncio
    async def test_error_handling_invalid_file(self):
        """Test error handling with invalid file"""
        initial_state = {
            "client_id": "test-client",
            "analysis_type": "financial_analysis", 
            "file_content": b"invalid data",
            "trace_id": "e2e-test-003",
            "processing_start_time": "2024-01-01T00:00:00",
            "workflow_phase": "initialized",
            "transactions": [],
            "error_messages": [],
            "warnings": [],
            "charts": []
        }
        
        result = await workflow_app.ainvoke(initial_state)
        
        # Should have error but not crash
        assert len(result["error_messages"]) > 0
        assert "data_extraction_failed" in result["workflow_phase"]
```

### 4.4 API Testing

**FastAPI endpoint testing (`test_api/test_routes.py`):**

```python
import pytest
from fastapi.testclient import TestClient
from src.api.routes import app
import io

client = TestClient(app)

class TestAPIRoutes:
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
    
    def test_create_report_success(self, sample_csv_content):
        """Test successful report creation"""
        files = {"file": ("test.csv", io.BytesIO(sample_csv_content), "text/csv")}
        data = {"client_id": "test-client", "analysis_type": "financial_analysis"}
        
        response = client.post("/reports", files=files, data=data)
        
        assert response.status_code == 200
        result = response.json()
        assert "report_id" in result
        assert result["status"] in ["completed", "processing"]
    
    def test_create_report_no_file(self):
        """Test report creation without file"""
        data = {"client_id": "test-client"}
        
        response = client.post("/reports", data=data)
        
        assert response.status_code == 422  # Validation error
    
    def test_create_report_large_file(self):
        """Test file size limit enforcement"""
        # Create large file content (>50MB)
        large_content = b"x" * (60 * 1024 * 1024)  # 60MB
        files = {"file": ("large.csv", io.BytesIO(large_content), "text/csv")}
        data = {"client_id": "test-client"}
        
        response = client.post("/reports", files=files, data=data)
        
        assert response.status_code == 413  # File too large
    
    def test_get_report_status(self):
        """Test report status endpoint"""
        response = client.get("/reports/test-report-id")
        
        assert response.status_code == 200
        result = response.json()
        assert "report_id" in result
        assert "status" in result
```

### 4.5 Performance Testing

**Load testing and benchmarks (`integration/test_performance.py`):**

```python
import pytest
import time
import asyncio
from src.workflows.financial_analysis import app as workflow_app
from src.utils.financial_calculations import calculate_financial_metrics

class TestPerformance:
    
    def test_large_dataset_processing(self, large_transaction_dataset):
        """Test processing of large dataset (1000+ transactions)"""
        start_time = time.time()
        
        metrics = calculate_financial_metrics(large_transaction_dataset)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should process 1000 transactions in under 5 seconds
        assert processing_time < 5.0
        assert metrics.total_revenue > 0
        assert len(metrics.expense_by_category) > 0
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, sample_csv_content):
        """Test handling of concurrent requests"""
        
        async def create_report(client_id):
            initial_state = {
                "client_id": client_id,
                "analysis_type": "financial_analysis",
                "file_content": sample_csv_content,
                "trace_id": f"perf-test-{client_id}",
                "processing_start_time": "2024-01-01T00:00:00",
                "workflow_phase": "initialized",
                "transactions": [],
                "error_messages": [],
                "warnings": [],
                "charts": []
            }
            return await workflow_app.ainvoke(initial_state)
        
        # Run 5 concurrent requests
        start_time = time.time()
        
        tasks = [create_report(f"client-{i}") for i in range(5)]
        results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # All should complete successfully
        assert all(len(result["error_messages"]) == 0 for result in results)
        # Should handle concurrency reasonably
        assert total_time < 30.0  # 30 seconds for 5 concurrent requests
    
    def test_memory_usage(self, large_transaction_dataset):
        """Test memory usage with large datasets"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Process large dataset multiple times
        for _ in range(10):
            metrics = calculate_financial_metrics(large_transaction_dataset)
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (< 100MB)
        assert memory_increase < 100 * 1024 * 1024
```

### 4.6 Security Testing

**Basic security tests (`test_api/test_security.py`):**

```python
import pytest
from fastapi.testclient import TestClient
from src.api.routes import app

client = TestClient(app)

class TestSecurity:
    
    def test_file_path_traversal_protection(self):
        """Test protection against path traversal attacks"""
        response = client.get("/reports/../../../etc/passwd")
        
        # Should not return sensitive files
        assert response.status_code in [404, 403, 422]
    
    def test_malicious_file_upload(self):
        """Test handling of malicious file uploads"""
        # Try uploading executable file
        malicious_content = b"#!/bin/bash\necho 'malicious'"
        files = {"file": ("malicious.sh", malicious_content, "application/x-sh")}
        data = {"client_id": "test"}
        
        response = client.post("/reports", files=files, data=data)
        
        # Should handle gracefully, not execute
        assert response.status_code in [400, 422, 500]
    
    def test_sql_injection_protection(self):
        """Test protection against SQL injection in parameters"""
        # Try SQL injection in client_id
        malicious_data = {"client_id": "'; DROP TABLE users; --"}
        
        response = client.post("/reports", data=malicious_data)
        
        # Should validate input properly
        assert response.status_code == 422  # Validation error
    
    def test_large_payload_protection(self):
        """Test protection against large payloads"""
        # This is covered in the file size test in API routes
        pass
```

### 4.7 Supabase Project Setup & Configuration

**Initialize Supabase Project:**

```bash
# Install Supabase CLI globally
npm install -g supabase

# Login to Supabase (for remote project access)
supabase login

# Initialize local Supabase project with vector support
supabase init

# Start local Supabase stack (includes PostgreSQL with pgvector)
supabase start

# This will output:
# Started supabase local development setup.
# 
#          API URL: http://127.0.0.1:54321
#           DB URL: postgresql://postgres:postgres@127.0.0.1:54322/postgres
#       Studio URL: http://127.0.0.1:54323
#     Inbucket URL: http://127.0.0.1:54324
#         anon key: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
# service_role key: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
#      JWT secret: your-super-secret-jwt-token-with-at-least-32-characters-long

# Create a new Supabase project (for production)
supabase projects create ashworth-engine --org-id your-org-id --db-password your-secure-password --region us-east-1

# Link local project to remote (optional, for production deployment)
# supabase link --project-ref your-project-ref
```

**Create Comprehensive Supabase Setup:**

```sql
-- Add to supabase/migrations/001_setup_extensions.sql
-- Enable required extensions
create extension if not exists vector with schema extensions;
create extension if not exists pg_cron with schema extensions;
create extension if not exists "uuid-ossp" with schema extensions;

-- Enable Row Level Security
alter database postgres set "app.jwt_secret" to 'your-super-secret-jwt-token-with-at-least-32-characters-long';
```

```sql
-- Add to supabase/migrations/002_setup_vector_tables.sql
-- Create documents table for RAG (knowledge base)
create table if not exists public.documents (
  id uuid default gen_random_uuid() primary key,
  content text not null,
  metadata jsonb default '{}',
  embedding vector(1536),  -- OpenAI embedding dimension
  created_at timestamp with time zone default now(),
  updated_at timestamp with time zone default now()
);

-- Create function for similarity search
create or replace function match_documents (
  query_embedding vector(1536),
  match_count int default 5,
  filter jsonb default '{}'
) returns table (
  id uuid,
  content text,
  metadata jsonb,
  similarity float
)
language plpgsql
as $$
#variable_conflict use_column
begin
  return query
  select
    documents.id,
    documents.content,
    documents.metadata,
    1 - (documents.embedding <=> query_embedding) as similarity
  from documents
  where metadata @> filter
  order by documents.embedding <=> query_embedding
  limit match_count;
end;
$$;

-- Create indexes for vector similarity search
create index on public.documents using ivfflat (embedding vector_cosine_ops)
with (lists = 100);

create index on public.documents using gin (metadata);
create index on public.documents (created_at desc);
```

```sql
-- Add to supabase/migrations/003_setup_analytics_tables.sql
-- Create reports table for analytics and tracking
create table if not exists public.reports (
  id uuid default gen_random_uuid() primary key,
  client_id text not null,
  analysis_type text not null,
  file_name text,
  file_size bigint,
  processing_start_time timestamp with time zone,
  processing_end_time timestamp with time zone,
  status text not null default 'processing',
  error_message text,
  report_path text,
  charts jsonb default '[]',
  metadata jsonb default '{}',
  financial_metrics jsonb,
  tax_summary jsonb,
  created_at timestamp with time zone default now(),
  updated_at timestamp with time zone default now()
);

-- Create indexes for reports
create index on public.reports (client_id);
create index on public.reports (status);
create index on public.reports (created_at desc);
create index on public.reports (analysis_type);

-- Create function to update updated_at timestamp
create or replace function update_updated_at_column()
returns trigger as $$
begin
    new.updated_at = now();
    return new;
end;
$$ language 'plpgsql';

-- Create triggers for updated_at
create trigger update_documents_updated_at before update on public.documents
    for each row execute procedure update_updated_at_column();

create trigger update_reports_updated_at before update on public.reports
    for each row execute procedure update_updated_at_column();
```

```sql
-- Add to supabase/migrations/004_setup_storage.sql
-- Create storage buckets
insert into storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
values 
  ('reports', 'reports', false, 52428800, array['application/pdf', 'text/plain']),
  ('charts', 'charts', false, 10485760, array['image/png', 'image/jpeg', 'image/svg+xml'])
on conflict (id) do nothing;

-- Create storage policies for reports bucket
create policy "Service role can manage all report files" on storage.objects
  for all using (bucket_id = 'reports' and auth.role() = 'service_role');

create policy "Authenticated users can view their reports" on storage.objects
  for select using (
    bucket_id = 'reports' and 
    (auth.role() = 'authenticated' and 
     (storage.foldername(name))[1] = auth.uid()::text)
  );

-- Create storage policies for charts bucket
create policy "Service role can manage all chart files" on storage.objects
  for all using (bucket_id = 'charts' and auth.role() = 'service_role');

create policy "Authenticated users can view charts" on storage.objects
  for select using (bucket_id = 'charts' and auth.role() = 'authenticated');
```

```sql
-- Add to supabase/migrations/005_setup_rls_policies.sql
-- Enable Row Level Security
alter table public.documents enable row level security;
alter table public.reports enable row level security;

-- RLS policies for documents table
create policy "Service role can manage all documents" on public.documents
  for all using (auth.role() = 'service_role');

create policy "Authenticated users can read documents" on public.documents
  for select using (auth.role() = 'authenticated');

-- RLS policies for reports table
create policy "Service role can manage all reports" on public.reports
  for all using (auth.role() = 'service_role');

create policy "Authenticated users can view their reports" on public.reports
  for select using (
    auth.role() = 'authenticated' and 
    client_id = auth.uid()::text
  );
```

**Configure Environment Variables:**

```bash
# Create .env file with Supabase credentials
cat > .env << EOF
# LLM Configuration
OPENAI_API_KEY=your-openai-api-key
LLM_PROVIDER=openai

# Supabase Configuration (Local Development)
SUPABASE_URL=http://127.0.0.1:54321
SUPABASE_ANON_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZS1kZW1vIiwicm9sZSI6ImFub24iLCJleHAiOjE5ODM4MTI5OTZ9.CRXP1A7WOeoJeXxjNni43kdQwgnWNReilDMblYTn_I0
SUPABASE_SERVICE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZS1kZW1vIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImV4cCI6MTk4MzgxMjk5Nn0.EGIM96RAZx35lJzdJsyH-qQwv8Hdp7fsn3W0YpN81IU
SUPABASE_JWT_SECRET=your-super-secret-jwt-token-with-at-least-32-characters-long

# Database Configuration (with pgvector support)
DATABASE_URL=postgresql://postgres:postgres@127.0.0.1:54322/postgres
VECS_CONNECTION_STRING=postgresql://postgres:postgres@127.0.0.1:54322/postgres

# Storage Configuration
STORAGE_PROVIDER=supabase
STORAGE_BUCKET=reports
SUPABASE_STORAGE_URL=http://127.0.0.1:54321/storage/v1

# Vector Database Configuration for RAG
VECTOR_COLLECTION_NAME=documents
VECTOR_DIMENSION=1536
EMBEDDING_MODEL=text-embedding-ada-002

# Application Configuration
AE_ENV=development
LOG_LEVEL=INFO
MAX_UPLOAD_SIZE=52428800
OCR_LANGUAGE=eng
REPORT_RETENTION_DAYS=90

# Production Supabase Configuration (uncomment for production):
# SUPABASE_URL=https://your-project-ref.supabase.co
# SUPABASE_ANON_KEY=your-production-anon-key
# SUPABASE_SERVICE_KEY=your-production-service-key
# DATABASE_URL=postgresql://postgres.[PROJECT_REF]:[PASSWORD]@aws-0-[REGION].pooler.supabase.com:6543/postgres
# VECS_CONNECTION_STRING=postgresql://postgres.[PROJECT_REF]:[PASSWORD]@aws-0-[REGION].pooler.supabase.com:5432/postgres
EOF
```

**Update requirements.txt to include Supabase dependencies:**

```txt
# Core LangGraph dependencies
langgraph>=0.2.0
langchain>=0.3.0
langchain-openai>=0.2.0
langchain-anthropic>=0.2.0

# FastAPI and API dependencies
fastapi>=0.110.0
uvicorn[standard]>=0.27.0
pydantic>=2.5.0
python-multipart

# Data processing
pandas>=2.0.0
numpy>=1.24.0
openpyxl>=3.1.0

# Document processing
pypdf2>=3.0.0
pdfplumber>=0.10.0
pytesseract>=0.3.10
pillow>=10.0.0

# Chart generation
pyecharts>=2.0.8
seaborn>=0.12.0
plotly>=5.17.0

# Supabase and vector database integration
supabase>=2.0.0
vecs>=0.4.0
psycopg2-binary>=2.9.7
sqlalchemy>=2.0.0

# PDF generation
weasyprint>=60.0
markdown>=3.5.0

# Development and testing tools
pytest>=7.4.0
pytest-cov>=4.1.0
pytest-asyncio>=0.21.0
black>=23.0.0
isort>=5.12.0
mypy>=1.7.0
bandit>=1.7.5
safety>=2.3.0

# Additional utilities
requests>=2.31.0
aiofiles>=23.0.0
python-dotenv>=1.0.0
structlog>=23.0.0
```

**Update settings configuration (`src/config/settings.py`):**

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

**Create Supabase client configuration (`src/utils/supabase_client.py`):**

```python
import os
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

**Create production-ready Dockerfile:**

```dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    poppler-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    postgresql-client \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd --create-home --shell /bin/bash app

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY langgraph.json .
COPY supabase/ ./supabase/

# Create necessary directories
RUN mkdir -p /tmp/charts /tmp/reports && \
    chown -R app:app /app /tmp/charts /tmp/reports

# Switch to app user
USER app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start application
CMD ["uvicorn", "src.api.routes:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Create docker-compose.yml with full Supabase stack:**

```yaml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      # LLM Configuration
      - LLM_PROVIDER=${LLM_PROVIDER:-openai}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - EMBEDDING_MODEL=${EMBEDDING_MODEL:-text-embedding-ada-002}
      
      # Supabase Configuration
      - SUPABASE_URL=http://kong:8000
      - SUPABASE_ANON_KEY=${ANON_KEY}
      - SUPABASE_SERVICE_KEY=${SERVICE_ROLE_KEY}
      - SUPABASE_JWT_SECRET=${JWT_SECRET}
      
      # Database Configuration
      - DATABASE_URL=postgresql://postgres:${POSTGRES_PASSWORD}@db:5432/postgres
      - VECS_CONNECTION_STRING=postgresql://postgres:${POSTGRES_PASSWORD}@db:5432/postgres
      
      # Storage Configuration
      - STORAGE_PROVIDER=supabase
      - STORAGE_BUCKET=reports
      - CHARTS_BUCKET=charts
      - SUPABASE_STORAGE_URL=http://kong:8000/storage/v1
      
      # Vector Configuration
      - VECTOR_COLLECTION_NAME=documents
      - VECTOR_DIMENSION=1536
      
      # Application Configuration
      - AE_ENV=${AE_ENV:-development}
      - LOG_LEVEL=${LOG_LEVEL:-INFO}
    depends_on:
      db:
        condition: service_healthy
      kong:
        condition: service_healthy
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped
    networks:
      - ashworth-network

  # Supabase Services
  kong:
    image: kong:2.8.1
    restart: unless-stopped
    ports:
      - "${KONG_HTTP_PORT:-8000}:8000/tcp"
      - "${KONG_HTTPS_PORT:-8443}:8443/tcp"
    environment:
      KONG_DATABASE: "off"
      KONG_DECLARATIVE_CONFIG: /var/lib/kong/kong.yml
      KONG_DNS_ORDER: LAST,A,CNAME
      KONG_PLUGINS: request-transformer,cors,key-auth,acl,basic-auth
      KONG_NGINX_PROXY_PROXY_BUFFER_SIZE: 160k
      KONG_NGINX_PROXY_PROXY_BUFFERS: 64 160k
    volumes:
      - ./volumes/api/kong.yml:/var/lib/kong/kong.yml:ro
    healthcheck:
      test: ["CMD", "kong", "health"]
      timeout: 10s
      retries: 10
    networks:
      - ashworth-network

  auth:
    image: supabase/gotrue:v2.99.0
    depends_on:
      db:
        condition: service_healthy
    healthcheck:
      test:
        [
          "CMD",
          "wget",
          "--no-verbose",
          "--tries=1",
          "--spider",
          "http://localhost:9999/health"
        ]
      timeout: 5s
      interval: 5s
      retries: 3
    restart: unless-stopped
    environment:
      GOTRUE_API_HOST: 0.0.0.0
      GOTRUE_API_PORT: 9999
      GOTRUE_DB_DRIVER: postgres
      GOTRUE_DB_DATABASE_URL: postgresql://supabase_auth_admin:${POSTGRES_PASSWORD}@db:5432/postgres
      GOTRUE_SITE_URL: ${SITE_URL}
      GOTRUE_URI_ALLOW_LIST: ${ADDITIONAL_REDIRECT_URLS}
      GOTRUE_DISABLE_SIGNUP: ${DISABLE_SIGNUP}
      GOTRUE_JWT_SECRET: ${JWT_SECRET}
      GOTRUE_JWT_EXP: ${JWT_EXPIRY}
      GOTRUE_JWT_DEFAULT_GROUP_NAME: authenticated
      GOTRUE_JWT_ADMIN_ROLES: service_role
      GOTRUE_JWT_AUD: authenticated
      GOTRUE_EXTERNAL_EMAIL_ENABLED: ${ENABLE_EMAIL_SIGNUP}
      GOTRUE_MAILER_AUTOCONFIRM: ${ENABLE_EMAIL_AUTOCONFIRM}
      GOTRUE_SMTP_ADMIN_EMAIL: ${SMTP_ADMIN_EMAIL}
      GOTRUE_SMTP_HOST: ${SMTP_HOST}
      GOTRUE_SMTP_PORT: ${SMTP_PORT}
      GOTRUE_SMTP_USER: ${SMTP_USER}
      GOTRUE_SMTP_PASS: ${SMTP_PASS}
      GOTRUE_SMTP_SENDER_NAME: ${SMTP_SENDER_NAME}
      GOTRUE_MAILER_URLPATHS_INVITE: "/auth/v1/verify"
      GOTRUE_MAILER_URLPATHS_CONFIRMATION: "/auth/v1/verify"
      GOTRUE_MAILER_URLPATHS_RECOVERY: "/auth/v1/verify"
      GOTRUE_MAILER_URLPATHS_EMAIL_CHANGE: "/auth/v1/verify"
    networks:
      - ashworth-network

  rest:
    image: postgrest/postgrest:v10.1.1
    depends_on:
      db:
        condition: service_healthy
    restart: unless-stopped
    environment:
      PGRST_DB_URI: postgresql://authenticator:${POSTGRES_PASSWORD}@db:5432/postgres
      PGRST_DB_SCHEMAS: ${PGRST_DB_SCHEMAS}
      PGRST_DB_ANON_ROLE: anon
      PGRST_JWT_SECRET: ${JWT_SECRET}
      PGRST_DB_USE_LEGACY_GUCS: "false"
      PGRST_APP_SETTINGS_JWT_SECRET: ${JWT_SECRET}
      PGRST_APP_SETTINGS_JWT_EXP: ${JWT_EXPIRY}
    networks:
      - ashworth-network

  realtime:
    image: supabase/realtime:v2.10.1
    depends_on:
      db:
        condition: service_healthy
    healthcheck:
      test:
        [
          "CMD",
          "bash",
          "-c",
          "printf \\0 > /dev/tcp/localhost/4000"
        ]
      timeout: 5s
      interval: 5s
      retries: 3
    restart: unless-stopped
    environment:
      PORT: 4000
      DB_HOST: db
      DB_PORT: 5432
      DB_USER: supabase_admin
      DB_PASSWORD: ${POSTGRES_PASSWORD}
      DB_NAME: postgres
      DB_AFTER_CONNECT_QUERY: 'SET search_path TO _realtime'
      DB_ENC_KEY: supabaserealtime
      API_JWT_SECRET: ${JWT_SECRET}
      FLY_ALLOC_ID: fly123
      FLY_APP_NAME: realtime
      SECRET_KEY_BASE: UpNVntn3cDxHJpq99YMc1T1AQgQpc8kfYTuRgBiYa15BLrx8etQoXz3gZv1/u2oq
      ERL_AFLAGS: -proto_dist inet_tcp
      ENABLE_TAILSCALE: "false"
      DNS_NODES: "''"
    command: >-
      sh -c "/app/bin/migrate && /app/bin/realtime eval 'Realtime.Release.seeds(Realtime.Repo)' && /app/bin/server"
    networks:
      - ashworth-network

  storage:
    image: supabase/storage-api:v0.40.4
    depends_on:
      db:
        condition: service_healthy
      rest:
        condition: service_started
    healthcheck:
      test:
        [
          "CMD",
          "wget",
          "--no-verbose",
          "--tries=1",
          "--spider",
          "http://localhost:5000/status"
        ]
      timeout: 5s
      interval: 5s
      retries: 3
    restart: unless-stopped
    environment:
      ANON_KEY: ${ANON_KEY}
      SERVICE_KEY: ${SERVICE_ROLE_KEY}
      POSTGREST_URL: http://rest:3000
      PGRST_JWT_SECRET: ${JWT_SECRET}
      DATABASE_URL: postgresql://supabase_storage_admin:${POSTGRES_PASSWORD}@db:5432/postgres
      FILE_SIZE_LIMIT: 52428800
      STORAGE_BACKEND: file
      FILE_STORAGE_BACKEND_PATH: /var/lib/storage
      TENANT_ID: stub
      REGION: stub
      GLOBAL_S3_BUCKET: stub
      ENABLE_IMAGE_TRANSFORMATION: "true"
      IMGPROXY_URL: http://imgproxy:5001
    volumes:
      - ./volumes/storage:/var/lib/storage:z
    networks:
      - ashworth-network

  imgproxy:
    image: darthsim/imgproxy:v3.8.0
    healthcheck:
      test: [ "CMD", "imgproxy", "health" ]
      timeout: 5s
      interval: 5s
      retries: 3
    environment:
      IMGPROXY_BIND: ":5001"
      IMGPROXY_LOCAL_FILESYSTEM_ROOT: /
      IMGPROXY_USE_ETAG: "true"
      IMGPROXY_ENABLE_WEBP_DETECTION: ${IMGPROXY_ENABLE_WEBP_DETECTION}
    volumes:
      - ./volumes/storage:/var/lib/storage:z
    networks:
      - ashworth-network

  meta:
    image: supabase/postgres-meta:v0.68.0
    depends_on:
      db:
        condition: service_healthy
    restart: unless-stopped
    environment:
      PG_META_PORT: 8080
      PG_META_DB_HOST: db
      PG_META_DB_PORT: 5432
      PG_META_DB_NAME: postgres
      PG_META_DB_USER: supabase_admin
      PG_META_DB_PASSWORD: ${POSTGRES_PASSWORD}
    networks:
      - ashworth-network

  functions:
    image: supabase/edge-runtime:v1.2.9
    depends_on:
      db:
        condition: service_healthy
    restart: unless-stopped
    environment:
      JWT_SECRET: ${JWT_SECRET}
      SUPABASE_URL: http://kong:8000
      SUPABASE_ANON_KEY: ${ANON_KEY}
      SUPABASE_SERVICE_ROLE_KEY: ${SERVICE_ROLE_KEY}
      SUPABASE_DB_URL: postgresql://postgres:${POSTGRES_PASSWORD}@db:5432/postgres
      VERIFY_JWT: "${FUNCTIONS_VERIFY_JWT}"
    volumes:
      - ./volumes/functions:/home/deno/functions:Z
    command:
      - start
      - --main-service
      - /home/deno/functions/main
    networks:
      - ashworth-network

  analytics:
    image: supabase/logflare:1.4.0
    healthcheck:
      test: [ "CMD", "curl", "http://localhost:4000/health" ]
      timeout: 5s
      interval: 5s
      retries: 10
    restart: unless-stopped
    depends_on:
      db:
        condition: service_healthy
    environment:
      LOGFLARE_NODE_HOST: 127.0.0.1
      DB_USERNAME: supabase_admin
      DB_DATABASE: postgres
      DB_HOSTNAME: db
      DB_PORT: 5432
      DB_PASSWORD: ${POSTGRES_PASSWORD}
      DB_SCHEMA: _analytics
      LOGFLARE_API_KEY: ${LOGFLARE_API_KEY}
      LOGFLARE_SINGLE_TENANT: true
      LOGFLARE_SUPABASE_MODE: true
      LOGFLARE_MIN_CLUSTER_SIZE: 1
      RELEASE_COOKIE: cookie
    networks:
      - ashworth-network
    ports:
      - "4000:4000"

  db:
    image: supabase/postgres:15.1.0.117
    healthcheck:
      test: pg_isready -U postgres -h localhost
      interval: 5s
      timeout: 5s
      retries: 10
    depends_on:
      vector:
        condition: service_completed_successfully
    command:
      - postgres
      - -c
      - config_file=/etc/postgresql/postgresql.conf
      - -c
      - log_min_messages=fatal
    restart: unless-stopped
    ports:
      - "${POSTGRES_PORT:-5432}:5432"
    environment:
      POSTGRES_HOST: /var/run/postgresql
      PGPORT: 5432
      POSTGRES_PORT: 5432
      PGPASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      PGDATABASE: postgres
      POSTGRES_DB: postgres
      POSTGRES_USER: postgres
      POSTGRES_INITDB_ARGS: "--lc-collate=C --lc-ctype=C"
    volumes:
      - ./volumes/db/realtime.sql:/docker-entrypoint-initdb.d/migrations/99-realtime.sql:Z
      - ./volumes/db/webhooks.sql:/docker-entrypoint-initdb.d/init-scripts/98-webhooks.sql:Z
      - ./volumes/db/roles.sql:/docker-entrypoint-initdb.d/init-scripts/99-roles.sql:Z
      - ./volumes/db/jwt.sql:/docker-entrypoint-initdb.d/init-scripts/99-jwt.sql:Z
      - ./supabase/migrations:/docker-entrypoint-initdb.d/migrations:Z
      - db-data:/var/lib/postgresql/data:Z
      - ./volumes/db/logs.sql:/docker-entrypoint-initdb.d/migrations/99-logs.sql:Z
    networks:
      - ashworth-network

  vector:
    image: timberio/vector:0.28.1-alpine
    healthcheck:
      test:
        [
          "CMD",
          "wget",
          "--no-verbose",
          "--tries=1",
          "--spider",
          "http://localhost:9001/health"
        ]
      timeout: 5s
      interval: 5s
      retries: 3
    volumes:
      - ./volumes/logs/vector.yml:/etc/vector/vector.yml:ro
      - ${DOCKER_SOCKET_LOCATION}:/var/run/docker.sock:ro
    command: [ "--config", "etc/vector/vector.yml" ]
    networks:
      - ashworth-network

volumes:
  db-data:

networks:
  ashworth-network:
    external: false
```

**Create .env template for docker-compose:**

```bash
# Create .env file for docker-compose
cat > .env << EOF

```yaml
# Supabase Configuration
POSTGRES_PASSWORD=your-super-secret-password
JWT_SECRET=your-super-secret-jwt-token-with-at-least-32-characters-long
ANON_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZS1kZW1vIiwicm9sZSI6ImFub24iLCJleHAiOjE5ODM4MTI5OTZ9.CRXP1A7WOeoJeXxjNni43kdQwgnWNReilDMblYTn_I0
SERVICE_ROLE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZS1kZW1vIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImV4cCI6MTk4MzgxMjk5Nn0.EGIM96RAZx35lJzdJsyH-qQwv8Hdp7fsn3W0YpN81IU

# Database
POSTGRES_PORT=5432
PGRST_DB_SCHEMAS=public,storage,graphql_public

# API Proxy
KONG_HTTP_PORT=8000
KONG_HTTPS_PORT=8443

# Auth
SITE_URL=http://localhost:3000
ADDITIONAL_REDIRECT_URLS=
JWT_EXPIRY=3600
DISABLE_SIGNUP=false
ENABLE_EMAIL_SIGNUP=true
ENABLE_EMAIL_AUTOCONFIRM=true
SMTP_ADMIN_EMAIL=admin@example.com
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASS=your-smtp-password
SMTP_SENDER_NAME=Ashworth Engine

# Storage
ENABLE_IMAGE_TRANSFORMATION=true
IMGPROXY_ENABLE_WEBP_DETECTION=true

# Functions
FUNCTIONS_VERIFY_JWT=false

# Analytics
LOGFLARE_API_KEY=your-logflare-api-key

# Vector/Logging
DOCKER_SOCKET_LOCATION=/var/run/docker.sock

# Application
OPENAI_API_KEY=your-openai-api-key
LLM_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-ada-002
AE_ENV=development
LOG_LEVEL=INFO
EOF
```

**Initialize Supabase volumes and configuration:**

```bash
# Create necessary directories and files
mkdir -p volumes/{api,db,functions,storage,logs}

# Create Kong configuration
cat > volumes/api/kong.yml << 'EOF'
_format_version: "1.1"
_transform: true

services:
  - name: auth-v1-open
    url: http://auth:9999/verify
    routes:
      - name: auth-v1-open
        strip_path: true
        paths:
          - /auth/v1/verify
    plugins:
      - name: cors
  - name: auth-v1-open-callback
    url: http://auth:9999/callback
    routes:
      - name: auth-v1-open-callback
        strip_path: true
        paths:
          - /auth/v1/callback
    plugins:
      - name: cors
  - name: auth-v1-open-authorize
    url: http://auth:9999/authorize
    routes:
      - name: auth-v1-open-authorize
        strip_path: true
        paths:
          - /auth/v1/authorize
    plugins:
      - name: cors

  - name: auth-v1
    _comment: "GoTrue: /auth/v1/* -> http://auth:9999/*"
    url: http://auth:9999/
    routes:
      - name: auth-v1-all
        strip_path: true
        paths:
          - /auth/v1/
    plugins:
      - name: cors
      - name: key-auth
        config:
          hide_credentials: false

  - name: rest-v1
    _comment: "PostgREST: /rest/v1/* -> http://rest:3000/*"
    url: http://rest:3000/
    routes:
      - name: rest-v1-all
        strip_path: true
        paths:
          - /rest/v1/
    plugins:
      - name: cors
      - name: key-auth
        config:
          hide_credentials: true

  - name: realtime-v1
    _comment: "Realtime: /realtime/v1/* -> ws://realtime:4000/socket/*"
    url: http://realtime:4000/socket/
    routes:
      - name: realtime-v1-all
        strip_path: true
        paths:
          - /realtime/v1/
    plugins:
      - name: cors
      - name: key-auth
        config:
          hide_credentials: false

  - name: storage-v1
    _comment: "Storage: /storage/v1/* -> http://storage:5000/*"
    url: http://storage:5000/
    routes:
      - name: storage-v1-all
        strip_path: true
        paths:
          - /storage/v1/
    plugins:
      - name: cors

  - name: functions-v1
    _comment: "Edge Functions: /functions/v1/* -> http://functions:9000/*"
    url: http://functions:9000/
    routes:
      - name: functions-v1-all
        strip_path: true
        paths:
          - /functions/v1/
    plugins:
      - name: cors

  - name: meta
    _comment: "pg-meta: /pg/* -> http://meta:8080/*"
    url: http://meta:8080/
    routes:
      - name: meta-all
        strip_path: true
        paths:
          - /pg/

consumers:
  - username: anon
    keyauth_credentials:
      - key: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZS1kZW1vIiwicm9sZSI6ImFub24iLCJleHAiOjE5ODM4MTI5OTZ9.CRXP1A7WOeoJeXxjNni43kdQwgnWNReilDMblYTn_I0
  - username: service_role
    keyauth_credentials:
      - key: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZS1kZW1vIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImV4cCI6MTk4MzgxMjk5Nn0.EGIM96RAZx35lJzdJsyH-qQwv8Hdp7fsn3W0YpN81IU

acls:
  - consumer: anon
    group: anon
  - consumer: service_role
    group: admin
EOF

# Create Vector logging configuration
cat > volumes/logs/vector.yml << 'EOF'
api:
  enabled: true
  address: 0.0.0.0:9001

sources:
  docker_host:
    type: docker_logs
    docker_host: unix:///var/run/docker.sock

sinks:
  logflare_logs:
    type: http
    inputs: ["docker_host"]
    uri: http://analytics:4000/api/logs
    method: post
    auth:
      strategy: bearer
      token: logflare-api-key
    encoding:
      codec: json
    healthcheck:
      enabled: false
EOF

# Create database initialization scripts
cp -r supabase/migrations volumes/db/ 2>/dev/null || echo "Migrations will be copied during build"
```

**Update test configuration to use Supabase (`tests/conftest.py`):**

```python
import pytest
import asyncio
from decimal import Decimal
from datetime import datetime
from unittest.mock import Mock, patch
from src.workflows.state_schemas import Transaction, FinancialMetrics, TaxSummary
from src.config.settings import settings

# Override settings for testing
settings.supabase_url = "http://localhost:54321"
settings.database_url = "postgresql://postgres:postgres@localhost:54322/postgres"
settings.storage_provider = "supabase"
settings.ae_env = "testing"

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def mock_supabase_client():
    """Mock Supabase client for testing"""
    with patch('src.utils.supabase_client.create_supabase_client') as mock:
        mock_client = Mock()
        mock_client.storage.from_.return_value.upload.return_value = {
            'data': {'path': 'test-report.pdf'},
            'error': None
        }
        mock_client.table.return_value.insert.return_value.execute.return_value = {
            'data': [{'id': 'test-report-id'}],
            'error': None
        }
        mock.return_value = mock_client
        yield mock_client

@pytest.fixture
def mock_vecs_client():
    """Mock vecs client for testing"""
    with patch('src.utils.supabase_client.create_vecs_client') as mock:
        mock_client = Mock()
        mock_collection = Mock()
        mock_collection.upsert.return_value = None
        mock_collection.query.return_value = []
        mock_client.get_or_create_collection.return_value = mock_collection
        mock.return_value = mock_client
        yield mock_client

@pytest.fixture
def sample_transactions():
    """Sample transactions for testing"""
    return [
        Transaction(
            date="2024-01-01",
            description="Office supplies from Staples",
            amount=Decimal('-150.00'),
            currency="USD"
        ),
        Transaction(
            date="2024-01-02", 
            description="Client payment received",
            amount=Decimal('5000.00'),
            currency="USD"
        ),
        Transaction(
            date="2024-01-03",
            description="Business lunch at restaurant",
            amount=Decimal('-85.50'),
            currency="USD"
        ),
        Transaction(
            date="2024-01-04",
            description="Software license renewal",
            amount=Decimal('-299.99'),
            currency="USD"
        )
    ]

@pytest.fixture
def sample_financial_metrics():
    """Sample financial metrics for testing"""
    return FinancialMetrics(
        total_revenue=Decimal('5000.00'),
        total_expenses=Decimal('535.49'),
        gross_profit=Decimal('4464.51'),
        gross_margin_pct=89.3,
        expense_by_category={
            "office_supplies": Decimal('150.00'),
            "meals": Decimal('85.50'),
            "equipment": Decimal('299.99')
        },
        anomalies=[],
        pattern_matches={"vendor_count": 3},
        detected_business_types=["consulting"]
    )

@pytest.fixture
def sample_csv_content():
    """Sample CSV content for testing"""
    return b"""Date,Description,Amount,Account
2024-01-01,Office supplies from Staples,-150.00,Business Checking
2024-01-02,Client payment received,5000.00,Business Checking
2024-01-03,Business lunch at restaurant,-85.50,Business Credit Card
2024-01-04,Software license renewal,-299.99,Business Credit Card"""

@pytest.fixture
def large_transaction_dataset():
    """Large dataset for performance testing"""
    transactions = []
    for i in range(1000):
        transactions.append(
            Transaction(
                date=f"2024-01-{(i % 30) + 1:02d}",
                description=f"Transaction {i}",
                amount=Decimal(str((-1 if i % 2 else 1) * (100 + i % 500))),
                currency="USD"
            )
        )
    return transactions

@pytest.fixture
def mock_llm_response():
    """Mock LLM response for testing"""
    return """
# Executive Summary

Based on analysis of 4 transactions totaling $5,535.49 in activity, the business shows strong profitability with an 89.3% gross margin.

## Key Findings
- Revenue: $5,000.00
- Expenses: $535.49
- Gross Profit: $4,464.51

## Strategic Recommendations
1. Continue current client engagement model
2. Monitor expense categorization for tax optimization
3. Consider scaling operations given strong margins
"""
```
      - supabase

  supabase-rest:
    image: postgrest/postgrest:v10.1.1
    depends_on:
      - supabase-db
    restart: unless-stopped
    environment:
      PGRST_DB_URI: postgresql://authenticator:${POSTGRES_PASSWORD}@supabase-db:5432/postgres
      PGRST_DB_SCHEMAS: ${PGRST_DB_SCHEMAS}
      PGRST_DB_ANON_ROLE: anon
      PGRST_JWT_SECRET: ${JWT_SECRET}
      PGRST_DB_USE_LEGACY_GUCS: "false"
    networks:
      - supabase

  supabase-realtime:
    image: supabase/realtime:v2.10.1
    depends_on:
      - supabase-db
    restart: unless-stopped
    environment:
      PORT: 4000
      DB_HOST: supabase-db
      DB_PORT: 5432
      DB_USER: supabase_admin
      DB_PASSWORD: ${POSTGRES_PASSWORD}
      DB_NAME: postgres
      DB_AFTER_CONNECT_QUERY: 'SET search_path TO _realtime'
      DB_ENC_KEY: supabaserealtime
      API_JWT_SECRET: ${JWT_SECRET}
      FLY_ALLOC_ID: fly123
      FLY_APP_NAME: realtime
      SECRET_KEY_BASE: UpNVntn3cDxHJpq99YMc1T1AQgQpc8kfYTuRgBiYa15BLrx8etQoXz3gZv1/u2oq
    networks:
      - supabase

  supabase-storage:
    image: supabase/storage-api:v0.40.4
    depends_on:
      - supabase-db
      - supabase-rest
    restart: unless-stopped
    environment:
      ANON_KEY: ${ANON_KEY}
      SERVICE_KEY: ${SERVICE_ROLE_KEY}
      POSTGREST_URL: http://supabase-rest:3000
      PGRST_JWT_SECRET: ${JWT_SECRET}
      DATABASE_URL: postgresql://supabase_storage_admin:${POSTGRES_PASSWORD}@supabase-db:5432/postgres
      FILE_SIZE_LIMIT: 52428800
      STORAGE_BACKEND: file
      FILE_STORAGE_BACKEND_PATH: /var/lib/storage
      TENANT_ID: stub
      REGION: stub
      GLOBAL_S3_BUCKET: stub
    volumes:
      - supabase_storage_data:/var/lib/storage
    networks:
      - supabase

  supabase-db:
    image: supabase/postgres:15.1.0.117
    restart: unless-stopped
    ports:
      - "5432:5432"
    environment:
      POSTGRES_HOST: /var/run/postgresql
      PGPORT: 5432
      POSTGRES_PORT: 5432
      PGPASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      PGDATABASE: postgres
      POSTGRES_DB: postgres
      POSTGRES_USER: postgres
      POSTGRES_INITDB_ARGS: "--lc-collate=C --lc-ctype=C"
    volumes:
      - supabase_db_data:/var/lib/postgresql/data
      - ./supabase/migrations:/docker-entrypoint-initdb.d
    networks:
      - supabase

volumes:
  supabase_db_data:
  supabase_storage_data:

networks:
  supabase:
    name: supabase
```

**Create GitHub Actions workflow (`.github/workflows/ci.yml`):**

```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: supabase/postgres:15.1.0.117
        env:
          POSTGRES_PASSWORD: postgres
          POSTGRES_DB: postgres
          POSTGRES_USER: postgres
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    
    - name: Set up Node.js for Supabase CLI
      uses: actions/setup-node@v3
      with:
        node-version: '18'
    
    - name: Install Supabase CLI
      run: |
        npm install -g supabase
    
    - name: Install system dependencies
      run: |
        sudo apt-get update
        sudo apt-get install -y tesseract-ocr tesseract-ocr-eng poppler-utils
    
    - name: Cache Python dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}
        restore-keys: |
          ${{ runner.os }}-pip-
    
    - name: Install Python dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest-cov
    
    - name: Start Supabase local stack
      run: |
        supabase init
        supabase start
      env:
        SUPABASE_ACCESS_TOKEN: ${{ secrets.SUPABASE_ACCESS_TOKEN }}
    
    - name: Run linting
      run: |
        black --check src/ tests/
        isort --check-only src/ tests/
        mypy src/
    
    - name: Run tests with coverage
      env:
        OPENAI_API_KEY: dummy-key-for-testing
        SUPABASE_URL: http://127.0.0.1:54321
        SUPABASE_ANON_KEY: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZS1kZW1vIiwicm9sZSI6ImFub24iLCJleHAiOjE5ODM4MTI5OTZ9.CRXP1A7WOeoJeXxjNni43kdQwgnWNReilDMblYTn_I0
        SUPABASE_SERVICE_KEY: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZS1kZW1vIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImV4cCI6MTk4MzgxMjk5Nn0.EGIM96RAZx35lJzdJsyH-qQwv8Hdp7fsn3W0YpN81IU
        DATABASE_URL: postgresql://postgres:postgres@localhost:54322/postgres
        VECS_CONNECTION_STRING: postgresql://postgres:postgres@localhost:54322/postgres
        STORAGE_PROVIDER: supabase
      run: |
        pytest --cov=src --cov-report=xml --cov-report=html --cov-fail-under=85
    
    - name: Upload coverage reports
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        fail_ci_if_error: true
    
    - name: Build Docker image
      run: |
        docker build -t ashworth-engine:test .
    
    - name: Test Docker image
      run: |
        docker run --rm --network host ashworth-engine:test python -c "import src.api.routes; print('Docker image test passed')"
    
    - name: Stop Supabase
      if: always()
      run: |
        supabase stop

  security:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    
    - name: Install security tools
      run: |
        pip install bandit safety
    
    - name: Run security checks
      run: |
        bandit -r src/ -f json -o bandit-report.json
        safety check -r requirements.txt --json --output safety-report.json
    
    - name: Upload security reports
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: security-reports
        path: |
          bandit-report.json
          safety-report.json

  integration:
    runs-on: ubuntu-latest
    needs: [test, security]
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v2
    
    - name: Login to Container Registry
      uses: docker/login-action@v2
      with:
        registry: ghcr.io
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
    
    - name: Build and push Docker image
      uses: docker/build-push-action@v4
      with:
        context: .
        push: true
        tags: |
          ghcr.io/${{ github.repository }}:latest
          ghcr.io/${{ github.repository }}:${{ github.sha }}
        cache-from: type=gha
        cache-to: type=gha,mode=max
```


### 4.9 Production Deployment with Supabase

**Deploy to hosted Supabase:**

```bash
# 1. Create production Supabase project
supabase projects create ashworth-engine-prod --org-id your-org-id \
  --db-password your-secure-production-password \
  --region us-east-1

# 2. Link local project to production
supabase link --project-ref your-production-project-ref

# 3. Push database migrations
supabase db push

# 4. Deploy edge functions (if any)
supabase functions deploy

# 5. Set production secrets
supabase secrets set OPENAI_API_KEY=your-production-openai-key
supabase secrets set LLM_PROVIDER=openai

# 6. Configure environment variables for production deployment
cat > .env.production << EOF
# Production Supabase Configuration
SUPABASE_URL=https://your-project-ref.supabase.co
SUPABASE_ANON_KEY=your-production-anon-key
SUPABASE_SERVICE_KEY=your-production-service-key

# Production Database
DATABASE_URL=postgresql://postgres.[PROJECT_REF]:[PASSWORD]@aws-0-[REGION].pooler.supabase.com:6543/postgres
VECS_CONNECTION_STRING=postgresql://postgres.[PROJECT_REF]:[PASSWORD]@aws-0-[REGION].pooler.supabase.com:5432/postgres

# Production Configuration
AE_ENV=production
LOG_LEVEL=INFO
STORAGE_PROVIDER=supabase
STORAGE_BUCKET=reports
CHARTS_BUCKET=charts

# Security
API_AUTH_KEY=your-production-api-auth-key
MAX_UPLOAD_SIZE=52428800
REPORT_RETENTION_DAYS=90
EOF
```

**Production Docker deployment:**

```bash
# Build production image
docker build -t ashworth-engine:production .

# Run with production configuration
docker run -d \
  --name ashworth-engine-prod \
  --env-file .env.production \
  -p 8000:8000 \
  --restart unless-stopped \
  ashworth-engine:production

# Verify deployment
curl -f http://localhost:8000/health
```

**Production monitoring setup:**

```bash
# Create monitoring configuration
cat > docker-compose.prod.yml << EOF
version: '3.8'

services:
  app:
    image: ashworth-engine:production
    environment:
      - SUPABASE_URL=https://your-project-ref.supabase.co
      - DATABASE_URL=postgresql://postgres.[PROJECT_REF]:[PASSWORD]@aws-0-[REGION].pooler.supabase.com:6543/postgres
      - AE_ENV=production
    ports:
      - "8000:8000"
    volumes:
      - ./logs:/app/logs
      - ./data:/app/data
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
        reservations:
          memory: 1G
          cpus: '0.5'
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro
    depends_on:
      - app
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards:ro
      - ./monitoring/grafana/datasources:/etc/grafana/provisioning/datasources:ro
    restart: unless-stopped

volumes:
  prometheus_data:
  grafana_data:
EOF
```

- [ ] All tests passing with coverage ≥85%
- [ ] Quality targets met (99.99% data accuracy, 100% categorization on test data)
- [ ] Docker application runs correctly in container environment
- [ ] API accessible and processes requests successfully in Docker
- [ ] No known critical bugs or memory leaks
- [ ] Security tests pass with no critical vulnerabilities
- [ ] Performance benchmarks meet requirements (1000 transactions <5 seconds)
- [ ] Documentation complete for deployment and troubleshooting
- [ ] CI/CD pipeline functional and passing
- [ ] Load testing shows acceptable concurrent request handling

## Next Steps

After Phase 4 completion, proceed to Phase 5: Launch & Future Enhancements.

## RACI Matrix

**Responsible:** Solo Developer  
**Accountable:** Solo Developer
**Consulted:** Security experts for vulnerability review, automated tools for testing
**Informed:** Stakeholders get test reports and deployment readiness confirmation