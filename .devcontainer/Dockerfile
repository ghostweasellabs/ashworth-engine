# Development container for Ashworth Engine
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js and Yarn
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && npm install -g yarn

# Install uv for Python package management
RUN pip install uv

# Set up workspace
WORKDIR /workspace

# Copy Python dependencies and install
COPY pyproject.toml README.md ./
RUN uv venv && uv sync

# Install pre-commit for code quality
RUN uv run pip install pre-commit

# Set up environment
ENV PYTHONPATH=/workspace
ENV PATH="/workspace/.venv/bin:$PATH"

# Keep container running
CMD ["sleep", "infinity"]