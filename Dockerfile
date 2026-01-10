FROM python:3.10-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
ENV POETRY_HOME=/opt/poetry
ENV PATH="$POETRY_HOME/bin:$PATH"

RUN curl -sSL https://install.python-poetry.org | python3 -
RUN poetry config virtualenvs.create false

# Workdir
WORKDIR /app

# Copy dependency files
COPY pyproject.toml poetry.lock* /app/

# Install dependencies ONLY first (layer cache)
RUN poetry install --no-interaction --no-ansi --no-root

# Copy project source
COPY . /app

# INSTALL THE PROJECT ITSELF
RUN poetry install --no-interaction --no-ansi

# Create non-root user
RUN useradd -ms /bin/bash llmuser && chown -R llmuser:llmuser /app
USER llmuser

# ENTRYPOINT
ENTRYPOINT ["python", "-m", "llm_eval.cli.main"]
