# Streamlit app for the explainable fire-safety RAG demo.
# Runs in retrieval-only mode unless OPENAI_API_KEY is provided at runtime:
#   docker build -t firerag .
#   docker run -p 8501:8501 --env-file .env firerag
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# CPU-only torch keeps the image small; then the rest of the pinned requirements.
COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1 \
 && pip install -r requirements.txt

COPY pyproject.toml README.md ./
COPY src ./src
COPY taxonomy ./taxonomy
COPY scripts ./scripts
COPY app.py ./
COPY .streamlit ./.streamlit
RUN pip install --no-deps .

# Fetch the official PDFs and build the index at build time so the container
# starts ready (the PDFs stay inside the image, never in the repository).
RUN python scripts/download_data.py && python -m firerag.ingest.build_index

RUN useradd --create-home appuser && mkdir -p /app/logs && chown -R appuser /app
USER appuser

EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
