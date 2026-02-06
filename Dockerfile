FROM python:3.12-slim

WORKDIR /app

# System deps (if needed for lxml, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libxml2-dev libxslt1-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1
ENV LLM_BACKEND=
ENV LLM_ENDPOINT=
ENV LLM_MODEL=llama3
ENV LLM_API_KEY=
ENV SCRAPY_BIN=scrapy

# Default ports: API 8000, Streamlit 8501
EXPOSE 8000 8501

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port 8000 & streamlit run ui/app.py --server.port 8501 --server.address 0.0.0.0"]
