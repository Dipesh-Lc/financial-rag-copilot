FROM python:3.11-slim

# System deps needed to build some wheels (lxml, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Non-root user required by HF Docker Spaces (uid 1000)
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Point HF / sentence-transformers caches to a location baked into the image
# (under $HOME which is writable at runtime too)
ENV HF_HOME=/home/user/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/home/user/.cache/sentence-transformers

WORKDIR /home/user/app

# Install Python deps first so the layer is cached independently of source changes
COPY --chown=user requirements.txt .

# Install CPU-only torch to keep the image smaller; other packages follow
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy project source (seed corpus included; raw/processed/vectorstore excluded via .dockerignore)
COPY --chown=user . .

# Pre-download the embedding model so runtime never calls HF Hub
RUN python -c "from app.embeddings.hf_embeddings import get_embeddings; get_embeddings()"

# Runtime env — index and Chroma write to the writable WORKDIR tree
ENV HOST=0.0.0.0 \
    PORT=7860 \
    APP_ENV=production

EXPOSE 7860

CMD ["python", "-m", "app.main"]
