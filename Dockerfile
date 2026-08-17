FROM python:3.11-slim

WORKDIR /app

# Install deps first so this layer is cached unless requirements.txt changes
COPY requirements.txt .
# --timeout/--retries: some networks (campus/corporate proxies especially)
# kill long-lived HTTPS connections after a few minutes, which breaks large
# downloads like torch. This gives pip more patience before giving up.
RUN pip install --no-cache-dir --timeout 1000 --retries 5 -r requirements.txt

# App code
COPY src/ ./src/

# Bundle the pre-built data (chunks, embeddings, Chroma index) into the
# image. Baking it in is simpler than wiring up a cloud persistent volume
# for deployment. Locally, docker-compose.yml mounts data/ as a read-only
# volume on top of this, which overrides the copy below for fast local
# iteration without rebuilding the image every time you regenerate the
# index. For a standalone `docker build` (e.g. what a cloud platform does),
# this COPY is what actually gets served.
COPY data/ ./data/

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]