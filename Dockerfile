FROM python:3.11-slim

# Install build dependencies for llama-cpp-python
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download the model from HuggingFace at build time
ARG HF_TOKEN
RUN mkdir -p models && \
    python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='bartowski/Llama-3.2-3B-Instruct-GGUF', filename='Llama-3.2-3B-Instruct-Q4_K_M.gguf', local_dir='models', token='$HF_TOKEN')"

# Copy application code
COPY src/ src/

ENV MODEL_PATH=models/Llama-3.2-3B-Instruct-Q4_K_M.gguf

CMD ["python", "-m", "src.app"]
