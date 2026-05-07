FROM python:3.10-slim-bullseye

# Set working directory
WORKDIR /app

# Install system dependencies including git-lfs
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    git \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Download all LFS model files from GitHub
RUN if [ -f ".gitattributes" ] && grep -q "\.pth.*lfs" .gitattributes; then \
        echo "Downloading model files from GitHub LFS..."; \
        curl -L "https://github.com/juliodiaz0209/TORTUOSITY/raw/main/final_model%20(11).pth" -o "final_model (11).pth" || echo "Failed: final_model (11).pth"; \
        curl -L "https://github.com/juliodiaz0209/TORTUOSITY/raw/main/final_model_tarsus_improved.pth" -o "final_model_tarsus_improved.pth" || echo "Failed: final_model_tarsus_improved.pth"; \
        curl -L "https://github.com/juliodiaz0209/TORTUOSITY/raw/main/final_model_tarsus.pth" -o "final_model_tarsus.pth" || echo "Failed: final_model_tarsus.pth"; \
        curl -L "https://github.com/juliodiaz0209/TORTUOSITY/raw/main/final_model_improved_fixed.pth" -o "final_model_improved_fixed.pth" || echo "Failed: final_model_improved_fixed.pth"; \
        curl -L "https://github.com/juliodiaz0209/TORTUOSITY/raw/main/final_model_tarsus_improved%20(6).pth" -o "final_model_tarsus_improved (6).pth" || echo "Failed: final_model_tarsus_improved (6).pth"; \
        curl -L "https://github.com/juliodiaz02091999/TORTUOSITY/raw/main/best_model%20(13).pth" -o "best_model (13).pth" || echo "Failed: best_model (13).pth"; \
        curl -L "https://github.com/juliodiaz02091999/TORTUOSITY/raw/main/best_model%20(17).pth" -o "best_model (17).pth" || echo "Failed: best_model (17).pth"; \
        curl -L "https://github.com/juliodiaz02091999/TORTUOSITY/raw/main/best_model%20(18).pth" -o "best_model (18).pth" || echo "Failed: best_model (18).pth"; \
        curl -L "https://github.com/juliodiaz02091999/TORTUOSITY/raw/main/best_model%20(20).pth" -o "best_model (20).pth" || echo "Failed: best_model (20).pth"; \
        curl -L "https://github.com/juliodiaz02091999/TORTUOSITY/raw/main/best_model%20(32).pth" -o "best_model (32).pth" || echo "Failed: best_model (32).pth"; \
        ls -la *.pth; \
    fi

# Save Mask2Former config + processor locally (no large weights needed — our checkpoint provides them)
RUN python -c "\
import os; os.makedirs('/app/mask2former_config', exist_ok=True); \
from transformers import Mask2FormerConfig, AutoImageProcessor; \
print('Downloading Mask2Former config...'); \
cfg = Mask2FormerConfig.from_pretrained('facebook/mask2former-swin-small-cityscapes-instance'); \
cfg.save_pretrained('/app/mask2former_config'); \
print('Downloading processor...'); \
proc = AutoImageProcessor.from_pretrained('facebook/mask2former-swin-small-cityscapes-instance', use_fast=False); \
proc.save_pretrained('/app/mask2former_config'); \
print('Saved to /app/mask2former_config') \
"

# Create necessary directories
RUN mkdir -p temp results static

# Debug: List files to verify everything was copied and downloaded
RUN echo "All files in app directory:" && ls -la
RUN echo "Model files downloaded:" && ls -la *.pth || echo "No .pth files found"

# Expose port (will be overridden by Cloud Run)
EXPOSE 8000

# Health check (use fixed port since Cloud Run assigns 8000)
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD uvicorn main:app --host 0.0.0.0 --port $PORT 