# GasWaterAI Engine -- production container.
#
# Pinned to the 3.11 minor line (see .python-version for the exact local-dev
# patch) so security patches to the base image keep flowing automatically;
# see docs/deployment.md for why an exact patch pin was not used here.
FROM python:3.11-slim

# pytesseract (requirements.txt) only wraps the `tesseract` CLI -- it is NOT
# a Python package and is never installed by pip. Without this system
# package, OCR text extraction (app/plan_analysis/text.py's OCR fallback)
# fails at runtime on any platform that does not happen to ship it. Only
# English and German language data are installed -- the only two this
# engine's plans use (see docs/architecture.md).
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-eng \
        tesseract-ocr-deu \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code.
COPY app/ app/

# The ONLY file under data/ the running service ever reads (app/detector/
# infer.py) -- confirmed by repo-wide search; everything else under data/
# (dev datasets, benchmark crops, annotation tooling output) is deliberately
# NOT copied into the image (see .dockerignore).
COPY data/dev_plans_v04/detector_v1/model.joblib data/dev_plans_v04/detector_v1/model.joblib
COPY data/dev_plans_v04/detector_v1/training_config.json data/dev_plans_v04/detector_v1/training_config.json

EXPOSE 8000

# No hardcoded secret: GASWATERAI_VISION_API_KEY, GASWATERAI_MULTI_AGENT_API_KEY,
# BASE44_AI_GATEWAY_URL and BASE44_AI_GATEWAY_API_KEY are all read from the
# environment at runtime (see app/main.py / app/multi_agent_v1/) -- inject
# them via the hosting platform's secret/environment-variable store, never
# via this image or a committed .env file.
#
# Binds to $PORT when the platform provides one (e.g. Render), else 8000.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
