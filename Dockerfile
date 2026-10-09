# SkyStay simulator image: builds the data, back-test, model and sensitivity outputs at build time,
# then serves the Streamlit app on port 8501. Video and deck builds are not part of the image.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SKYSTAY_ROOT=/app

RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 curl \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --create-home --uid 1000 app

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip && pip install .

COPY --chown=app:app . .
USER app

# Regenerate everything from data/raw (real mode) or the seeded synthetic generator.
RUN skystay setup && skystay data && skystay backtest && skystay train && skystay sensitivity

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD curl -fsS http://localhost:8501/_stcore/health || exit 1
CMD ["skystay", "app", "--port", "8501", "--headless"]
