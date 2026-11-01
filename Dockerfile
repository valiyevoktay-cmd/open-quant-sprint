# Image for the official holdout and forward runs: python -m src.audit ... --container oqs-sandbox
# It holds the harness and the pinned dependencies, never the data files.
FROM python:3.11-slim
WORKDIR /app
COPY requirements.lock .
RUN pip install --no-cache-dir -r requirements.lock
COPY src/ src/
RUN mkdir /work
ENV PYTHONPATH=/app PYTHONDONTWRITEBYTECODE=1
USER nobody
