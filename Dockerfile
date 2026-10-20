FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY duecall ./duecall
# Bake VAD weights into the image so cold starts don't download them.
RUN python -m livekit.agents download-files
RUN useradd -m app && mkdir -p /app/data && chown -R app /app
USER app
# The worker serves a health check on $PORT (Cloud Run sets it) and keeps an
# outbound websocket open to LiveKit, so it must not be CPU-throttled.
CMD ["python", "-m", "duecall.main", "start"]
