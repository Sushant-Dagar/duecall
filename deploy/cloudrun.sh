#!/usr/bin/env bash
# Deploy the voice worker to Google Cloud Run.
# Usage: PROJECT=my-gcp-project REGION=asia-south1 ./deploy/cloudrun.sh
set -euo pipefail
: "${PROJECT:?set PROJECT}"; REGION="${REGION:-asia-south1}"; SERVICE="${SERVICE:-duecall-agent}"
IMAGE="$REGION-docker.pkg.dev/$PROJECT/duecall/$SERVICE:$(git rev-parse --short HEAD 2>/dev/null || date +%s)"

gcloud artifacts repositories describe duecall --location "$REGION" --project "$PROJECT" >/dev/null 2>&1 || \
  gcloud artifacts repositories create duecall --repository-format docker --location "$REGION" --project "$PROJECT"
gcloud builds submit --tag "$IMAGE" --project "$PROJECT"

# Secrets live in Secret Manager, e.g.:
#   printf %s "$LIVEKIT_API_SECRET" | gcloud secrets create LIVEKIT_API_SECRET --data-file=- --project "$PROJECT"
gcloud run deploy "$SERVICE" --image "$IMAGE" --region "$REGION" --project "$PROJECT" \
  --no-cpu-throttling --min-instances 1 --max-instances 3 --cpu 2 --memory 2Gi \
  --no-allow-unauthenticated \
  --set-env-vars "LIVEKIT_URL=${LIVEKIT_URL:?},LLM_PROVIDER=${LLM_PROVIDER:-google},STT_PROVIDER=${STT_PROVIDER:-deepgram},TTS_PROVIDER=${TTS_PROVIDER:-cartesia},PROMPT_VERSION=${PROMPT_VERSION:-v2}" \
  --set-secrets "LIVEKIT_API_KEY=LIVEKIT_API_KEY:latest,LIVEKIT_API_SECRET=LIVEKIT_API_SECRET:latest,SIP_OUTBOUND_TRUNK_ID=SIP_OUTBOUND_TRUNK_ID:latest,DEEPGRAM_API_KEY=DEEPGRAM_API_KEY:latest,GOOGLE_API_KEY=GOOGLE_API_KEY:latest,CARTESIA_API_KEY=CARTESIA_API_KEY:latest"
