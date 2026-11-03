# MKUU Foundation Model

**African Intelligence. Global Impact.**

MKUU AI is an assistant designed for natural Tanzanian Kiswahili and English.

## Direction
- Use an existing pre-trained, open-weight instruction model as the base.
- No training or fine-tuning in this phase.
- Apply MKUU identity and language behavior at inference time.
- Expose an HTTP API for future Android/client integration.
- Keep model weights, secrets, and private data out of Git.

## Current foundation
This repository contains a FastAPI inference service, configurable Hugging Face model selection, MKUU system instructions, and Docker setup. This is an inference/API foundation, not a newly trained model.

## Run locally
1. Copy `.env.example` to `.env` and review the model ID and settings.
2. Install: `pip install -r requirements.txt`.
3. Start: `uvicorn app.main:app --host 0.0.0.0 --port 8000`.
4. Check `GET /health`; API docs are at `/docs`.
5. Send `POST /v1/chat/completions` with a `messages` array.

Inference needs a computer/server with sufficient RAM/VRAM. GitHub or an Android phone alone does not host the model. The first model load downloads weights from Hugging Face. Review the chosen model's license and hardware requirements.

## Security
Set `MKUU_API_KEY` before public exposure, use HTTPS, and never commit `.env` or access tokens. Add rate limiting and monitoring before public launch.

## Scope
No training, fine-tuning, benchmark, or model-quality claim is made in this phase.
