# MKUU API Runtime Deployment

## What this adds

Docker Compose runs the FastAPI service as a persistent container and stores downloaded Hugging Face model files in a named volume. The model is downloaded when the first chat request loads it; it is not bundled into the Git repository or Docker image.

## Requirements

- A Linux host or cloud VM with Docker Engine and the Docker Compose plugin.
- Sufficient RAM/VRAM for the selected model. The configured 1.5B model can still require several GB of memory; actual needs depend on precision, runtime, and request length.
- Outbound internet access for the initial model download.
- A stable public URL and HTTPS reverse proxy before connecting a public Android client.

## Start

1. Clone this repository onto the host.
2. Copy `.env.example` to `.env`.
3. Set a long, random `MKUU_API_KEY` in `.env`.
4. Review `MKUU_BASE_MODEL` and the model's license.
5. Run `docker compose up -d --build`.
6. Check logs with `docker compose logs -f mkuu-api`.
7. Check health at `http://HOST:8000/health`.

The Compose service sets `MKUU_ENV=production`; it refuses API requests unless `MKUU_API_KEY` is configured. Send the key as an HTTP Bearer token when calling `POST /v1/chat/completions`.

## Stop and update

- Stop: `docker compose down`
- Update code: `git pull` then `docker compose up -d --build`
- The named `mkuu-model-cache` volume keeps downloaded model files when the container is recreated. Do not remove the volume unless you intend to download the model again.

## Important limitations

This repository does not provision a server, GPU, domain, HTTPS certificate, or free hosting. A GitHub push does not start inference. Public deployment needs a secured host and may incur compute, bandwidth, and storage costs. Do not expose port 8000 directly to the public internet without HTTPS, access controls, and rate limiting.
