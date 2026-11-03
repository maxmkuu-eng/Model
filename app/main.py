from __future__ import annotations

import os
import secrets
import threading
from pathlib import Path
from typing import Literal

import torch
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()
ROOT = Path(__file__).resolve().parents[1]
app = FastAPI(title="MKUU Foundation Model API", version="0.2.0")
_model = None
_tokenizer = None
_model_lock = threading.Lock()
_model_loading = False
_model_error: str | None = None
_api_key_header = APIKeyHeader(name="Authorization", auto_error=False)


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=20000)


class ChatRequest(BaseModel):
    model: str = "mkuu-foundation"
    messages: list[Message] = Field(min_length=1, max_length=40)
    max_tokens: int | None = Field(default=None, ge=1, le=4096)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)


def verify_api_key(authorization: str | None = Security(_api_key_header)) -> None:
    expected = os.getenv("MKUU_API_KEY", "").strip()
    if not expected:
        if os.getenv("MKUU_ENV", "development").strip().lower() == "production":
            raise HTTPException(status_code=503, detail="MKUU_API_KEY must be set in production")
        return

    supplied = (authorization or "").strip()
    if supplied.lower().startswith("bearer "):
        supplied = supplied[7:].strip()

    if not supplied or not secrets.compare_digest(supplied, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def load_model():
    global _model, _tokenizer, _model_loading, _model_error
    if _model is not None:
        return _tokenizer, _model

    with _model_lock:
        if _model is None:
            _model_loading = True
            try:
                model_id = os.getenv("MKUU_BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct").strip()
                if not model_id:
                    raise RuntimeError("MKUU_BASE_MODEL is not configured")

                tokenizer = AutoTokenizer.from_pretrained(model_id)
                device = os.getenv("MKUU_DEVICE", "auto").strip().lower()
                kwargs = {"torch_dtype": "auto"}
                if device == "auto":
                    kwargs["device_map"] = "auto"

                model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
                if device != "auto":
                    model.to(device)
                model.eval()

                _tokenizer, _model = tokenizer, model
                _model_error = None
            except Exception as exc:
                # Keep the service alive and expose only the exception type, never secrets.
                _model_error = type(exc).__name__
                raise HTTPException(
                    status_code=503,
                    detail=f"MKUU base model could not be loaded ({_model_error})",
                ) from exc
            finally:
                _model_loading = False

    return _tokenizer, _model


def system_prompt() -> str:
    path = ROOT / "SYSTEM_PROMPT.md"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return "You are MKUU AI. Reply naturally in Tanzanian Kiswahili or English. Be helpful and honest."


@app.get("/")
def root():
    return {"name": "MKUU Foundation Model API", "health": "/health", "model_health": "/health/model"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "mkuu-foundation",
        "model_loaded": _model is not None,
        "model_loading": _model_loading,
        "model_error": _model_error,
        "base_model": os.getenv("MKUU_BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"),
        "api_key_configured": bool(os.getenv("MKUU_API_KEY", "").strip()),
        "environment": os.getenv("MKUU_ENV", "development").strip().lower(),
    }


@app.get("/health/model")
def model_health():
    if _model is not None:
        status = "loaded"
    elif _model_loading:
        status = "loading"
    elif _model_error:
        status = "error"
    else:
        status = "not_loaded"

    return {
        "status": status,
        "model_loaded": _model is not None,
        "model_loading": _model_loading,
        "model_error": _model_error,
        "base_model": os.getenv("MKUU_BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"),
    }


@app.post("/v1/chat/completions", dependencies=[Depends(verify_api_key)])
def chat_completions(request: ChatRequest):
    tokenizer, model = load_model()
    messages = [{"role": "system", "content": system_prompt()}]
    messages.extend(m.model_dump() for m in request.messages if m.role != "system")

    if not hasattr(tokenizer, "apply_chat_template"):
        raise HTTPException(status_code=500, detail="Selected tokenizer has no chat template")

    try:
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Could not prepare the conversation for the model") from exc

    device = model.get_input_embeddings().weight.device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    max_tokens = request.max_tokens if request.max_tokens is not None else int(os.getenv("MKUU_MAX_NEW_TOKENS", "512"))
    temperature = request.temperature if request.temperature is not None else float(os.getenv("MKUU_TEMPERATURE", "0.7"))
    top_p = request.top_p if request.top_p is not None else float(os.getenv("MKUU_TOP_P", "0.9"))

    generation = {
        "max_new_tokens": max_tokens,
        "do_sample": temperature > 0,
    }
    if tokenizer.eos_token_id is not None:
        generation["pad_token_id"] = tokenizer.eos_token_id
    if temperature > 0:
        generation.update({"temperature": temperature, "top_p": top_p})

    try:
        with torch.inference_mode():
            output = model.generate(**inputs, **generation)
        answer = tokenizer.decode(
            output[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
        ).strip()
    except Exception as exc:
        raise HTTPException(status_code=502, detail="MKUU model generation failed") from exc

    if not answer:
        raise HTTPException(status_code=502, detail="The model returned an empty response")

    return {
        "id": "mkuu-chat-completion",
        "object": "chat.completion",
        "model": request.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": answer},
            "finish_reason": "stop",
        }],
    }
