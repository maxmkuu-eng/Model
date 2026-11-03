from __future__ import annotations

import os
import secrets
import threading
from pathlib import Path
from typing import Literal

import torch
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.responses import HTMLResponse
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


def _preload_model_in_background() -> None:
    """Start loading Qwen as soon as the service starts, without blocking health checks."""
    try:
        load_model()
    except HTTPException:
        # The exact failure type is exposed through /health/model; keep API alive.
        pass


@app.on_event("startup")
def start_model_preload() -> None:
    threading.Thread(
        target=_preload_model_in_background,
        name="mkuu-model-preload",
        daemon=True,
    ).start()


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


@app.get("/chat", response_class=HTMLResponse)
def chat_page():
    return HTMLResponse(r'''<!doctype html>
<html lang="sw">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="theme-color" content="#ffffff">
<title>MKUU AI — Chat</title>
<style>
:root{font-family:Inter,system-ui,-apple-system,"Segoe UI",sans-serif;color:#172033;background:#f7f9fc}
*{box-sizing:border-box}body{margin:0;min-height:100vh;display:flex;flex-direction:column}
header{padding:15px 18px;background:#fff;border-bottom:1px solid #e7ebf2;display:flex;align-items:center;gap:12px}
.logo{width:38px;height:38px;border-radius:12px;background:linear-gradient(145deg,#0d4e91,#63b9ef);display:grid;place-items:center;color:white;font-weight:800}
h1{font-size:17px;margin:0}header small{color:#718096}
main{width:min(850px,100%);margin:0 auto;flex:1;display:flex;flex-direction:column;padding:18px 14px 12px;min-height:0}
#chat{flex:1;overflow-y:auto;display:flex;flex-direction:column;gap:14px;padding:8px 0 20px}
.msg{max-width:88%;padding:12px 15px;border-radius:17px;white-space:pre-wrap;overflow-wrap:anywhere;line-height:1.5}
.user{align-self:flex-end;background:#dceeff;border-bottom-right-radius:5px}
.assistant{align-self:flex-start;background:#fff;border:1px solid #e8edf4;border-bottom-left-radius:5px}
#empty{text-align:center;color:#64748b;margin:auto;padding:35px 10px}
#empty strong{display:block;color:#1e3a5f;font-size:25px;margin-bottom:8px}
.controls{background:#fff;border:1px solid #e2e8f0;border-radius:18px;padding:10px;box-shadow:0 5px 24px #1c35500d}
#key{width:100%;border:0;border-bottom:1px solid #edf0f5;padding:8px 5px;margin-bottom:8px;outline:none;font-size:13px}
.composer{display:flex;gap:8px;align-items:flex-end}
textarea{flex:1;resize:none;min-height:46px;max-height:140px;border:0;outline:none;padding:12px 8px;font:inherit}
button{border:0;border-radius:13px;background:#145da0;color:white;padding:12px 19px;font-weight:700;cursor:pointer}
button:disabled{opacity:.55;cursor:wait}
.note{font-size:11px;color:#8491a3;padding:7px 3px 0}
#error{color:#b42318;font-size:13px;padding:8px 3px;display:none}
@media(min-width:700px){main{padding-top:28px}.msg{max-width:75%}}
</style>
</head>
<body>
<header><div class="logo">M</div><div><h1>MKUU AI</h1><small>MKUU Foundation Model</small></div></header>
<main>
<section id="chat" aria-live="polite">
<div id="empty"><strong>Karibu MKUU AI</strong>Andika ujumbe wako hapa chini tuanze mazungumzo.</div>
</section>
<div class="controls">
<input id="key" type="password" autocomplete="off" placeholder="API key (weka hapa ikiwa inahitajika)">
<div class="composer">
<textarea id="message" rows="1" placeholder="Andika ujumbe..." aria-label="Ujumbe wako"></textarea>
<button id="send">Tuma</button>
</div>
<div class="note">API key haitahifadhiwa kwenye simu. Mazungumzo haya yanatumwa moja kwa moja kwa MKUU Model.</div>
<div id="error" role="alert"></div>
</div>
</main>
<script>
const chat=document.getElementById('chat');
const input=document.getElementById('message');
const keyInput=document.getElementById('key');
const send=document.getElementById('send');
const errorBox=document.getElementById('error');
const history=[];
function addMessage(role,text){
 const empty=document.getElementById('empty'); if(empty) empty.remove();
 const node=document.createElement('div');
 node.className='msg '+(role==='user'?'user':'assistant');
 node.textContent=text;
 chat.appendChild(node); chat.scrollTop=chat.scrollHeight;
 return node;
}
async function submitMessage(){
 const text=input.value.trim(); if(!text||send.disabled)return;
 errorBox.style.display='none'; errorBox.textContent='';
 addMessage('user',text); history.push({role:'user',content:text});
 input.value=''; input.style.height='auto'; send.disabled=true; send.textContent='...';
 const waiting=addMessage('assistant','MKUU anafikiria...');
 try{
  const headers={'Content-Type':'application/json'};
  const apiKey=keyInput.value.trim(); if(apiKey)headers.Authorization='Bearer '+apiKey;
  const response=await fetch('/v1/chat/completions',{method:'POST',headers,body:JSON.stringify({model:'mkuu-foundation',messages:history,max_tokens:512,temperature:0.7})});
  const data=await response.json().catch(()=>({}));
  if(!response.ok)throw new Error(data.detail||('Ombi limeshindikana (HTTP '+response.status+')'));
  const answer=data.choices&&data.choices[0]&&data.choices[0].message&&data.choices[0].message.content;
  if(!answer)throw new Error('Model haikurudisha jibu.');
  waiting.textContent=answer; history.push({role:'assistant',content:answer});
 }catch(err){
  waiting.textContent='Imeshindikana kupata jibu.';
  errorBox.textContent=err.message;errorBox.style.display='block';
  history.pop();
 }finally{send.disabled=false;send.textContent='Tuma';input.focus();chat.scrollTop=chat.scrollHeight;}
}
send.addEventListener('click',submitMessage);
input.addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();submitMessage();}});
input.addEventListener('input',()=>{input.style.height='auto';input.style.height=Math.min(input.scrollHeight,140)+'px';});
</script>
</body></html>''')


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
