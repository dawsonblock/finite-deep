
import os, uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, List
from capsule_bridge.deepseek_temporal_core import (
    build_model, TemporalCore, build_x_t, context_prefix, deepseek_chat, DEFAULT_CFG
)

class TickIn(BaseModel):
    text: str
    meta: Dict[str, Any] = Field(default_factory=dict)

class ChatIn(BaseModel):
    messages: List[Dict[str, str]]
    extra: Dict[str, Any] = Field(default_factory=dict)

app = FastAPI(title="FRNN-DeepSeek Pipeline", version="0.2.4")
core = None

@app.on_event("startup")
def _startup():
    global core
    if not os.getenv("DEEPSEEK_API_KEY"):
        raise RuntimeError("DEEPSEEK_API_KEY not set")
    frnn = build_model(DEFAULT_CFG)
    core = TemporalCore(frnn)

@app.get("/healthz")
def healthz():
    return {"ok": True, "model": "ConfigB", "K": DEFAULT_CFG["K"], "bank": DEFAULT_CFG["bank"]}

@app.post("/tick")
def tick(inp: TickIn):
    x_t = build_x_t(inp.text, inp.meta, di=DEFAULT_CFG["Di"])
    ctx_vec, top_modes = core.tick(x_t)
    return {"ok": True, "top_modes": top_modes, "ctx_len": int(ctx_vec.numel())}

@app.post("/chat")
def chat(inp: ChatIn):
    if not inp.messages:
        raise HTTPException(status_code=400, detail="messages list cannot be empty")
    x_t = build_x_t(inp.messages[-1]["content"], {"latency_ms":0,"tokens_last":0}, di=DEFAULT_CFG["Di"])
    ctx_vec, top_modes = core.tick(x_t)
    prefix = context_prefix(ctx_vec, top_modes)
    try:
        out = deepseek_chat(inp.messages, context_prefix_str=prefix, extra=inp.extra)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
    return {"reply": out, "modes": top_modes}

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8080, workers=1)
