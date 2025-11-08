
# ⚙️ FRNN Path B — DeepSeek Ready (v0.2.4)
Fully hardened deterministic FRNN Path‑B core + DeepSeek bridge.

## Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

## Env
- `DEEPSEEK_API_KEY` (required)
- `DEEPSEEK_URL` (optional, default: https://api.deepseek.com/chat/completions)
- `DEEPSEEK_MODEL` (optional, default: deepseek-chat)
- `DEEPSEEK_TIMEOUT_S` (optional, default: 60)

## Quick demo
```bash
export DEEPSEEK_API_KEY=sk-...
python capsule_bridge/cli_demo.py --prompt "Plan next step" --steps 3
```

## Determinism
- TF32 disabled
- Eval forwards bit‑exact

---

## Standalone FRNN→DeepSeek Pipeline

### CLI demo
```bash
python pipeline.py
```

### REST server
```bash
pip install fastapi uvicorn
export DEEPSEEK_API_KEY=sk-...
python server.py
```

### Docker
```bash
docker build -t frnn-deepseek:0.2.4 .
docker run -p 8080:8080 -e DEEPSEEK_API_KEY=sk-... frnn-deepseek:0.2.4
```

Endpoints:
- `GET /healthz` — readiness
- `POST /tick` — advance FRNN state
- `POST /chat` — context-aware DeepSeek reply
