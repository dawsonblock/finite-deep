
import os, json
from capsule_bridge.deepseek_temporal_core import (
    build_model, TemporalCore, build_x_t, context_prefix, deepseek_chat, DEFAULT_CFG
)

PROMPT = "Summarize user intent and propose next step."

def main():
    if not os.getenv("DEEPSEEK_API_KEY"):
        raise RuntimeError("DEEPSEEK_API_KEY missing")
    frnn = build_model(DEFAULT_CFG)
    core = TemporalCore(frnn)

    events = [
        {"user": "What did we do last time?", "meta": {"latency_ms": 110, "tokens_last": 320}},
        {"user": "We need a concrete plan for data ingestion", "meta": {"latency_ms": 95, "tokens_last": 280}},
        {"user": "Draft an outline with bullet points", "meta": {"latency_ms": 88, "tokens_last": 250}},
    ]

    for e in events:
        x_t = build_x_t(e["user"], e.get("meta", {}), di=DEFAULT_CFG["Di"])
        ctx_vec, top_modes = core.tick(x_t)

    prefix = context_prefix(ctx_vec, top_modes)
    reply = deepseek_chat([{"role": "user", "content": PROMPT}], context_prefix_str=prefix)
    print("\n=== DeepSeek ===\n" + reply + "\n")

if __name__ == "__main__":
    main()
