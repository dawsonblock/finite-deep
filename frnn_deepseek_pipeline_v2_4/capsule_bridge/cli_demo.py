
#!/usr/bin/env python3
import argparse,json,os,sys
from capsule_bridge.deepseek_temporal_core import DEFAULT_CFG,TemporalCore,build_model,build_x_t,context_prefix,deepseek_chat

def main():
    a=argparse.ArgumentParser()
    a.add_argument("--model",default=os.getenv("DEEPSEEK_MODEL","deepseek-chat"))
    a.add_argument("--api-url",default=os.getenv("DEEPSEEK_URL","https://api.deepseek.com/chat/completions"))
    a.add_argument("--prompt",default="Summarize the last action.")
    a.add_argument("--tool-json",default="{}");a.add_argument("--steps",type=int,default=3)
    a.add_argument("--device",default=None);a.add_argument("--timeout",type=float,default=float(os.getenv("DEEPSEEK_TIMEOUT_S","60")))
    args=a.parse_args()
    if not os.getenv("DEEPSEEK_API_KEY"): print("error: DEEPSEEK_API_KEY not set",file=sys.stderr);sys.exit(2)
    frnn=build_model(DEFAULT_CFG); core=TemporalCore(frnn,device=args.device)
    meta=json.loads(args.tool_json)
    for i in range(args.steps):
        x=build_x_t(f"tick {i}: {args.prompt}",meta,di=DEFAULT_CFG["Di"]); vec,m=core.tick(x)
    pref=context_prefix(vec,m); msgs=[{"role":"user","content":args.prompt}]
    try:
        out=deepseek_chat(msgs,context_prefix_str=pref,model=args.model,api_url=args.api_url,timeout_s=args.timeout);print(out)
    except Exception as e: print(f"[DeepSeek error] {e}",file=sys.stderr);sys.exit(2)
if __name__=="__main__": main()
