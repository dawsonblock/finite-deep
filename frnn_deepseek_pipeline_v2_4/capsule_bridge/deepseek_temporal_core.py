
import os,time,base64,json,torch,requests,numpy as np
from typing import List,Dict,Tuple,Optional
from frnn_path_b import FRNNPathB

DEFAULT_CFG=dict(Di=128,Do=64,K=512,Dm=256,H=512,bank=128,stickiness=0.15,ema_decay=0.99,bank_scale=4.0,use_gumbel=False,use_bank=True)
DEFAULT_URL=os.getenv("DEEPSEEK_URL","https://api.deepseek.com/chat/completions")
DEFAULT_MODEL=os.getenv("DEEPSEEK_MODEL","deepseek-chat")
DEFAULT_TIMEOUT=float(os.getenv("DEEPSEEK_TIMEOUT_S","60"))

class TemporalCore:
    def __init__(self,frnn:FRNNPathB,device:Optional[str]=None):
        self.frnn=frnn;self.device=device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.frnn.to(self.device).eval();self.prev_mode=None
    @torch.no_grad()
    def tick(self,x_t:torch.Tensor)->Tuple[torch.Tensor,List[int]]:
        x_t=x_t.to(self.device).view(1,1,-1);y,m=self.frnn(x_t,prev_mode=self.prev_mode)
        self.prev_mode=m[0,0].detach();probs=self.prev_mode.float()
        k=min(5,probs.numel());return y[0,0].cpu(),torch.topk(probs,k=k).indices.tolist()

def build_model(cfg=None)->FRNNPathB:
    c={**DEFAULT_CFG,**(cfg or {})}
    return FRNNPathB(c["Di"],c["Do"],c["K"],c["Dm"],c["H"],c["bank"],stickiness=c["stickiness"],ema_decay=c["ema_decay"],
                     bank_scale=c["bank_scale"],use_gumbel=c["use_gumbel"],use_bank=c["use_bank"])

def build_x_t(text:str,meta:Dict,di:int=128)->torch.Tensor:
    v=np.zeros(di,np.float32);n=min(len(text),2048);v[0]=n/2048.0
    v[1]=sum(c.isupper() for c in text)/max(1,len(text));v[2]=text.count("?")/max(1,len(text))
    v[3]=text.count("!")/max(1,len(text));v[4]=text.count(".")/max(1,len(text));v[5]=text.count(",")/max(1,len(text))
    v[64]=float(meta.get("latency_ms",0))/1000.0;v[65]=float(meta.get("tokens_last",0))/4096.0
    v[66]=float(meta.get("success",1));v[67]=float(meta.get("errors",0));return torch.from_numpy(v)

def context_prefix(vec:torch.Tensor,modes:List[int])->str:
    v=vec.numpy();v=v/(np.linalg.norm(v)+1e-8);b=np.clip(np.round((v+1)*15).astype(int),0,30)
    payload={"frnn_do":b.tolist(),"frnn_modes_top5":modes}
    return "[FRNN-CONTEXT]"+base64.b64encode(json.dumps(payload).encode()).decode()

def _parse(js:Dict)->str:
    ch=(js.get("choices") or [{}])[0];m=ch.get("message",{});return m.get("content") or ch.get("text","")

def deepseek_chat(msgs:List[Dict],context_prefix_str:str,model=None,api_key=None,api_url=None,
                  temperature=0.2,top_p=0.9,timeout_s=None,max_retries=3,extra=None)->str:
    api_key=api_key or os.getenv("DEEPSEEK_API_KEY","");
    if not api_key: raise RuntimeError("DEEPSEEK_API_KEY not set")
    url=api_url or DEFAULT_URL; model=model or DEFAULT_MODEL; timeout_s=timeout_s or DEFAULT_TIMEOUT
    messages=[{"role":"system","content":context_prefix_str}]+msgs
    head={"Authorization":f"Bearer {api_key}","Content-Type":"application/json"}
    body={"model":model,"messages":messages,"temperature":float(temperature),"top_p":float(top_p),"stream":False}
    if extra: body.update(extra)
    err=None
    for i in range(max_retries):
        try:
            r=requests.post(url,headers=head,json=body,timeout=timeout_s)
            if r.status_code in (429,500,502,503,504): raise RuntimeError(f"DeepSeek HTTP {r.status_code}")
            r.raise_for_status();return _parse(r.json())
        except Exception as e: err=e; time.sleep(min(2**i,8))
    raise RuntimeError(f"DeepSeek failed after {max_retries}: {err}")
