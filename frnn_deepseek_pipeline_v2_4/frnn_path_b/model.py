
import torch
import torch.nn as nn, torch.nn.functional as F
from typing import Tuple

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.g = nn.Parameter(torch.ones(d)); self.eps=eps
    def forward(self,x):
        x=torch.nan_to_num(x)
        return x*(self.g/torch.sqrt((x*x).mean(-1,keepdim=True)+self.eps))

class FRNNPathB(nn.Module):
    __version__="0.2.4"
    def __init__(self, input_dim, output_dim, num_states, memory_dim, hidden_dim, bank_size,
                 use_gumbel=False, tau=1.0, stickiness=0.1, ema_decay=0.99, bank_scale=4.0, use_bank=True):
        super().__init__()
        self.input_dim=input_dim;self.output_dim=output_dim
        self.K=num_states;self.Dm=memory_dim;self.H=hidden_dim;self.bank_size=bank_size
        self.use_gumbel=use_gumbel;self.tau=tau;self.stickiness=float(max(0.0,min(stickiness,1.0)))
        self.ema_decay=float(max(0.0,min(ema_decay,0.9999)));self.bank_scale=bank_scale;self.use_bank=use_bank
        self.Wtr=nn.Linear(input_dim,hidden_dim);self.Wms=nn.Linear(hidden_dim,num_states)
        self.Vproj=nn.Linear(hidden_dim,memory_dim,bias=False)
        self.M=nn.Parameter(torch.randn(num_states,memory_dim)*0.02)
        self.read_norm=RMSNorm(memory_dim);self.Wrd=nn.Linear(memory_dim,output_dim)
        self.register_buffer("bank_keys",torch.zeros(bank_size,memory_dim))
        self.register_buffer("bank_vals",torch.zeros(bank_size,memory_dim))
        self.register_buffer("bank_used",torch.zeros(bank_size))
        self.register_buffer("bank_ptr",torch.zeros(1,dtype=torch.long))
    def _select_mode(self,logits,prev):
        if self.stickiness>0 and prev is not None: logits=logits+self.stickiness*prev
        if self.use_gumbel and self.training: m=F.gumbel_softmax(logits,tau=self.tau,hard=True)
        else: m=F.one_hot(logits.argmax(-1),self.K).float()
        return torch.nan_to_num(m)
    def _bank_push(self,k,v):
        k=torch.nan_to_num(k.mean(0).detach());v=torch.nan_to_num(self.Vproj(v).mean(0).detach())
        p=int(self.bank_ptr.item())
        if self.bank_used[p]>0.5:
            self.bank_keys[p]=self.ema_decay*self.bank_keys[p]+(1-self.ema_decay)*k
            self.bank_vals[p]=self.ema_decay*self.bank_vals[p]+(1-self.ema_decay)*v
        else:
            self.bank_keys[p]=k;self.bank_vals[p]=v;self.bank_used[p]=1.0
        self.bank_ptr[0]=(p+1)%self.bank_size
    def _bank_read(self,q):
        if not self.use_bank:return torch.zeros_like(q)
        mask=self.bank_used[:self.bank_size]>0.5
        if not mask.any():return torch.zeros_like(q)
        K=self.bank_keys[:self.bank_size][mask];V=self.bank_vals[:self.bank_size][mask]
        qn=F.normalize(q,dim=-1);kn=F.normalize(K,dim=-1)
        attn=torch.nan_to_num(qn@kn.t()*self.bank_scale)
        attn=F.softmax(attn,dim=-1);return torch.nan_to_num(attn@V)
    def forward(self,x,prev_mode=None)->Tuple[torch.Tensor,torch.Tensor]:
        B,S,_=x.shape;device=x.device
        if prev_mode is None:
            prev_mode=torch.zeros(B,self.K,device=device);prev_mode[:,0]=1.0
        else: prev_mode=prev_mode.detach()
        outs=[];modes=[]
        for t in range(S):
            h=F.relu(self.Wtr(x[:,t,:]));log=self.Wms(h)
            m=self._select_mode(log,prev_mode);mem=m@self.M
            if self.training:self._bank_push(mem,h)
            bank=self._bank_read(mem)
            y=self.Wrd(self.read_norm(mem+bank))
            outs.append(y);modes.append(m);prev_mode=m
        return torch.stack(outs,1),torch.stack(modes,1)
