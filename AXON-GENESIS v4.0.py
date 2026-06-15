"""
AXON-GENESIS v4.0 — FIXED
==========================
FIX 1: Clifford algebra _blade_prod bug corrected
FIX 2: NF4 replaced with Evidential uncertainty (Sensoy et al. 2018)
FIX 3: Trinity R2 uses batch diversity instead of softmax entropy
FIX 4: Honest framing — this is goal-conditioned regression, not magic
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import logging
from typing import Tuple, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger("axon")


# ─── FIX 1: CORRECT Clifford Algebra Cl(n,0) ─────────────────────────────────

class CliffordAlgebra:
    """
    Clifford Algebra Cl(n,0): e_i^2=+1, e_i*e_j=-e_j*e_i (i!=j)
    Multivector dimension: 2^n

    THE BUG THAT WAS FIXED:
    Old _blade_prod computed sign by sorting arr in-place, then REBUILT
    arr from scratch — making the computed sign meaningless.
    
    New algorithm: single bubble-sort pass on the concatenated list.
    Every adjacent swap of DIFFERENT indices multiplies sign by -1.
    Pairs of identical adjacent indices cancel (e_i^2=+1), no sign change.
    """
    def __init__(self, n: int):
        self.n = n
        self.dim = 2**n
        blades = [frozenset()]
        for grade in range(1, n+1):
            blades += [frozenset(c) for c in self._combos(list(range(n)), grade)]
        self.blades = blades
        self.b2i = {b:i for i,b in enumerate(blades)}
        self._build_table()

    def _combos(self, lst, r):
        if r==0: yield []; return
        for i,v in enumerate(lst):
            for c in self._combos(lst[i+1:], r-1): yield [v]+c

    def _blade_prod(self, b1, b2):
        """FIXED: sign and result are consistent."""
        arr = sorted(b1) + sorted(b2)
        sign = 1
        n = len(arr)
        # Single bubble-sort pass — tracks EVERY transposition
        for i in range(n):
            for j in range(n-1-i):
                if arr[j] > arr[j+1]:
                    arr[j], arr[j+1] = arr[j+1], arr[j]
                    sign *= -1
        # Remove pairs: e_i^2 = +1 (no sign change on removal)
        result = []
        i = 0
        while i < len(arr):
            if i+1 < len(arr) and arr[i] == arr[i+1]:
                i += 2
            else:
                result.append(arr[i]); i += 1
        return sign, frozenset(result)

    def _build_table(self):
        d = self.dim
        self.gp_sign = torch.zeros(d,d)
        self.gp_idx  = torch.zeros(d,d,dtype=torch.long)
        for i,b1 in enumerate(self.blades):
            for j,b2 in enumerate(self.blades):
                s,r = self._blade_prod(b1,b2)
                self.gp_sign[i,j]=float(s); self.gp_idx[i,j]=self.b2i[r]

    def gp(self, a, b):
        dev=a.device; d=self.dim; B=a.shape[0]
        sign=self.gp_sign.to(dev); idx=self.gp_idx.to(dev)
        af=a.reshape(B,d); bf=b.reshape(B,d)
        outer=af.unsqueeze(2)*bf.unsqueeze(1)
        signed=outer*sign.unsqueeze(0)
        c=torch.zeros(B,d,device=dev)
        c.scatter_add_(1,idx.unsqueeze(0).expand(B,-1,-1).reshape(B,-1),signed.reshape(B,-1))
        return c.reshape(*a.shape[:-1],d)

    def grade(self, mv, k):
        mask=torch.zeros(self.dim,device=mv.device)
        for i,b in enumerate(self.blades):
            if len(b)==k: mask[i]=1.
        return mv*mask

    def verify(self):
        """Test against known Cl(3,0) identities."""
        ok=True
        e0=torch.zeros(1,self.dim); e0[0,1]=1.0
        e1=torch.zeros(1,self.dim); e1[0,2]=1.0
        # e0^2 = +1
        r=self.gp(e0,e0); ok&=(abs(r[0,0].item()-1.)<1e-6 and r[0,1:].abs().max()<1e-6)
        # e0*e1 = +e01
        e01_i=self.b2i[frozenset({0,1})]; r=self.gp(e0,e1)
        ok&=(abs(r[0,e01_i].item()-1.)<1e-6)
        # e1*e0 = -e01
        r=self.gp(e1,e0); ok&=(abs(r[0,e01_i].item()+1.)<1e-6)
        # I^2 = -1 (pseudoscalar for n=3)
        if self.n==3:
            pi=self.b2i[frozenset({0,1,2})]
            ps=torch.zeros(1,self.dim); ps[0,pi]=1.0
            r=self.gp(ps,ps); ok&=(abs(r[0,0].item()+1.)<1e-6)
        return ok

_CA={}
def get_clifford(n):
    if n not in _CA: _CA[n]=CliffordAlgebra(n)
    return _CA[n]


class GALayer(nn.Module):
    """GA layer using FIXED Clifford algebra."""
    def __init__(self,d_in,d_out,n=3):
        super().__init__()
        self.ca=get_clifford(n); self.n=n; self.mv=2**n
        self.to_mv=nn.Linear(d_in,self.mv)
        self.W=nn.Parameter(torch.zeros(self.mv)); self.W.data[0]=1.
        self.gw=nn.Parameter(torch.ones(n+1)/(n+1))
        self.proj=nn.Linear(self.mv,d_out)
        self.norm=nn.LayerNorm(d_out)
        self.res=nn.Linear(d_in,d_out,bias=False) if d_in!=d_out else nn.Identity()
    def forward(self,x):
        B=x.shape[0]; mv=self.to_mv(x)
        w=self.W.unsqueeze(0).expand(B,-1)
        rot=self.ca.gp(w,mv)
        gw=torch.softmax(self.gw,0)
        agg=sum(gw[k]*self.ca.grade(rot,k) for k in range(self.n+1))
        return self.norm(self.proj(agg)+self.res(x))
    def grade_entropy(self):
        g=torch.softmax(self.gw,0)
        return -(g*(g+1e-8).log()).sum()


# ─── FIX 2: Evidential Uncertainty (replaces arbitrary NF4) ──────────────────

class EvidentialHead(nn.Module):
    """
    Evidential uncertainty — Sensoy et al., NeurIPS 2018.
    Outputs Dirichlet parameters α from which uncertainty u = K/Σα.
    Theoretically grounded: uncertainty decreases as evidence accumulates.
    """
    def __init__(self,d_in,n_out):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(d_in,d_in//2),nn.ReLU(),nn.Linear(d_in//2,n_out))
        self.n=n_out
    def forward(self,x):
        ev=F.softplus(self.net(x))   # evidence >=0
        alpha=ev+1.0                  # Dirichlet params >=1
        S=alpha.sum(-1,keepdim=True)
        u=self.n/S.squeeze(-1)       # uncertainty in [0,1]
        return alpha, u
    @staticmethod
    def edl_loss(alpha,y,step,max_steps):
        S=alpha.sum(-1,keepdim=True); p_hat=alpha/S
        nll=(y-p_hat).pow(2).sum(-1).mean()
        lam=min(1.,step/max(max_steps,1))
        at=y+(1-y)*alpha
        kl=_dirichlet_kl(at).mean()
        return nll+lam*kl

def _dirichlet_kl(alpha):
    K=alpha.shape[-1]; S=alpha.sum(-1,keepdim=True)
    kl=(torch.lgamma(S)-torch.lgamma(torch.tensor(float(K)))
        -torch.lgamma(alpha).sum(-1,keepdim=True)
        +((alpha-1)*(torch.digamma(alpha)-torch.digamma(S))).sum(-1,keepdim=True))
    return kl.squeeze(-1)


# ─── MINE Confidence (preserved, correct) ────────────────────────────────────

class MINEConf(nn.Module):
    def __init__(self,a_dim,g_dim,h=64):
        super().__init__()
        self.T=nn.Sequential(nn.Linear(a_dim+g_dim,h),nn.ELU(),nn.Linear(h,1))
        self.register_buffer('ema',torch.tensor(1.))
    def forward(self,a,g):
        B=a.shape[0]; idx=torch.randperm(B,device=a.device)
        tj=self.T(torch.cat([a,g],-1)); tm=self.T(torch.cat([a[idx],g],-1))
        et=torch.exp(tm-tm.max().detach())
        self.ema=((1-.01)*self.ema+.01*et.mean().detach()).detach()
        mi=tj.mean()-(et.mean()/(self.ema+1e-8)).log()
        return mi,torch.sigmoid(mi)
    def loss(self,a,g): return -self.forward(a,g)[0]


# ─── IB Layer (preserved, correct) ───────────────────────────────────────────

class IBLayer(nn.Module):
    def __init__(self,d,beta_init=0.001):
        super().__init__()
        self.mu=nn.Linear(d,d); self.lv=nn.Linear(d,d)
        self.beta_init=beta_init
        self.register_buffer('beta',torch.tensor(beta_init))
    def forward(self,x):
        mu=self.mu(x); lv=self.lv(x).clamp(-4,4)
        z=mu+(0.5*lv).exp()*torch.randn_like(mu) if self.training else mu
        kl=-0.5*(1+lv-mu.pow(2)-lv.exp()).sum(-1).mean()
        return z,self.beta*kl
    def anneal(self,step,max_steps):
        p=min(step/max(max_steps,1),1.)
        with torch.no_grad(): self.beta.copy_(torch.tensor(self.beta_init*(1+9*p)))


# ─── FIX 3: Trinity Reward — correct R2 ──────────────────────────────────────

class TrinityReward(nn.Module):
    """
    R1=−MSE  R2=batch_diversity  R3=I(a;g)  R4=conf×correct+0.2
    FIX: R2 now uses mean pairwise cosine distance (genuine diversity measure)
    """
    def __init__(self,a_dim,g_dim):
        super().__init__()
        self.log_alpha=nn.Parameter(torch.zeros(4))
        self.mine_T=nn.Sequential(nn.Linear(a_dim+g_dim,64),nn.ELU(),nn.Linear(64,1))
        self.register_buffer('ema',torch.tensor(1.))
    def compute(self,a_pred,a_true,goal,uncertainty):
        B=a_pred.shape[0]; alpha=torch.softmax(self.log_alpha,0)*4
        err=(a_pred-a_true).pow(2).sum(-1)
        R1=-err
        # R2: batch diversity (FIXED — not softmax entropy of continuous vector)
        a_n=F.normalize(a_pred,-1)
        sim=a_n@a_n.T
        mask=~torch.eye(B,dtype=torch.bool,device=a_pred.device)
        div=(1-sim[mask]).mean()
        R2=div.expand(B)
        # R3: MINE
        idx=torch.randperm(B,device=a_pred.device)
        tj=self.mine_T(torch.cat([a_pred,goal],-1))
        tm=self.mine_T(torch.cat([a_pred[idx],goal],-1))
        et=torch.exp(tm-tm.max().detach())
        self.ema=((1-.01)*self.ema+.01*et.mean().detach()).detach()
        mi=tj.mean()-(et.mean()/(self.ema+1e-8)).log()
        R3=torch.sigmoid(mi).expand(B)
        # R4: calibrated risk (0.2 base, down from 0.3)
        conf=1.-uncertainty.detach()
        corr=1.-(err/(err.max()+1e-8)).detach()
        R4=conf*corr+(1-conf)*0.2
        total=(alpha[0]*R1+alpha[1]*R2+alpha[2]*R3+alpha[3]*R4).mean()
        return {'total':total,'correct':R1.mean().item(),'explore':div.item(),
                'understand':float(torch.sigmoid(mi).item()),'risk':R4.mean().item(),
                'weights':alpha.detach().tolist()}


# ─── Config + Model ───────────────────────────────────────────────────────────

@dataclass
class AXONConfig:
    dim:int=64; action_dim:int=16; n_ga:int=3
    beta_init:float=0.001; lr:float=3e-4
    batch_size:int=32; max_steps:int=1000
    def to_dict(self): return self.__dict__
    @classmethod
    def from_dict(cls,d): return cls(**d)


class AXONModel(nn.Module):
    """
    Goal-conditioned regression with geometric algebra and information bottleneck.
    
    HONEST: This is a neural network that learns f:(state,goal)->action
    by supervised regression on example (state,action,goal) triples.
    The GA layers, IB, and MINE improve generalization and calibration.
    It is NOT a general-purpose inverse solver — it learns a specific inverse.
    """
    def __init__(self, cfg:AXONConfig):
        super().__init__()
        self.cfg=cfg; d=cfg.dim; ad=cfg.action_dim
        # Verify algebra is correct before building
        ca=get_clifford(cfg.n_ga)
        assert ca.verify(), "Clifford algebra incorrect — check implementation"
        self.embed=nn.Sequential(nn.Linear(d,d),nn.LayerNorm(d),nn.GELU())
        self.goal_embed=nn.Sequential(nn.Linear(d,d),nn.LayerNorm(d),nn.GELU())
        self.ga1=GALayer(d,d,n=cfg.n_ga)
        self.ga2=GALayer(d,d,n=cfg.n_ga)
        self.ib=IBLayer(d,beta_init=cfg.beta_init)
        self.fuse=nn.Sequential(nn.Linear(d*2,d*2),nn.GELU(),nn.LayerNorm(d*2),
                                 nn.Linear(d*2,d),nn.GELU())
        self.inv_head=nn.Sequential(nn.Linear(d,d*2),nn.GELU(),nn.LayerNorm(d*2),
                                     nn.Linear(d*2,d),nn.GELU(),nn.LayerNorm(d),
                                     nn.Linear(d,ad))
        self.inv_res=nn.Linear(d,ad,bias=False); nn.init.zeros_(self.inv_res.weight)
        self.evid=EvidentialHead(d,ad)
        self.mine=MINEConf(ad,d)
        self.trinity=TrinityReward(ad,d)
        self.log_sigma=nn.ParameterDict({n:nn.Parameter(torch.tensor(0.))
                                          for n in ['inv','ib','mine','trinity','evid']})
        self.step_count=0; self._opt=None

    def forward(self,state,goal):
        x=self.embed(state); g=self.goal_embed(goal)
        x=self.ga1(x); x=self.ga2(x)
        x,ib_loss=self.ib(x)
        h=self.fuse(torch.cat([x,g],-1))
        action=self.inv_head(h)+self.inv_res(h)
        alpha,uncertainty=self.evid(h)
        mi,_=self.mine(torch.tanh(action),g)
        return {'action':action,'uncertainty':uncertainty,'confidence':1.-uncertainty,
                'mi':mi,'alpha':alpha,'ib_loss':ib_loss}

    def predict(self,state,goal):
        self.eval()
        with torch.no_grad():
            d=self.cfg.dim
            s=torch.FloatTensor((list(state)+[0.]*d)[:d]).unsqueeze(0)
            g=torch.FloatTensor((list(goal)+[0.]*d)[:d]).unsqueeze(0)
            out=self.forward(s,g)
            conf=float((1-out['uncertainty']).item())
            return {'action':out['action'].squeeze(0).numpy().tolist()[:self.cfg.action_dim],
                    'confidence':conf,'uncertainty':float(out['uncertainty'].item()),
                    'mi':float(out['mi'].item()),
                    'decision':'ACT' if conf>0.5 else 'EXPLORE'}

    def _hl(self,n,L):
        ls=self.log_sigma[n]; return L/(2*(2*ls).exp()+1e-8)+ls

    def train_step(self,state,action_true,goal):
        if self._opt is None:
            self._opt=torch.optim.AdamW(self.parameters(),lr=self.cfg.lr,weight_decay=1e-5)
        self._opt.zero_grad()
        self.ib.anneal(self.step_count,self.cfg.max_steps)
        out=self.forward(state,goal)
        action=out['action']; unc=out['uncertainty']
        losses={}
        losses['inv']=F.mse_loss(action,action_true)
        losses['ib']=out['ib_loss']
        losses['mine']=self.mine.loss(torch.tanh(action.detach()),self.goal_embed(goal))
        t=self.trinity.compute(action,action_true,goal,unc)
        losses['trinity']=-t['total']*0.1
        a_dist=torch.softmax(action_true,-1)
        losses['evid']=EvidentialHead.edl_loss(out['alpha'],a_dist,self.step_count,self.cfg.max_steps)
        total=sum(self._hl(n,L) for n,L in losses.items())
        total.backward()
        nn.utils.clip_grad_norm_(self.parameters(),1.0)
        self._opt.step(); self.step_count+=1
        return {'loss':total.item(),'inv_mse':losses['inv'].item(),
                'conf':float((1-unc).mean().item()),
                'R_correct':t['correct'],'R_explore':t['explore'],'R_understand':t['understand']}

    def save(self,path):
        torch.save({'state_dict':self.state_dict(),'config':self.cfg.to_dict(),'step':self.step_count},path)
    @classmethod
    def load(cls,path):
        ck=torch.load(path,map_location='cpu')
        m=cls(AXONConfig.from_dict(ck['config'])); m.load_state_dict(ck['state_dict'])
        m.step_count=ck.get('step',0); return m
