"""CNN-head port of the frozen stacked, validation-gated constructor on fresh KMNIST tasks (budgets 32/128)."""
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from metricdp_pytorch.utils.split_data import partition_by_class_counts
from research.calculations.private_descriptor_probe import ROOT, probabilities
from research.calculations.client_energy_filter_probe import C, scores
from research.calculations.public_residual_probe import digest, public_geometry, bounded_public
from research.calculations.public_residual_signal_diagnostic import example_gradients
from research.calculations.class_conditional_probe import client_queries
from research.calculations.limited_public_probe import noise_total
from research.calculations.stacked_constructor_probe import FREEZE as STACKED_FREEZE, DRAWS_CONFIRM
from research.calculations.gated_step_probe import load
from research.calculations.public_residual_probe import public_scale

PROTOCOL=Path('research/proposals/2026-10-08_cnn_head_protocol.md')
CODE=Path('research/calculations/cnn_head_probe.py')
OUT=ROOT/'cnn_head.json'
NPZ=ROOT/'cnn_head.npz'
BUDGETS=(32,128)
MULTIPLIERS=(0.,1/30,1/10,1/3,1.,3.,10.,30.)
EPOCHS=(30,100,300)
LRS=(1e-3,3e-3)
RISKS=(.65,.80)
SEED=20261012
FEATURES=32


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.body=nn.Sequential(nn.Conv2d(1,8,3),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(8,16,3),nn.ReLU(),nn.MaxPool2d(2),nn.Flatten(),nn.Linear(400,FEATURES),nn.ReLU())
        self.head=nn.Linear(FEATURES,4)

    def forward(self,x): return self.head(self.body(x))


def images(data,ids):
    return np.stack([np.asarray(data[int(i)]['image'],dtype=np.float32) for i in ids])[:,None]/255-.5


def train(x,y,epochs,lr,seed):
    torch.manual_seed(seed); net=Net(); opt=torch.optim.Adam(net.parameters(),lr=lr,weight_decay=1e-3); xt=torch.tensor(x); yt=torch.tensor(y)
    for _ in range(epochs):
        opt.zero_grad(); nn.functional.cross_entropy(net(xt),yt).backward(); opt.step()
    return net.eval()


def embed(net,x):
    with torch.no_grad(): f=net.body(torch.tensor(x)).numpy().astype(float)
    return np.column_stack((f,np.ones(len(f))))


def head_theta(net):
    w=np.column_stack((net.head.weight.detach().numpy(),net.head.bias.detach().numpy())).astype(float); return C.T@w


def hessian(x,theta):
    p=probabilities(x@theta.T); s=-p[:,:,None]*p[:,None,:]; s[:,np.arange(4),np.arange(4)]+=p
    ident=np.einsum('ca,ncd,db->nab',C,s,C); n=3*x.shape[1]; return np.einsum('nab,ni,nj->aibj',ident,x,x,optimize=True).reshape(n,n)/len(x)


def main():
    fz=json.loads(STACKED_FREEZE.read_text())['frozen']; saved={}; tasks={}
    for name,first in (('kmnist_classes0to3',0),('kmnist_classes4to7',4)):
        data,raw=load('kmnist'); rng=np.random.default_rng(SEED+first); out={f'public{b}_{s}':[] for b in BUDGETS for s in range(3)}; out.update(validation=[],A=[],B=[],evaluation=[])
        for k in range(4):
            ids=np.where(raw==first+k)[0].copy(); rng.shuffle(ids); pos=0
            for b in BUDGETS:
                for s in range(3): out[f'public{b}_{s}'].append(ids[pos:pos+b//4]); pos+=b//4
            for n,m in (('validation',128),('A',512),('B',512),('evaluation',512)): out[n].append(ids[pos:pos+m]); pos+=m
        r={n:np.concatenate(v) for n,v in out.items()}; allids=np.concatenate(list(r.values())); assert len(set(allids.tolist()))==len(allids); lab=lambda ids:raw[ids]-first
        vx=images(data,r['validation']); vy=lab(r['validation']); ex=images(data,r['evaluation']); ey=lab(r['evaluation']); coh={}
        counts=np.full((8,4),17); counts[np.arange(8),np.arange(8)%4]=205
        for c in 'AB': coh[c]=(images(data,r[c]),lab(r[c]),partition_by_class_counts(lab(r[c]),counts,seed=SEED+first))
        cells=[]
        for b in BUDGETS:
            for s in range(3):
                ids=r[f'public{b}_{s}']; px=images(data,ids); py=lab(ids); cands=[]
                for e in EPOCHS:
                    for lr in LRS:
                        net=train(px,py,e,lr,SEED+b+s+e); th=head_theta(net); cands.append((net,th,float(scores(embed(net,vx),vy,th[None])[0][0]),{'epochs':e,'lr':lr}))
                net,theta0,vce0,base_cfg=min(cands,key=lambda c:c[2]); fv=embed(net,vx); fe=embed(net,ex); fp=embed(net,px); cce=float(scores(fe,ey,theta0[None])[0][0]); cacc=float(scores(fe,ey,theta0[None])[1][0])
                gamma=np.stack([example_gradients(fp[py==k],py[py==k],theta0).mean(0) for k in range(4)]); h=hessian(fp,theta0)
                for c in 'AB':
                    cx,cy,parts=coh[c]; fc=embed(net,cx); queries,_=client_queries(fc,cy,parts,theta0,gamma,np.ones(4)/4)
                    row={'task':name,'budget':b,'subset':s,'cohort':c,'base':base_cfg,'control':{'ce':cce,'accuracy':cacc,'validation_ce':vce0},'risks':{}}
                    for risk in RISKS:
                        key=f'noisy_{int(risk*100)}'; cfg=fz[key]; d=cfg['dimension']; g=public_geometry(h,d,'euclidean'); q=queries[cfg['mode']].reshape(8,-1)@g['encoder']
                        signal=bounded_public(q,cfg['cap'])[0].sum(0); seed=SEED+1000*b+100*s+10*'AB'.index(c)+int(risk*100)
                        noise=noise_total(DRAWS_CONFIRM,seed,d,public_scale(cfg['cap'],risk)); total=signal[None]+noise; delta=(total@g['decoder']).reshape(-1,3,theta0.shape[1])
                        vce=[];ece=[];eacc=[]
                        for m in MULTIPLIERS:
                            th=theta0[None]-cfg['eta']*m*delta; vce.append(scores(fv,vy,th)[0]); e,a=scores(fe,ey,th); ece.append(e); eacc.append(a)
                        vce,ece,eacc=map(np.array,(vce,ece,eacc)); pick=vce.argmin(0); idx=np.arange(ece.shape[1]); gce=ece[pick,idx]; gacc=eacc[pick,idx]
                        row['risks'][str(risk)]={'gated_ce':float(gce.mean()),'gated_gain':cce-float(gce.mean()),'gated_accuracy':float(gacc.mean()),'pick_fraction':[float((pick==i).mean()) for i in range(len(MULTIPLIERS))],'by_multiplier_ce':[float(x) for x in ece.mean(axis=1)]}
                        saved[f'{name}_b{b}_s{s}_{c}_{key}_gated_ce']=gce
                    cells.append(row)
                print(name,b,s,'base',base_cfg,'control ce',round(cce,4),flush=True)
        tasks[name]=cells; saved[name+'_roles']=allids
    np.savez_compressed(NPZ,**saved)
    OUT.write_text(json.dumps({'kind':'CNN-head port of frozen stacked constructor with validation-gated wide step grid; fresh KMNIST.','protocol_sha':digest(PROTOCOL),'calculator_sha':digest(CODE),'freeze_sha':digest(STACKED_FREEZE),'multipliers':MULTIPLIERS,'draws':DRAWS_CONFIRM,'tasks':tasks},indent=1,allow_nan=False)+'\n')


if __name__=='__main__': main()
