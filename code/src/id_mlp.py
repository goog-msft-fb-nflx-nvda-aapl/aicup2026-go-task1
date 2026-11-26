"""Nonlinear few-shot player embedding: MLP+ArcFace on saved SVD features Z (rank_svd%d_Z.npy'%ZNC+', tokens flags (1,1,1,1,1,1)); new-set sim on held-out players (same protocol as id_lda.py)."""
import sys,time,math,numpy as np,torch,torch.nn as nn,torch.nn.functional as F
D=(__import__('os').environ.get('GO_ROOT','/home/jtan/solardefect/go_20261113')+'/');dev=torch.device('cuda',0);t=torch.arange(32,device=dev,dtype=torch.float32);assert (t*t).sum().item()==10416;print('REAL_TENSOR_PASS',flush=True)
import os;ZNC=int(os.environ.get('ZNC','400'));EP=int(sys.argv[1]) if len(sys.argv)>1 else 20;H=int(sys.argv[2]) if len(sys.argv)>2 else 1024;MARG=float(sys.argv[3]) if len(sys.argv)>3 else 0.2;DROP=float(sys.argv[4]) if len(sys.argv)>4 else 0.2;NOISE=float(sys.argv[5]) if len(sys.argv)>5 else 0.05;SEED=int(sys.argv[6]) if len(sys.argv)>6 else 0;torch.manual_seed(SEED)
z=np.load(D+'data/train_arr.npz');col,P=z['col'],z['P'];Z=np.load(D+'results/rank_svd%d_Z.npy'%ZNC+'').astype(np.float32)
rng=np.random.default_rng(1234);players=np.unique(P);hold=rng.choice(players,800,replace=False);ish=np.isin(P,hold);tr=np.flatnonzero(~ish);va=np.flatnonzero(ish)
mu=Z[tr].mean(0);sd=Z[tr].std(0)+1e-6;Z=(Z-mu)/sd;pm=-np.ones(P.max()+1,np.int64);trp=np.unique(P[tr]);pm[trp]=np.arange(len(trp))
Zg=torch.from_numpy(Z).to(dev);Pg=torch.from_numpy(pm[P]).to(dev)
net=nn.Sequential(nn.Linear(ZNC,H),nn.BatchNorm1d(H),nn.GELU(),nn.Dropout(DROP),nn.Linear(H,H),nn.BatchNorm1d(H),nn.GELU(),nn.Linear(H,256)).to(dev);W=nn.Parameter(torch.randn(len(trp),256,device=dev)*0.02)
opt=torch.optim.AdamW(list(net.parameters())+[W],lr=2e-3,weight_decay=1e-2);bs=4096;spe=len(tr)//bs;sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=2e-3,total_steps=EP*spe);trg=torch.from_numpy(tr).to(dev)
def emb(idx):
    net.eval()
    with torch.no_grad():E=torch.cat([F.normalize(net(Zg[idx[s:s+8192]]),dim=-1) for s in range(0,len(idx),8192)])
    net.train();return E.cpu().numpy()
def sim():
    ix=va;E=emb(torch.from_numpy(va).to(dev));cnt=np.bincount(P[va],minlength=P.max()+1);r2=np.random.default_rng(5);elig=[p for p in hold if cnt[p]>=15];pl=r2.choice(elig,min(600,len(elig)),replace=False);loc={g:i for i,g in enumerate(va)};cand=[];qry=[]
    for p in pl:
        g=r2.permutation(va[P[va]==p]);cand.append(g[:10]);qry.append(g[10:15])
    Pr=np.stack([E[[loc[i] for i in c]].mean(0) for c in cand]);Pr/=np.linalg.norm(Pr,axis=1,keepdims=True);res={}
    for n in (1,2,3,5):
        s=[]
        for _ in range(500):
            j=r2.integers(len(pl));q=E[[loc[i] for i in qry[j][:n]]].mean(0);o=np.argsort(-(Pr@q))[:5];r=np.flatnonzero(o==j);s.append(math.exp(-r[0]) if len(r) else 0)
        res[n]=round(float(np.mean(s)),3)
    return res,float(np.mean(list(res.values())))
for ep in range(EP):
    perm=trg[torch.randperm(len(trg),device=dev)];tl=0
    for s in range(spe):
        ib=perm[s*bs:(s+1)*bs];x=Zg[ib]+NOISE*torch.randn_like(Zg[ib]);e=F.normalize(net(x),dim=-1);cos=e@F.normalize(W,dim=-1).T;lab=Pg[ib]
        th=torch.acos(cos.clamp(-1+1e-6,1-1e-6));lg=cos.scatter(1,lab[:,None],torch.cos(th.gather(1,lab[:,None])+MARG))*30;loss=F.cross_entropy(lg,lab)
        opt.zero_grad();loss.backward();opt.step();sched.step();tl+=loss.item()
    if ep%2==1 or ep==EP-1:r,a=sim();print('ep',ep,'loss %.3f'%(tl/spe),'NEW sim',r,'ALL %.4f'%a,flush=True)
torch.save({'net':net.state_dict(),'mu':mu,'sd':sd},D+f'results/id_mlp{ZNC}_{EP}_{H}_{MARG}_{DROP}_{NOISE}_{SEED}.pt')
