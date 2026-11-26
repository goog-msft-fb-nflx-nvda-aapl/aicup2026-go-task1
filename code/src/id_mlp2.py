"""Set-level MLP+ArcFace on SVD400 features (mean of m in 1..5 game embeddings per sample). usage: id_mlp2.py steps H margin drop noise seed"""
import sys,time,math,numpy as np,torch,torch.nn as nn,torch.nn.functional as F
D=(__import__('os').environ.get('GO_ROOT','/home/jtan/solardefect/go_20261113')+'/');dev=torch.device('cuda',0)
STEPS=int(sys.argv[1]);H=int(sys.argv[2]);MARG=float(sys.argv[3]);DROP=float(sys.argv[4]);NOISE=float(sys.argv[5]);SEED=int(sys.argv[6]);torch.manual_seed(SEED)
z=np.load(D+'data/train_arr.npz');col,P=z['col'],z['P'];Z=np.load(D+'results/rank_svd400_Z.npy').astype(np.float32)
rng=np.random.default_rng(1234);players=np.unique(P);hold=rng.choice(players,800,replace=False);ish=np.isin(P,hold);tr=np.flatnonzero(~ish);va=np.flatnonzero(ish)
mu=Z[tr].mean(0);sd=Z[tr].std(0)+1e-6;Z=(Z-mu)/sd;pm=-np.ones(P.max()+1,np.int64);trp=np.unique(P[tr]);pm[trp]=np.arange(len(trp));Zg=torch.from_numpy(Z).to(dev)
order=tr[np.argsort(pm[P[tr]],kind='stable')];cnt=np.bincount(pm[P[tr]],minlength=len(trp));starts=np.r_[0,np.cumsum(cnt)[:-1]];ORD=torch.from_numpy(order).to(dev);ST=torch.from_numpy(starts).to(dev);CN=torch.from_numpy(cnt).to(dev)
net=nn.Sequential(nn.Linear(400,H),nn.BatchNorm1d(H),nn.GELU(),nn.Dropout(DROP),nn.Linear(H,H),nn.BatchNorm1d(H),nn.GELU(),nn.Linear(H,256)).to(dev);W=nn.Parameter(torch.randn(len(trp),256,device=dev)*0.02)
opt=torch.optim.AdamW(list(net.parameters())+[W],lr=2e-3,weight_decay=1e-2);sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=2e-3,total_steps=STEPS);BS=1024;K=5
def sim():
    net.eval()
    with torch.no_grad():E=torch.cat([F.normalize(net(Zg[torch.from_numpy(va[s:s+8192]).to(dev)]),dim=-1) for s in range(0,len(va),8192)]).cpu().numpy()
    net.train();cn=np.bincount(P[va],minlength=P.max()+1);r2=np.random.default_rng(5);elig=[p for p in hold if cn[p]>=15];pl=r2.choice(elig,min(600,len(elig)),replace=False);loc={g:i for i,g in enumerate(va)};cand=[];qry=[]
    for p in pl:
        g=r2.permutation(va[P[va]==p]);cand.append(g[:10]);qry.append(g[10:15])
    Pr=np.stack([E[[loc[i] for i in c]].mean(0) for c in cand]);Pr/=np.linalg.norm(Pr,axis=1,keepdims=True);res={}
    for n in (1,2,3,5):
        s=[]
        for _ in range(700):
            j=r2.integers(len(pl));q=E[[loc[i] for i in qry[j][:n]]].mean(0);o=np.argsort(-(Pr@q))[:5];r=np.flatnonzero(o==j);s.append(math.exp(-r[0]) if len(r) else 0)
        res[n]=round(float(np.mean(s)),3)
    return res,float(np.mean(list(res.values())))
tl=0
for s in range(STEPS):
    pi=torch.randint(0,len(trp),(BS,),device=dev);m=torch.randint(1,6,(BS,),device=dev);r=torch.floor(torch.rand(BS,K,device=dev)*CN[pi][:,None]).long();g=ORD[ST[pi][:,None]+r].reshape(-1)
    x=Zg[g]+NOISE*torch.randn(BS*K,400,device=dev);E=F.normalize(net(x),dim=-1).reshape(BS,K,-1);mask=(torch.arange(K,device=dev)[None]<m[:,None]).float()[...,None];e=F.normalize((E*mask).sum(1),dim=-1)
    cos=e@F.normalize(W,dim=-1).T;th=torch.acos(cos.clamp(-1+1e-6,1-1e-6));lg=cos.scatter(1,pi[:,None],torch.cos(th.gather(1,pi[:,None])+MARG))*30;loss=F.cross_entropy(lg,pi)
    opt.zero_grad();loss.backward();opt.step();sched.step();tl+=loss.item()
    if (s+1)%1000==0:r_,a=sim();print('step',s+1,'loss %.3f'%(tl/1000),'NEW sim',r_,'ALL %.4f'%a,flush=True);tl=0
torch.save({'net':net.state_dict(),'mu':mu,'sd':sd},D+f'results/id_mlp2_{STEPS}_{H}_{MARG}_{DROP}_{NOISE}_{SEED}.pt')
