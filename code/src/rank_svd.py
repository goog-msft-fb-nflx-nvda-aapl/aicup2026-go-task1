"""Rank: hashed own-move token TF-IDF -> SVD(NC) per game, saved; HistGB on [SVD + handcrafted] ; hold-player question sim; saves per-game log-probs for hold games."""
import sys,time,math,numpy as np,scipy.sparse as sp
sys.path.insert(0,(__import__('os').environ.get('GO_ROOT','/home/jtan/solardefect/go_20261113')+'/src'))
from id2 import *
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.ensemble import HistGradientBoostingClassifier
NC=int(sys.argv[1]);t0=time.time()
z=np.load(D+'data/train_arr.npz');M,C,col,P,y,hd,nm=z['M'],z['C'],z['col'],z['P'],z['y'],z['hd'],z['nm'];N=len(M)
fl=(1,1,1,1,1,1)
X=sp.vstack([mat(M,C,np.arange(s,min(s+50000,N)),col[s:s+50000],fl) for s in range(0,N,50000)]).tocsr();X.data[:]=1;print('X',X.nnz,time.time()-t0,flush=True)
df=np.asarray(X.sum(0)).ravel();idf=np.log((N+1)/(df+1)).astype(np.float32);Xn=normalize(X@sp.diags(idf))
rng=np.random.default_rng(1234);players=np.unique(P);hold=rng.choice(players,800,replace=False);ish=np.isin(P,hold);tr=np.flatnonzero(~ish);va=np.flatnonzero(ish)
svd=TruncatedSVD(NC,algorithm='randomized',n_iter=4,random_state=0).fit(Xn[tr[rng.permutation(len(tr))[:200000]]]);print('svd',time.time()-t0,flush=True)
Z=svd.transform(Xn).astype(np.float32);np.save(D+f'results/rank_svd{NC}_Z.npy',Z.astype(np.float16))
import joblib;joblib.dump(dict(svd=svd,idf=idf),D+f'results/rank_svd{NC}.joblib')
F=np.hstack([Z,hd[:,None],col[:,None],(nm[:,None]/300.)]).astype(np.float32)
clf=HistGradientBoostingClassifier(max_iter=250,learning_rate=0.08,max_leaf_nodes=63,l2_regularization=1.0,early_stopping=False).fit(F[tr[::2]],y[tr[::2]]);print('gbm',time.time()-t0,flush=True)
pr=clf.predict_proba(F[va]);LP=np.log(pr+1e-9);np.save(D+f'results/rank_svd{NC}_holdLP.npy',LP);np.save(D+f'results/rank_svd{NC}_holdidx.npy',va)
print('game exact',(pr.argmax(1)==y[va]).mean(),flush=True)
loc={g:i for i,g in enumerate(va)};byp={}
for g in va:byp.setdefault((P[g],col[g]),[]).append(g)
keys=[k for k,v in byp.items() if len(v)>=20];r2=np.random.default_rng(3)
def qs(lp,t):
    p=np.exp(lp-lp.max());p/=p.sum();sc=p.copy();sc[1:]+=math.exp(-1)*p[:-1];sc[:-1]+=math.exp(-1)*p[1:];k=sc.argmax();return 1. if k==t else (math.exp(-1) if abs(k-t)==1 else 0.)
tot=[]
for n in (5,10,15,20):
    s_=[]
    for _ in range(800):
        k=keys[r2.integers(len(keys))];ix=r2.choice(byp[k],n,replace=False);s_.append(qs(LP[[loc[i] for i in ix]].sum(0)/3,y[ix[0]]))
    print('RANK n',n,'%.4f'%np.mean(s_));tot+=s_
print('RANK ALL %.4f'%np.mean(tot))
