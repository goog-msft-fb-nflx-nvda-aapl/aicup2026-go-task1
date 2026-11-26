import sys,time,numpy as np,scipy.sparse as sp,joblib
sys.path.insert(0,(__import__('os').environ.get('GO_ROOT','/home/jtan/solardefect/go_20261113')+'/src'))
from id2 import V
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
D=(__import__('os').environ.get('GO_ROOT','/home/jtan/solardefect/go_20261113')+'/');NC=int(sys.argv[1]);t0=time.time()
z=np.load(D+'data/train_arr.npz');P=z['P'];N=len(P);c=np.load(D+'results/F6_csr.npz');X=sp.csr_matrix((np.ones(len(c['indices']),np.float32),c['indices'],c['indptr']),shape=(N,V))
df=np.asarray(X.sum(0)).ravel();idf=np.log((N+1)/(df+1)).astype(np.float32);Xn=normalize(X@sp.diags(idf));print('X',time.time()-t0,flush=True)
rng=np.random.default_rng(1234);players=np.unique(P);hold=rng.choice(players,800,replace=False);tr=np.flatnonzero(~np.isin(P,hold))
svd=TruncatedSVD(NC,algorithm='randomized',n_iter=3,random_state=0).fit(Xn[tr[rng.permutation(len(tr))[:200000]]]);print('svd',time.time()-t0,flush=True)
Z=np.vstack([svd.transform(Xn[s:s+100000]).astype(np.float16) for s in range(0,N,100000)]);np.save(D+f'results/rank_svd{NC}_Z.npy',Z);joblib.dump(dict(svd=svd,idf=idf),D+f'results/rank_svd{NC}.joblib');print('done',time.time()-t0,flush=True)
