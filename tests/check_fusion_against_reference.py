"""Differential synthetic test against the historical accepted fusion scripts."""
import csv,json,os,subprocess,sys,tempfile
from pathlib import Path
import numpy as np
SOURCE=Path(__file__).resolve().parents[1]/'code/src'
sys.path.insert(0,str(SOURCE))
import inference_final_fusion as fusion

rng=np.random.default_rng(274119);fixture=Path(tempfile.mkdtemp(prefix='go_gate_differential_'));(fixture/'results').mkdir(parents=True,exist_ok=True);(fixture/'data/tests1v2/tests').mkdir(parents=True,exist_ok=True)
rows=[['question_id','top_1','top_2','top_3','top_4','top_5']];expect={}
for pool,suffix,updated in fusion.POOLS:
 n,nc=8,10;data=fixture/'results';base=rng.normal(size=(n,nc)).astype(np.float32);np.save(data/f'S_{pool}.npy',base)
 ids=np.array([pool+'_p_'+str(i) for i in range(nc)]);np.save(data/f'S_{pool}_pids.npy',ids)
 scores={t:rng.normal(size=(n,nc)).astype(np.float32) for t in ['aa','bb']}
 for tag,v in scores.items():np.save(data/f'Sidft_{tag}_{pool}.npy',v)
 graphs=[]
 for tag in ['aa','bb']:
  q=rng.normal(size=(n,5)).astype(np.float32);q[1]=q[0]+.01;graphs.append(q);np.save(data/f'Qemb_{tag}_{pool}.npy',q)
 cp=rng.normal(size=(nc,10)).astype(np.float32);qp=rng.normal(size=(n,10)).astype(np.float32)
 for prefix in ['', 'codex1118_']:
  np.save(data/f'LPc_{prefix}{pool}.npy',cp);np.save(data/f'LPq_{prefix}{pool}.npy',qp)
 num=np.array(([1,2,3,5]*2) if updated else ([5,10,15,20]*2),dtype=float)
 infix='2' if updated else ''
 with (fixture/f'data/tests1v2/tests/player_identification_test{infix}_{suffix}.csv').open('w',newline='') as h:
  w=csv.writer(h);w.writerow(['question_id','num_games'])
  for i,games in enumerate(num):
   qid=pool+'_q_'+str(i);w.writerow([qid,int(games)]);rows.append([qid,*ids[:5]])
 expect[pool]=fusion.coordinate_fusion(base,scores,['aa','bb','aa'],cp,qp,num,np.concatenate(graphs,1),updated)
with (fixture/'template.csv').open('w',newline='') as h:csv.writer(h).writerows(rows)
env=dict(os.environ,GO_ROOT=str(fixture),SFINAL='reference')
subprocess.run([sys.executable,str(SOURCE/'ens_sidft.py'),'ref','aa','bb','aa'],env=env,check=True)
args=[str(fixture/'template.csv'),str(fixture/'output.csv'),'2','3','1.4','1.','ref','3','1','.6','.5','.5','aa,bb','0']
subprocess.run([sys.executable,str(SOURCE/'mk_gate5.py'),*args],env=env,check=True)
for pool in ['pub','priv']:
 actual=np.load(fixture/f'results/Sfinal_reference_{pool}.npy');np.testing.assert_array_equal(actual,expect[pool])
subprocess.run([sys.executable,str(SOURCE/'codex_mk_gate_1118.py'),*args],env=dict(env,NEW_FAMILY_ONLY='1'),check=True)
for pool in ['pub2','priv2']:
 actual=np.load(fixture/f'results/Sfinal_reference_{pool}.npy');np.testing.assert_array_equal(actual,expect[pool])

print('SYNTHETIC_GATE_DIFFERENTIAL_PASSED',flush=True)

