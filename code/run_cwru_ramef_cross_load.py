"""Geometry-conditioned RAMEF confirmation on native CWRU cross-load bags."""
from __future__ import annotations
import argparse,csv,json
from collections import defaultdict
from pathlib import Path
import numpy as np,torch
from scipy.io import loadmat
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from bottleneck_deconvolution import sha256
from geometry_static_evidence import bag_descriptor,block_evidence
from racp_ablation import deterministic_nuisance
from run_core_component_ablation import make_u2
from run_cwru_10class_baseline import coarse_metric,metric
from run_cwru_native_a1_multiseed import SEEDS
from run_cwru_native_u2 import build_waves,embed

METHODS=('A1_MEAN8','RAMEF','A1_MEAN8_RANDOM36');ORDERS=(3.5848,5.4152,2.3570);BANDS=((500.,1800.),(1800.,3500.),(3500.,5500.))

def write_csv(path,rows):
 with path.open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def physical_cache(pool,path):
 ids=[r['window_id'] for r in pool]
 if path.exists():
  p=np.load(path,allow_pickle=False)
  if p['ids'].tolist()!=ids:return (_ for _ in ()).throw(RuntimeError('stale physical cache'))
  return p['evidence'].astype(np.float32)
 by_record=defaultdict(list)
 for i,r in enumerate(pool):by_record[r['record_id']].append((i,r))
 evidence=np.zeros((len(pool),3,5),dtype=np.float32)
 for record,items in sorted(by_record.items()):
  rows=[r for _,r in items];key=rows[0]['signal_key'];signal=np.asarray(loadmat(rows[0]['path'],variable_names=[key])[key]).squeeze();blocks=np.stack([np.asarray(signal[int(r['raw_start']):int(r['raw_end'])],dtype=np.float32) for r in rows]);rpms=np.asarray([float(r['rpm']) for r in rows]);value=block_evidence(blocks,12000.,rpms,ORDERS,BANDS)
  for (i,_),v in zip(items,value):evidence[i]=v
  print('physical',record,flush=True)
 np.savez_compressed(path,ids=np.asarray(ids),evidence=evidence);return evidence

def feature_cache(model,target,seed,waves,rpms,device,pool,path):
 ids=[r['window_id'] for r in pool]
 if path.exists():
  p=np.load(path,allow_pickle=False)
  if p['ids'].tolist()!=ids:return (_ for _ in ()).throw(RuntimeError('stale A1 cache'))
  return p['features'].astype(np.float32)
 z=embed(model,waves,rpms,device,batch=24).astype(np.float16);np.savez_compressed(path,ids=np.asarray(ids),features=z);return z.astype(np.float32)

def fit_predict(x,y,tx,ty,qx,seed):
 train=np.vstack([x,tx]);labels=np.concatenate([y,ty]);sc=StandardScaler().fit(train);h=LogisticRegression(C=.1,max_iter=2000,solver='lbfgs',random_state=seed);h.fit(sc.transform(train),labels,sample_weight=np.concatenate([np.ones(len(x)),np.full(len(tx),10.)]));return h.decision_function(sc.transform(qx))

def stats(values):return {'mean':float(np.mean(values)),'std':float(np.std(values,ddof=1)),'min':float(np.min(values)),'max':float(np.max(values))}

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('outputs/staged/CWRU_RAMEF_CROSS_LOAD'));p.add_argument('--protocol',type=Path,default=Path('CWRU_RAMEF_CROSS_LOAD_PROTOCOL.md'));a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 with Path('cwru_10class_window_pool.csv').open(encoding='utf-8',newline='') as f:pool=list(csv.DictReader(f))
 split=json.loads(Path('cwru_ramef_bag_splits.json').read_text());index={r['window_id']:i for i,r in enumerate(pool)};labels=np.asarray([int(r['label']) for r in pool]);rpms=torch.tensor([float(r['rpm']) for r in pool]);waves=build_waves(pool,Path('outputs/staged/CWRU_NATIVE_A1_MULTISEED/waves.pt'));physical=physical_cache(pool,a.output/'window_physical.npz');device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');episode_rows=[];prediction_rows=[]
 for seed in SEEDS:
  for target in range(4):
   checkpoint=Path(f'outputs/staged/CWRU_NATIVE_A1_MULTISEED/native_a1_T{target}_seed{seed}.pt');payload=torch.load(checkpoint,map_location='cpu',weights_only=False);model=make_u2('hz','frozen/M2_LOCKED_v1_20260715/model/resolved_config.yaml',10);model.load_state_dict(payload['state']);model.to(device).eval();z=feature_cache(model,target,seed,waves,rpms,device,pool,a.output/f'a1_T{target}_seed{seed}.npz');episodes=[e for e in split['episodes'] if int(e['target_load'])==target]
   for episode in episodes:
    views={name:{'source':[],'support':[],'query':[]} for name in METHODS};ys={'source':[],'support':[],'query':[]};bag_meta={'query':[]}
    for bag in episode['bags']:
     wi=np.asarray([index[w] for w in bag['window_ids']]);mean=z[wi].mean(0);phys=bag_descriptor(physical[wi],3);rand=deterministic_nuisance([bag['bag_id']],dimensions=36)[0];role=bag['role'];views['A1_MEAN8'][role].append(mean);views['RAMEF'][role].append(np.concatenate([mean,phys]));views['A1_MEAN8_RANDOM36'][role].append(np.concatenate([mean,rand]));ys[role].append(int(bag['label']))
     if role=='query':bag_meta['query'].append(bag)
    for name in METHODS:
     logits=fit_predict(np.asarray(views[name]['source']),np.asarray(ys['source']),np.asarray(views[name]['support']),np.asarray(ys['support']),np.asarray(views[name]['query']),seed+int(episode['seed']));truth=np.asarray(ys['query']);pred=logits.argmax(1);fine=metric(truth,pred);coarse=coarse_metric(truth,pred);episode_rows.append({'backbone_seed':seed,'target_load':target,'episode_id':episode['episode_id'],'method':name,'accuracy':fine['accuracy'],'macro_f1':fine['macro_f1'],'coarse_accuracy':coarse['accuracy'],'errors':fine['errors']})
     for bag,t,g in zip(bag_meta['query'],truth,pred):prediction_rows.append({'backbone_seed':seed,'episode_id':episode['episode_id'],'target_load':target,'method':name,'bag_id':bag['bag_id'],'class_name':bag['class_name'],'true':int(t),'prediction':int(g),'correct':int(t==g)})
   del model;torch.cuda.empty_cache();print('model',seed,target,flush=True)
 summary={name:{key:stats([r[key] for r in episode_rows if r['method']==name]) for key in ('accuracy','macro_f1','coarse_accuracy')} for name in METHODS};per_target={str(t):{name:stats([r['accuracy'] for r in episode_rows if r['method']==name and r['target_load']==t]) for name in METHODS} for t in range(4)};base=summary['A1_MEAN8'];candidate=summary['RAMEF'];random=summary['A1_MEAN8_RANDOM36'];g={'accuracy_noninferior_0_1pp':candidate['accuracy']['mean']>=base['accuracy']['mean']-.001,'macro_f1_noninferior_0_1pp':candidate['macro_f1']['mean']>=base['macro_f1']['mean']-.001,'mean_accuracy_ge_99':candidate['accuracy']['mean']>=.99,'all_target_means_ge_98':min(per_target[str(t)]['RAMEF']['mean'] for t in range(4))>=.98,'random_not_above_RAMEF':random['accuracy']['mean']<=candidate['accuracy']['mean']};result={'experiment':'CWRU_RAMEF_CROSS_LOAD','status':'PASS' if all(g.values()) else 'FAIL','epistemic_status':'geometry-conditioned cross-dataset confirmation; bag-level target split shares parent records','evaluations':len(episode_rows),'summary':summary,'per_target':per_target,'RAMEF_minus_A1_accuracy_pp':100*(candidate['accuracy']['mean']-base['accuracy']['mean']),'gates':g,'protocol_sha256':sha256(a.protocol),'bag_split_sha256':sha256(Path('cwru_ramef_bag_splits.json')),'pool_sha256':sha256(Path('cwru_10class_window_pool.csv'))};write_csv(a.output/'episode_metrics.csv',episode_rows);write_csv(a.output/'predictions.csv',prediction_rows);(a.output/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
