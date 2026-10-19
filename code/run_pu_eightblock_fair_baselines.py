"""Give conventional PU baselines the same eight-block observation budget."""
from __future__ import annotations
import argparse,csv,json,time
from pathlib import Path
import numpy as np,torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,f1_score,recall_score
from sklearn.preprocessing import StandardScaler
from bottleneck_deconvolution import load_vibration,sha256
from fair_baseline_models import make_model
from prepare_splits import scan_records
from racp_or import uniform_nonoverlap_starts,WINDOW
from run_mc_sbt_p0 import SOURCE_DOMAINS
from run_racp_measurement_final_once import load_query
from run_tser_hb_support_p0 import support_only

MODELS=('wdcnn','resnet1d','stft_cnn','wdcnn_dann')

def standardise(x):x=np.asarray(x,dtype=np.float32);return (x-x.mean())/max(float(x.std()),1e-6)

def wave_cache(rows,path):
 ids=[r['measurement_id'] for r in rows]
 if path.exists():
  p=torch.load(path,map_location='cpu',weights_only=False)
  if p['ids']!=ids:raise RuntimeError('stale multi-block wave cache')
  return p['waves'].float()
 waves=np.empty((len(rows),8,WINDOW),dtype=np.float16)
 for i,row in enumerate(rows):
  signal=np.asarray(load_vibration(row['path'],row.get('signal_key','signal')),dtype=np.float32).reshape(-1);starts=uniform_nonoverlap_starts(len(signal));waves[i]=np.stack([standardise(signal[s:s+WINDOW]) for s in starts]).astype(np.float16)
  if (i+1)%100==0:print('waves',i+1,len(rows),flush=True)
 out=torch.from_numpy(waves);torch.save({'ids':ids,'waves':out},path);return out.float()

@torch.no_grad()
def infer(model,waves,device,batch=64):
 flat=waves.reshape(-1,waves.shape[-1]);rows=[];model.eval();start_time=time.perf_counter()
 for s in range(0,len(flat),batch):rows.append(model(flat[s:s+batch].to(device))['embedding'].cpu())
 return torch.cat(rows).reshape(len(waves),8,-1).mean(1).numpy(),time.perf_counter()-start_time

def metric(y,p):
 r=recall_score(y,p,labels=[0,1,2],average=None,zero_division=0)
 return {'accuracy':float(np.mean(y==p)),'macro_f1':float(f1_score(y,p,labels=[0,1,2],average='macro',zero_division=0)),'recall_N':float(r[0]),'recall_IR':float(r[1]),'recall_OR':float(r[2]),'errors':int(np.sum(y!=p)),'confusion_matrix':confusion_matrix(y,p,labels=[0,1,2]).tolist()}

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('outputs/staged/PU_EIGHTBLOCK_FAIR_BASELINES'));p.add_argument('--protocol',type=Path,default=Path('RAMEF_FAIR_BASELINE_STATISTICS_PROTOCOL.md'));a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 source=sorted([r for r in scan_records(Path('D:/Paderborn')) if r['domain'] in SOURCE_DOMAINS],key=lambda r:r['measurement_id']);support,_=support_only(Path('split_independent_measurement.json'));query=load_query(Path('split_independent_measurement.json'));rows=source+support+query;waves=wave_cache(rows,a.output/'eightblock_waves.pt');ns=len(source);nt=len(support);y=np.asarray([int(r['label']) for r in rows]);weights=np.concatenate([np.ones(ns),np.full(nt,10.)]);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');results={};pred_rows=[];meta={}
 for mi,name in enumerate(MODELS):
  payload=torch.load(Path(f'outputs/staged/FAIR_BASELINE_PU/{name}.pt'),map_location='cpu',weights_only=False);model=make_model(name);model.load_state_dict(payload['state']);model.to(device);z,seconds=infer(model,waves,device);train=z[:ns+nt];sc=StandardScaler().fit(train);head=LogisticRegression(C=.1,max_iter=2000,solver='lbfgs',random_state=20268200+mi);head.fit(sc.transform(train),y[:ns+nt],sample_weight=weights);pred=head.predict(sc.transform(z[ns+nt:]));results[name]=metric(y[ns+nt:],pred);meta[name]={'parameters':sum(v.numel() for v in model.parameters()),'inference_seconds_all_blocks':seconds}
  for row,t,g in zip(query,y[ns+nt:],pred):pred_rows.append({'model':name,'measurement_id':row['measurement_id'],'bearing_code':row['bearing_code'],'true':int(t),'prediction':int(g),'correct':int(t==g)})
  del model;torch.cuda.empty_cache();print(name,results[name],flush=True)
 result={'experiment':'PU_EIGHTBLOCK_FAIR_BASELINES','epistemic_status':'post-hoc equal-observation-budget benchmark','results':results,'models':meta,'protocol_sha256':sha256(a.protocol),'split_sha256':sha256(Path('split_independent_measurement.json'))};
 with (a.output/'predictions.csv').open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=list(pred_rows[0]));w.writeheader();w.writerows(pred_rows)
 (a.output/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
