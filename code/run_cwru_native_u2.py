"""CWRU-native controlled training of U2-Hz and full physics representations."""
from __future__ import annotations
import argparse,csv,json,random,shutil
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
from scipy.io import loadmat
from scipy.signal import resample_poly
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader,TensorDataset
from bottleneck_deconvolution import sha256
from run_core_component_ablation import make_u2
from run_cwru_10class_baseline import aggregate,coarse_metric,metric,record_metric
from run_lcga_fresh_p0 import digest_text

VARIANTS={'NATIVE_HZ':'hz','NATIVE_FULL':'full'}
def seed_all(seed):random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
def build_waves(pool,path):
 ids=[x['window_id'] for x in pool]
 if path.exists():
  p=torch.load(path,map_location='cpu',weights_only=False)
  if p['window_ids']!=ids:raise RuntimeError('stale CWRU wave cache')
  return p['waves'].float()
 signals={};values=[]
 for row in pool:
  key=(row['path'],row['signal_key'])
  if key not in signals:signals[key]=np.asarray(loadmat(row['path'],variable_names=[row['signal_key']])[row['signal_key']]).squeeze()
  raw=np.asarray(signals[key][int(row['raw_start']):int(row['raw_end'])],dtype=np.float32);wave=resample_poly(raw,16,3).astype(np.float32);wave=(wave-wave.mean())/max(float(wave.std()),1e-6);values.append(wave)
 result=torch.from_numpy(np.stack(values)).half();torch.save({'window_ids':ids,'waves':result},path);return result.float()
@torch.no_grad()
def embed(model,waves,rpms,device,batch=16):
 model.eval();rows=[]
 for s in range(0,len(waves),batch):rows.append(model(waves[s:s+batch].to(device),rpms[s:s+batch].to(device))['tokens'].flatten(1).cpu())
 return torch.cat(rows).numpy()
def train_one(name,mode,target,waves,labels,rpms,pool,source_idx,device,out):
 ck=out/f'{name}_T{target}.pt'
 if ck.exists():p=torch.load(ck,map_location='cpu',weights_only=False);m=make_u2(mode,'frozen/M2_LOCKED_v1_20260715/model/resolved_config.yaml',10);m.load_state_dict(p['state']);return m.to(device),p['best'],p['history']
 train=np.asarray([i for i in source_idx if int(pool[i]['raw_window_index'])%5!=0]);val=np.asarray([i for i in source_idx if int(pool[i]['raw_window_index'])%5==0]);seed_all(20260828+target);m=make_u2(mode,'frozen/M2_LOCKED_v1_20260715/model/resolved_config.yaml',10).to(device);opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-4);loader=DataLoader(TensorDataset(waves[train],torch.tensor(labels[train]),rpms[train]),batch_size=12,shuffle=True,generator=torch.Generator().manual_seed(20260828+target));best=-1.;state=None;history=[]
 for epoch in range(12):
  m.train();losses=[]
  for x,y,r in loader:
   x,y,r=x.to(device),y.to(device),r.to(device);opt.zero_grad(set_to_none=True);o=m(x,r);loss=F.cross_entropy(o['logits'],y);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5.);opt.step();losses.append(float(loss.detach()))
  m.eval();pred=[]
  with torch.no_grad():
   for s in range(0,len(val),16):pred.extend(m(waves[val[s:s+16]].to(device),rpms[val[s:s+16]].to(device))['logits'].argmax(1).cpu().tolist())
  score=float(f1_score(labels[val],pred,labels=list(range(10)),average='macro',zero_division=0));history.append({'variant':name,'target_load':target,'epoch':epoch,'train_loss':float(np.mean(losses)),'val_macro_f1':score})
  if score>best:best=score;state={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
 m.load_state_dict(state);torch.save({'state':state,'best':best,'history':history},ck);return m,best,history
def main():
 pa=argparse.ArgumentParser();pa.add_argument('--output',type=Path,default=Path('outputs/staged/CWRU_NATIVE_U2'));pa.add_argument('--protocol',type=Path,default=Path('CWRU_NATIVE_U2_PROTOCOL.md'));args=pa.parse_args();args.output.mkdir(parents=True,exist_ok=True)
 if (args.output/'evaluation.lock').exists():raise RuntimeError('CWRU native locked')
 frozen=args.output/'protocol_frozen.md'
 if frozen.exists() and digest_text(frozen)!=digest_text(args.protocol):raise RuntimeError('protocol mismatch')
 if not frozen.exists():shutil.copy2(args.protocol,frozen)
 split=json.loads(Path('cwru_10class_splits.json').read_text());
 with Path('cwru_10class_window_pool.csv').open(encoding='utf-8',newline='') as f:pool=list(csv.DictReader(f))
 idx={x['window_id']:i for i,x in enumerate(pool)};labels=np.asarray([int(x['label']) for x in pool]);rpms=torch.tensor([float(x['rpm']) for x in pool]);waves=build_waves(pool,args.output/'waves.pt');device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');rows=[];history=[];meta={}
 for target in range(4):
  template=next(x for x in split['episodes'] if int(x['target_load'])==target);si=np.asarray([idx[x] for x in template['source_window_ids']])
  for name,mode in VARIANTS.items():
   model,best,h=train_one(name,mode,target,waves,labels,rpms,pool,si,device,args.output);history.extend(h);z=embed(model,waves,rpms,device);meta[f'{name}_T{target}']={'source_val_macro_f1':best,'parameters':sum(p.numel() for p in model.parameters())}
   for ep in [x for x in split['episodes'] if int(x['target_load'])==target]:
    ti=np.asarray([idx[x] for x in ep['support_window_ids']]);qi=np.asarray([idx[x] for x in ep['query_window_ids']]);x=np.vstack([z[si],z[ti]]);y=np.concatenate([labels[si],labels[ti]]);sc=StandardScaler().fit(x);head=LogisticRegression(C=.1,max_iter=4000,solver='lbfgs',random_state=20260828+int(ep['seed']));head.fit(sc.transform(x),y,sample_weight=np.concatenate([np.ones(len(si)),np.full(len(ti),10.)]));logits=head.decision_function(sc.transform(z[qi]));pred=logits.argmax(1);m=metric(labels[qi],pred);c=coarse_metric(labels[qi],pred);r=record_metric([pool[i] for i in qi],logits,labels[qi]);rows.append({'method':name,'target_load':target,'episode_id':ep['episode_id'],'accuracy':m['accuracy'],'macro_f1':m['macro_f1'],'recall':m['recall'],'confusion_matrix':m['confusion_matrix'],'coarse_accuracy':c['accuracy'],'coarse_macro_f1':c['macro_f1'],'record_accuracy':r['accuracy']})
   del model;torch.cuda.empty_cache()
 summary={name:aggregate(rows,name) for name in VARIANTS};per={str(t):{name:aggregate([x for x in rows if int(x['target_load'])==t],name) for name in VARIANTS} for t in range(4)};hz={x['episode_id']:x for x in rows if x['method']=='NATIVE_HZ'};delta=[x['accuracy']-hz[x['episode_id']]['accuracy'] for x in rows if x['method']=='NATIVE_FULL'];result={'experiment':'CWRU_NATIVE_U2','query_used_for_selection':False,'models':meta,'summary':summary,'per_target':per,'full_minus_hz_accuracy_delta_mean':float(np.mean(delta)),'full_better_episodes':int(np.sum(np.asarray(delta)>0)),'ties':int(np.sum(np.asarray(delta)==0))};Path(args.output/'training_history.json').write_text(json.dumps(history,indent=2));path=args.output/'result.json';path.write_text(json.dumps(result,indent=2));(args.output/'evaluation.lock').write_text(json.dumps({'completed':True,'result_sha256':sha256(path)},indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
