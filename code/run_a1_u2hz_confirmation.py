"""Evaluate the immutable A1 U2-Hz candidate on PU K5 and CWRU ten-class tasks."""
from __future__ import annotations
import argparse,csv,json,shutil
from pathlib import Path
import numpy as np,torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from bottleneck_deconvolution import load_contract,sha256
from prepare_splits import scan_records
from run_core_component_ablation import make_u2
from run_cwru_10class_baseline import aggregate as cwru_aggregate,coarse_metric,extract_features,metric as metric10,record_metric
from run_fair_baseline_pu import build_cache,waveform
from run_lcga_fresh_p0 import digest_text
from run_mc_sbt_p0 import SOURCE_DOMAINS,metric as metric3,write_csv

CHECKPOINT='frozen/U2HZ_A1_LOCKED_v1_20260817/outputs/staged/CORE_COMPONENT_ABLATION/A1_HZ.pt'
CHECKPOINT_SHA='5e48cd1b65212f6493602577067d4f7c8f39ef5fe03e2f2ecaae0e494c98f07b'

@torch.no_grad()
def embed(model,waves,rpms,device,batch=16):
 model.eval();rows=[]
 for s in range(0,len(waves),batch):rows.append(model(waves[s:s+batch].to(device),rpms[s:s+batch].to(device))['tokens'].flatten(1).cpu())
 return torch.cat(rows).numpy()

def diagonal(sx,sy,qx,classes):
 means=[];vars=[]
 for c in range(classes):v=sx[sy==c];means.append(v.mean(0));vars.append(v.var(0)+.05)
 return -(((qx[:,None]-np.stack(means)[None])**2/np.stack(vars)[None]).mean(2))

def fit_head(source_x,source_y,support_x,support_y,query_x,seed):
 x=np.vstack([source_x,support_x]);y=np.concatenate([source_y,support_y]);sc=StandardScaler().fit(x);m=LogisticRegression(C=.1,max_iter=4000,solver='lbfgs',random_state=seed);m.fit(sc.transform(x),y,sample_weight=np.concatenate([np.ones(len(source_x)),np.full(len(support_x),10.)]));return m.decision_function(sc.transform(query_x))

def main():
 pa=argparse.ArgumentParser();pa.add_argument('--output',type=Path,default=Path('outputs/staged/A1_U2HZ_CONFIRMATION'));pa.add_argument('--protocol',type=Path,default=Path('A1_U2HZ_CONFIRMATION_PROTOCOL.md'));args=pa.parse_args();args.output.mkdir(parents=True,exist_ok=True)
 if (args.output/'evaluation.lock').exists():raise RuntimeError('A1 confirmation locked')
 frozen=args.output/'protocol_frozen.md'
 if frozen.exists() and digest_text(frozen)!=digest_text(args.protocol):raise RuntimeError('protocol mismatch')
 if not frozen.exists():shutil.copy2(args.protocol,frozen)
 if sha256(CHECKPOINT)!=CHECKPOINT_SHA:raise RuntimeError('A1 checkpoint mismatch')
 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');model=make_u2('hz','frozen/M2_LOCKED_v1_20260715/model/resolved_config.yaml');p=torch.load(CHECKPOINT,map_location='cpu',weights_only=False);model.load_state_dict(p['state']);model.to(device)
 # PU K=5
 records=scan_records(Path('D:/Paderborn'));source=sorted([x for x in records if x['domain'] in SOURCE_DOMAINS],key=lambda x:x['measurement_id']);_,support,query=load_contract(Path('split_independent_measurement.json'))
 def centered(rows):
  out=[]
  for x in rows:
   start=int(x.get('signal_length',256001))//2-8192;out.append({**x,'start':start,'end':start+16384})
  return out
 allrows=centered(source)+centered(support)+centered(query);waves=torch.from_numpy(np.stack([waveform(x) for x in allrows]));rpms=torch.tensor([float(x['rpm']) for x in allrows]);z=embed(model,waves,rpms,device);ns,nt=len(source),len(support);sx,tx,qx=z[:ns],z[ns:ns+nt],z[ns+nt:];sy=np.asarray([int(x['label']) for x in source]);ty=np.asarray([int(x['label']) for x in support]);qy=np.asarray([int(x['label']) for x in query])
 pu_logits={'target_diagonal_proto':diagonal(tx,ty,qx,3),'source_support_linear':fit_head(sx,sy,tx,ty,qx,20260827)};pu={}
 for name,logits in pu_logits.items():pu[name]=metric3(qy,logits.argmax(1))
 # CWRU
 split=json.loads(Path('cwru_10class_splits.json').read_text());split_hash=sha256('cwru_10class_splits.json')
 with Path('cwru_10class_window_pool.csv').open(encoding='utf-8',newline='') as f:pool=list(csv.DictReader(f));idx={x['window_id']:i for i,x in enumerate(pool)};labels=np.asarray([int(x['label']) for x in pool])
 features=extract_features(pool,model,device,args.output/'cwru_a1_tokens.pt',split_hash,24).flatten(1).float().numpy();rows=[]
 for ep in split['episodes']:
  si=np.asarray([idx[x] for x in ep['source_window_ids']]);ti=np.asarray([idx[x] for x in ep['support_window_ids']]);qi=np.asarray([idx[x] for x in ep['query_window_ids']]);qy10=labels[qi];logits=fit_head(features[si],labels[si],features[ti],labels[ti],features[qi],20260827+int(ep['seed']));pred=logits.argmax(1);m=metric10(qy10,pred);c=coarse_metric(qy10,pred);r=record_metric([pool[i] for i in qi],logits,qy10);rows.append({'method':'A1_U2HZ','target_load':ep['target_load'],'accuracy':m['accuracy'],'macro_f1':m['macro_f1'],'recall':m['recall'],'confusion_matrix':m['confusion_matrix'],'coarse_accuracy':c['accuracy'],'coarse_macro_f1':c['macro_f1'],'record_accuracy':r['accuracy']})
 cwru=cwru_aggregate(rows,'A1_U2HZ');per={str(t):cwru_aggregate([x for x in rows if int(x['target_load'])==t],'A1_U2HZ') for t in range(4)}
 result={'experiment':'A1_U2HZ_FROZEN_CONFIRMATION','candidate_checkpoint_sha256':CHECKPOINT_SHA,'candidate_modified_after_selection':False,'pu_k5':pu,'cwru_10class':cwru,'cwru_per_target':per,'query_used_for_tuning':False};path=args.output/'result.json';path.write_text(json.dumps(result,indent=2),encoding='utf-8');(args.output/'evaluation.lock').write_text(json.dumps({'completed':True,'result_sha256':sha256(path)},indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
