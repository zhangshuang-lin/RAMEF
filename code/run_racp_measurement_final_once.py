"""One-time DS-query evaluation of the frozen RACP measurement candidate."""
from __future__ import annotations
import argparse,csv,json
from collections import defaultdict
from pathlib import Path
import joblib,numpy as np,torch
from sklearn.metrics import confusion_matrix,f1_score,recall_score
from bottleneck_deconvolution import sha256
from run_a1_u2hz_confirmation import CHECKPOINT,CHECKPOINT_SHA
from run_racp_or_information_p0 import extract_cache

def write_csv(path,rows):
 with path.open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def load_query(split):
 p=json.loads(split.read_text(encoding='utf-8'));s=[r for r in p['windows'] if r.get('domain')=='DS' and r.get('role')=='support'];q=[r for r in p['windows'] if r.get('domain')=='DS' and r.get('role')=='query']
 if len(s)!=15 or len(q)!=240 or {r['measurement_id'] for r in s}&{r['measurement_id'] for r in q}:raise RuntimeError('target contract violation')
 return q

def metric(y,p):
 r=recall_score(y,p,labels=[0,1,2],average=None,zero_division=0)
 return {'accuracy':float(np.mean(y==p)),'macro_f1':float(f1_score(y,p,labels=[0,1,2],average='macro',zero_division=0)),'recall_N':float(r[0]),'recall_IR':float(r[1]),'recall_OR':float(r[2]),'confusion_matrix':confusion_matrix(y,p,labels=[0,1,2]).tolist(),'errors':int(np.sum(y!=p))}

def main():
 p=argparse.ArgumentParser();p.add_argument('--package',type=Path,default=Path('frozen/RACP_MEASUREMENT_FINAL_v1_20260821'));p.add_argument('--output',type=Path,default=Path('outputs/staged/RACP_MEASUREMENT_FINAL'));a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);lock=a.output/'target_evaluation.lock'
 if lock.exists():raise RuntimeError('RACP measurement target evaluation already locked')
 manifest=json.loads((a.package/'manifest.json').read_text(encoding='utf-8'))
 if sha256(Path(CHECKPOINT))!=CHECKPOINT_SHA or manifest['checkpoint_sha256']!=CHECKPOINT_SHA:raise RuntimeError('A1 freeze mismatch')
 if sha256(Path('split_independent_measurement.json'))!=manifest['split_sha256']:raise RuntimeError('split freeze mismatch')
 if sha256(a.package/'heads.joblib')!=manifest['heads_sha256']:raise RuntimeError('head freeze mismatch')
 for name,digest in manifest['files'].items():
  if sha256(Path(name))!=digest:raise RuntimeError(f'live code changed after freeze: {name}')
 query=load_query(Path('split_independent_measurement.json'));device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');feature_dir=a.output/'query_features';feature_dir.mkdir();cache=extract_cache(query,feature_dir,device);blocks=cache['blocks'].astype(np.float32);centre=cache['centre'].astype(np.float32);physical=cache['physical'].astype(np.float32);racp=np.concatenate([blocks.mean(1),physical],1);heads=joblib.load(a.package/'heads.joblib');y=np.asarray([int(r['label']) for r in query]);predictions={};results={};rows=[]
 for name,x in (('A1_SINGLE',centre),('RACP_MEASUREMENT',racp)):
  item=heads[name];pred=item['head'].predict(item['scaler'].transform(x));predictions[name]=pred;results[name]=metric(y,pred)
 for i,row in enumerate(query):rows.append({'measurement_id':row['measurement_id'],'bearing_code':row['bearing_code'],'true':int(y[i]),'A1_SINGLE':int(predictions['A1_SINGLE'][i]),'RACP_MEASUREMENT':int(predictions['RACP_MEASUREMENT'][i]),'A1_correct':int(predictions['A1_SINGLE'][i]==y[i]),'RACP_correct':int(predictions['RACP_MEASUREMENT'][i]==y[i])})
 by_code=[]
 for code in sorted({r['bearing_code'] for r in query}):
  idx=np.asarray([i for i,r in enumerate(query) if r['bearing_code']==code]);truth=int(y[idx[0]]);b=predictions['A1_SINGLE'][idx];c=predictions['RACP_MEASUREMENT'][idx];by_code.append({'bearing_code':code,'true':truth,'count':len(idx),'A1_accuracy':float(np.mean(b==truth)),'RACP_accuracy':float(np.mean(c==truth)),'RACP_pred_N':int(np.sum(c==0)),'RACP_pred_IR':int(np.sum(c==1)),'RACP_pred_OR':int(np.sum(c==2))})
 r=results['RACP_MEASUREMENT'];g={'accuracy_gt_90':r['accuracy']>.9,'macro_f1_ge_895':r['macro_f1']>=.895,'N_recall_ge_95':r['recall_N']>=.95,'IR_recall_ge_83':r['recall_IR']>=.83,'OR_recall_ge_82':r['recall_OR']>=.82,'accuracy_exceeds_A1':r['accuracy']>results['A1_SINGLE']['accuracy']};result={'experiment':'RACP_MEASUREMENT_FINAL','status':'PASS' if all(g.values()) else 'FAIL','epistemic_status':'support-selected frozen candidate; project DS query historically exposed to other experiments','observation_unit':'measurement with 8 non-overlapping 16384-sample blocks','results':results,'fixes':int(np.sum((predictions['A1_SINGLE']!=y)&(predictions['RACP_MEASUREMENT']==y))),'new_errors':int(np.sum((predictions['A1_SINGLE']==y)&(predictions['RACP_MEASUREMENT']!=y))),'gates':g,'package_manifest_sha256':sha256(a.package/'manifest.json'),'query_feature_cache_sha256':sha256(feature_dir/'authorised_source_support_features.npz'),'query_used_for_candidate_selection':False}
 write_csv(a.output/'predictions.csv',rows);write_csv(a.output/'bearing_code_results.csv',by_code);path=a.output/'result.json';path.write_text(json.dumps(result,indent=2),encoding='utf-8');lock.write_text(json.dumps({'completed':True,'result_sha256':sha256(path),'package_manifest_sha256':sha256(a.package/'manifest.json')},indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
