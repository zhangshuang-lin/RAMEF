"""All-pairs support perturbation audit for the fixed RACP linear candidate."""
from __future__ import annotations
import argparse,csv,itertools,json
from pathlib import Path
import numpy as np
from sklearn.metrics import confusion_matrix,f1_score,recall_score
from bottleneck_deconvolution import sha256
from prepare_splits import scan_records
from run_mc_sbt_p0 import SOURCE_DOMAINS
from run_tser_hb_support_p0 import linear_predict,support_only

def write_csv(path,rows):
 with path.open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def metric(y,p):
 r=recall_score(y,p,labels=[0,1,2],average=None,zero_division=0)
 return {'accuracy':float(np.mean(y==p)),'macro_f1':float(f1_score(y,p,labels=[0,1,2],average='macro',zero_division=0)),'recall_N':float(r[0]),'recall_IR':float(r[1]),'recall_OR':float(r[2]),'N_to_OR':int(np.sum((y==0)&(p==2))),'confusion_matrix':confusion_matrix(y,p,labels=[0,1,2]).tolist()}

def main():
 pa=argparse.ArgumentParser();pa.add_argument('--output',type=Path,default=Path('outputs/staged/RACP_LINEAR_SUPPORT_STABILITY'));pa.add_argument('--protocol',type=Path,default=Path('RACP_LINEAR_SUPPORT_STABILITY_PROTOCOL.md'));pa.add_argument('--cache',type=Path,default=Path('outputs/staged/RACP_OR_INFORMATION_P0/authorised_source_support_features.npz'));a=pa.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 source=sorted([r for r in scan_records(Path('D:/Paderborn')) if r['domain'] in SOURCE_DOMAINS],key=lambda r:r['measurement_id']);support,qcount=support_only(Path('split_independent_measurement.json'));cache=np.load(a.cache,allow_pickle=False);ids=[r['measurement_id'] for r in source+support]
 if cache['ids'].tolist()!=ids:raise RuntimeError('cache mismatch')
 ns=len(source);sy=np.asarray([int(r['label']) for r in source]);ty=np.asarray([int(r['label']) for r in support]);blocks=cache['blocks'].astype(np.float32);centre=cache['centre'].astype(np.float32);physical=cache['physical'].astype(np.float32);racp=np.concatenate([blocks.mean(1),physical],1)
 folds=[];predictions=[]
 for fold,pair in enumerate(itertools.combinations(range(15),2)):
  held=np.asarray(pair);keep=np.asarray([i for i in range(15) if i not in pair]);base=linear_predict(centre[:ns],sy,centre[ns:][keep],ty[keep],centre[ns:][held],20263000+fold);cand=linear_predict(racp[:ns],sy,racp[ns:][keep],ty[keep],racp[ns:][held],20264000+fold);ba=float(np.mean(base==ty[held]));ca=float(np.mean(cand==ty[held]));folds.append({'fold':fold,'held':'+'.join(support[i]['measurement_id'] for i in held),'base_accuracy':ba,'candidate_accuracy':ca,'candidate_not_lower':int(ca>=ba)})
  for i,b,c in zip(held,base,cand):predictions.append({'fold':fold,'measurement_id':support[i]['measurement_id'],'bearing_code':support[i]['bearing_code'],'true':int(ty[i]),'A1_SINGLE':int(b),'RACP_LINEAR':int(c)})
 y=np.asarray([r['true'] for r in predictions]);p=np.asarray([r['RACP_LINEAR'] for r in predictions]);value=metric(y,p);rate=float(np.mean([r['candidate_not_lower'] for r in folds]));g={'accuracy_ge_90':value['accuracy']>=.9,'N_recall_100':value['recall_N']==1.,'OR_recall_ge_80':value['recall_OR']>=.8,'zero_N_to_OR':value['N_to_OR']==0,'nonlower_folds_ge_80pct':rate>=.8};result={'experiment':'RACP_LINEAR_SUPPORT_STABILITY','status':'PASS' if all(g.values()) else 'FAIL','candidate_freeze_authorised':bool(all(g.values())),'metrics':value,'nonlower_fold_rate':rate,'gates':g,'firewall':{'query_rows_in_split':qcount,'query_measurements_loaded':0,'query_waveforms_loaded':False,'query_labels_used':False},'protocol_sha256':sha256(a.protocol),'cache_sha256':sha256(a.cache)};write_csv(a.output/'folds.csv',folds);write_csv(a.output/'predictions.csv',predictions);(a.output/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
