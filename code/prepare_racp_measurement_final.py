"""Freeze the support-selected RACP measurement-level candidate."""
from __future__ import annotations
import argparse,json,shutil
from pathlib import Path
import joblib,numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from bottleneck_deconvolution import sha256
from prepare_splits import scan_records
from run_a1_u2hz_confirmation import CHECKPOINT,CHECKPOINT_SHA
from run_mc_sbt_p0 import SOURCE_DOMAINS
from run_tser_hb_support_p0 import support_only

FILES=('racp_or.py','RACP_MEASUREMENT_FINAL_PROTOCOL.md','run_racp_measurement_final_once.py','u2_stft_model.py','run_core_component_ablation.py','run_a1_u2hz_confirmation.py')

def fit(x,y,w):
 sc=StandardScaler().fit(x);h=LogisticRegression(C=.1,max_iter=1500,solver='lbfgs',random_state=20268021);h.fit(sc.transform(x),y,sample_weight=w);return sc,h

def main():
 p=argparse.ArgumentParser();p.add_argument('--package',type=Path,default=Path('frozen/RACP_MEASUREMENT_FINAL_v1_20260821'));p.add_argument('--cache',type=Path,default=Path('outputs/staged/RACP_OR_INFORMATION_P0/authorised_source_support_features.npz'));a=p.parse_args()
 if a.package.exists():raise RuntimeError('Frozen RACP package already exists')
 if sha256(Path(CHECKPOINT))!=CHECKPOINT_SHA:raise RuntimeError('A1 checkpoint mismatch')
 source=sorted([r for r in scan_records(Path('D:/Paderborn')) if r['domain'] in SOURCE_DOMAINS],key=lambda r:r['measurement_id']);support,_=support_only(Path('split_independent_measurement.json'));c=np.load(a.cache,allow_pickle=False);ids=[r['measurement_id'] for r in source+support]
 if c['ids'].tolist()!=ids:raise RuntimeError('authorised feature cache mismatch')
 ns=len(source);y=np.asarray([int(r['label']) for r in source+support]);w=np.concatenate([np.ones(ns),np.full(15,10.)]);blocks=c['blocks'].astype(np.float32);centre=c['centre'].astype(np.float32);physical=c['physical'].astype(np.float32);racp=np.concatenate([blocks.mean(1),physical],1);single_sc,single_h=fit(centre,y,w);racp_sc,racp_h=fit(racp,y,w)
 a.package.mkdir(parents=True);joblib.dump({'A1_SINGLE':{'scaler':single_sc,'head':single_h},'RACP_MEASUREMENT':{'scaler':racp_sc,'head':racp_h}},a.package/'heads.joblib')
 code=a.package/'code';code.mkdir();
 for name in FILES:shutil.copy2(name,code/name)
 manifest={'release':'RACP_MEASUREMENT_FINAL_v1_20260821','selected_without_DS_query':True,'checkpoint_path':CHECKPOINT,'checkpoint_sha256':CHECKPOINT_SHA,'split_sha256':sha256(Path('split_independent_measurement.json')),'source_support_cache_sha256':sha256(a.cache),'heads_sha256':sha256(a.package/'heads.joblib'),'files':{name:sha256(Path(name)) for name in FILES},'source_measurements':ns,'target_support_measurements':15,'feature_dimensions':{'A1_SINGLE':int(centre.shape[1]),'RACP_MEASUREMENT':int(racp.shape[1])}}
 (a.package/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
