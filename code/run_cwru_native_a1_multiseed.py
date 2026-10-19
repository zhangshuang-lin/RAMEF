"""Five-backbone-seed confirmation of CWRU-native U2-Hz."""
from __future__ import annotations
import argparse,csv,json,random
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader,TensorDataset

from audit_x0b_phase_topology import sha256,write_csv
from run_core_component_ablation import make_u2
from run_cwru_10class_baseline import coarse_metric,metric,record_metric
from run_cwru_native_u2 import build_waves,embed

SEEDS=(20260828,20260829,20260830,20260831,20260832)

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)

def train(target,seed,waves,labels,rpms,pool,source,device,out):
    path=out/f'native_a1_T{target}_seed{seed}.pt'
    if path.exists():
        payload=torch.load(path,map_location='cpu',weights_only=False);model=make_u2('hz','frozen/M2_LOCKED_v1_20260715/model/resolved_config.yaml',10);model.load_state_dict(payload['state']);return model.to(device),payload
    train_idx=np.asarray([i for i in source if int(pool[i]['raw_window_index'])%5!=0]);val=np.asarray([i for i in source if int(pool[i]['raw_window_index'])%5==0])
    seed_all(seed+target*100);model=make_u2('hz','frozen/M2_LOCKED_v1_20260715/model/resolved_config.yaml',10).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    loader=DataLoader(TensorDataset(waves[train_idx],torch.tensor(labels[train_idx]),rpms[train_idx]),batch_size=12,shuffle=True,generator=torch.Generator().manual_seed(seed+target*100))
    best=-1.;state=None;history=[]
    for epoch in range(12):
        model.train();losses=[]
        for x,y,r in loader:
            x,y,r=x.to(device),y.to(device),r.to(device);optimizer.zero_grad(set_to_none=True);loss=F.cross_entropy(model(x,r)['logits'],y);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step();losses.append(float(loss.detach()))
        model.eval();prediction=[]
        with torch.no_grad():
            for start in range(0,len(val),24):prediction.extend(model(waves[val[start:start+24]].to(device),rpms[val[start:start+24]].to(device))['logits'].argmax(1).cpu().tolist())
        score=float(f1_score(labels[val],prediction,labels=list(range(10)),average='macro',zero_division=0));history.append({'epoch':epoch,'train_loss':float(np.mean(losses)),'source_val_macro_f1':score})
        if score>best:best=score;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(state);payload={'state':state,'best':best,'history':history};torch.save(payload,path);return model,payload

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=Path('outputs/staged/CWRU_NATIVE_A1_MULTISEED'));parser.add_argument('--protocol',type=Path,default=Path('CWRU_NATIVE_A1_MULTISEED_PROTOCOL.md'));args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    split_path=Path('cwru_10class_splits.json');split=json.loads(split_path.read_text());split_hash=sha256(split_path)
    with Path('cwru_10class_window_pool.csv').open(encoding='utf-8',newline='') as stream:pool=list(csv.DictReader(stream))
    index={x['window_id']:i for i,x in enumerate(pool)};labels=np.asarray([int(x['label']) for x in pool]);rpms=torch.tensor([float(x['rpm']) for x in pool]);waves=build_waves(pool,args.output/'waves.pt');device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');rows=[];models=[]
    for seed in SEEDS:
        for target in range(4):
            target_episodes=[x for x in split['episodes'] if int(x['target_load'])==target];source=np.asarray([index[x] for x in target_episodes[0]['source_window_ids']])
            model,payload=train(target,seed,waves,labels,rpms,pool,source,device,args.output);features=embed(model,waves,rpms,device);models.append({'seed':seed,'target_load':target,'source_val_macro_f1':payload['best']})
            for episode in target_episodes:
                support=np.asarray([index[x] for x in episode['support_window_ids']]);query=np.asarray([index[x] for x in episode['query_window_ids']]);x=np.vstack((features[source],features[support]));y=np.concatenate((labels[source],labels[support]));scaler=StandardScaler().fit(x);head=LogisticRegression(C=.1,max_iter=4000,solver='lbfgs',random_state=seed+int(episode['seed']));head.fit(scaler.transform(x),y,sample_weight=np.concatenate((np.ones(len(source)),np.full(len(support),10.))));logits=head.decision_function(scaler.transform(features[query]));prediction=logits.argmax(1);value=metric(labels[query],prediction);coarse=coarse_metric(labels[query],prediction);record=record_metric([pool[i] for i in query],logits,labels[query]);rows.append({'backbone_seed':seed,'target_load':target,'episode_id':episode['episode_id'],'support_seed':int(episode['seed']),'accuracy':value['accuracy'],'macro_f1':value['macro_f1'],'coarse_accuracy':coarse['accuracy'],'record_accuracy':record['accuracy'],'errors':value['errors']})
            del model;torch.cuda.empty_cache()
    def stats(values):return {'mean':float(np.mean(values)),'std':float(np.std(values,ddof=1)),'min':float(np.min(values)),'max':float(np.max(values))}
    overall={key:stats([x[key] for x in rows]) for key in ('accuracy','macro_f1','coarse_accuracy','record_accuracy')};per_seed={str(seed):stats([x['accuracy'] for x in rows if x['backbone_seed']==seed]) for seed in SEEDS};per_target={str(target):stats([x['accuracy'] for x in rows if x['target_load']==target]) for target in range(4)}
    gates={'mean_accuracy_ge_0_99':overall['accuracy']['mean']>=.99,'worst_seed_mean_ge_0_98':min(x['mean'] for x in per_seed.values())>=.98,'all_target_means_ge_0_98':min(x['mean'] for x in per_target.values())>=.98}
    result={'experiment':'CWRU_NATIVE_A1_MULTISEED','status':'PASS' if all(gates.values()) else 'FAIL','query_used_for_selection':False,'random_initialization':True,'evaluations':len(rows),'backbone_models':models,'overall':overall,'per_backbone_seed':per_seed,'per_target_load':per_target,'gates':gates,'protocol_sha256':sha256(args.protocol),'split_sha256':split_hash}
    write_csv(args.output/'episode_metrics.csv',rows);(args.output/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
