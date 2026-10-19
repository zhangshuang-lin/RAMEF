"""Train controlled U2 physics-front-end ablations and register locked stages."""
from __future__ import annotations

import argparse, copy, json, random, shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from bottleneck_deconvolution import sha256
from prepare_splits import scan_records
from run_c2of import metrics
from run_fair_baseline_pu import build_cache, fixed_source_split
from run_lcga_fresh_p0 import digest_text
from run_lcga_paper_main import aggregate
from run_mc_sbt_p0 import SOURCE_DOMAINS, write_csv
from run_paper_main import PROTOCOL_MANIFEST_SHA256, load_contract
from u2_stft_model import U2STFT


VARIANTS = {"A1_HZ": ("hz", False), "A2_HZ_ORDER": ("hz_order", False),
            "A3_CYCLIC": ("hz_order_cyclic", False),
            "A4_FULL_PHASE": ("full", False), "A5_PHASE_CF": ("full", True)}
RUN_VARIANTS = {k: v for k, v in VARIANTS.items() if k != "A5_PHASE_CF"}


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def make_u2(mode, config_path, num_classes=3):
    cfg=yaml.safe_load(Path(config_path).read_text())["model"]
    allowed={k:v for k,v in cfg.items() if k not in {"num_classes","lite","use_speed_adapter","use_iror_physics_branch","use_low_order_branch","support_conditioned_gate","late_fusion"}}
    allowed["frontend_ablation"]=mode
    return U2STFT(num_classes=num_classes,lite=True,use_speed_adapter=False,
                  use_iror_physics_branch=False,use_low_order_branch=False,
                  support_conditioned_gate=False,late_fusion=False,**allowed)


@torch.no_grad()
def infer(model,waves,rpms,device,batch=16):
    model.eval(); rows=[]
    for start in range(0,len(waves),batch):
        out=model(waves[start:start+batch].to(device),rpms[start:start+batch].to(device))
        rows.append(out["tokens"].flatten(1).cpu())
    return torch.cat(rows).numpy()


def train_variant(name,mode,use_cf,waves,labels,rpms,train_idx,val_idx,device,out,config_path):
    ckpt=out/f"{name}.pt"
    if ckpt.exists():
        p=torch.load(ckpt,map_location="cpu",weights_only=False); m=make_u2(mode,config_path);m.load_state_dict(p["state"]);return m.to(device),p["best_val_macro_f1"],p["history"]
    seed_all(20260825);m=make_u2(mode,config_path).to(device)
    epochs=12; learning_rate=1e-3
    if use_cf:
        a4=out/"A4_FULL_PHASE.pt"
        if not a4.exists(): raise RuntimeError("A5 requires the stable A4 checkpoint")
        m.load_state_dict(torch.load(a4,map_location="cpu",weights_only=False)["state"])
        # Controlled A4 never trained the counterfactual branch. Make the A5
        # transfer exactly identity at entry instead of exposing the encoder to
        # a random complex correction on its first batch.
        torch.nn.init.zeros_(m.counterfactual.coeff.weight)
        torch.nn.init.zeros_(m.counterfactual.coeff.bias)
        epochs=4; learning_rate=2e-4
    opt=torch.optim.AdamW(m.parameters(),lr=learning_rate,weight_decay=1e-4)
    ds=TensorDataset(waves[train_idx],torch.tensor(labels[train_idx]),rpms[train_idx]);loader=DataLoader(ds,batch_size=12,shuffle=True,generator=torch.Generator().manual_seed(20260825))
    best=-1.;state=None;history=[]
    for epoch in range(epochs):
        m.train();losses=[]
        for x,y,rpm in loader:
            x,y,rpm=x.to(device),y.to(device),rpm.to(device);opt.zero_grad(set_to_none=True)
            outp=m(x,rpm);loss=F.cross_entropy(outp["logits"],y)
            if use_cf and len(x)>1:
                perm=torch.roll(torch.arange(len(x),device=device),1);cf=m.make_counterfactual(outp,perm)
                loss=loss+.1*F.cross_entropy(cf["logits"],y)+.1*F.mse_loss(F.normalize(cf["embedding"],dim=1),F.normalize(outp["embedding"].detach(),dim=1))
            loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5.0);opt.step();losses.append(float(loss.detach()))
        m.eval();pred=[]
        with torch.no_grad():
            for start in range(0,len(val_idx),16):
                idx=val_idx[start:start+16];pred.extend(m(waves[idx].to(device),rpms[idx].to(device))["logits"].argmax(1).cpu().tolist())
        score=float(f1_score(labels[val_idx],pred,labels=[0,1,2],average="macro",zero_division=0));history.append({"variant":name,"epoch":epoch,"train_loss":float(np.mean(losses)),"val_macro_f1":score})
        if score>best:best=score;state={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
    m.load_state_dict(state);torch.save({"state":state,"best_val_macro_f1":best,"history":history},ckpt);return m,best,history


def main():
    pa=argparse.ArgumentParser();pa.add_argument("--output",type=Path,default=Path("outputs/staged/CORE_COMPONENT_ABLATION_R3"));pa.add_argument("--protocol",type=Path,default=Path("CORE_COMPONENT_ABLATION_R3_PROTOCOL.md"));args=pa.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/"evaluation.lock").exists():raise RuntimeError("Core ablation locked")
    frozen=args.output/"protocol_frozen.md"
    if frozen.exists() and digest_text(frozen)!=digest_text(args.protocol):raise RuntimeError("Protocol mismatch")
    if not frozen.exists():shutil.copy2(args.protocol,frozen)
    paper=Path("paper/evidence/protocol/PAPER_MAIN_V1")
    if sha256(paper/"MANIFEST.sha256")!=PROTOCOL_MANIFEST_SHA256:raise RuntimeError("PAPER_MAIN mismatch")
    query,supports,_=load_contract(paper);records=scan_records(Path("D:/Paderborn"));source,target,waves=build_cache(records,query,supports,Path("outputs/staged/FAIR_BASELINE_PU/waveforms.pt"))
    ns=len(source);labels=np.asarray([int(x["label"]) for x in source]);rpms=torch.tensor([float(x["rpm"]) for x in source+target],dtype=torch.float32);train_idx,val_idx=fixed_source_split(source);target_idx={x["measurement_id"]:ns+i for i,x in enumerate(target)};device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config=Path("frozen/M2_LOCKED_v1_20260715/model/resolved_config.yaml");histories=[];episode_rows=[];meta={}
    legacy=Path("outputs/staged/CORE_COMPONENT_ABLATION")
    for name in ("A1_HZ","A2_HZ_ORDER","A3_CYCLIC","A4_FULL_PHASE"):
        if not (args.output/f"{name}.pt").exists() and (legacy/f"{name}.pt").exists():
            shutil.copy2(legacy/f"{name}.pt",args.output/f"{name}.pt")
    for name,(mode,use_cf) in RUN_VARIANTS.items():
        model,best,history=train_variant(name,mode,use_cf,waves[:ns],labels,rpms[:ns],train_idx,val_idx,device,args.output,config);histories.extend(history);embedding=infer(model,waves,rpms,device);meta[name]={"mode":mode,"counterfactual":use_cf,"source_val_macro_f1":best,"parameters":sum(p.numel() for p in model.parameters())}
        for episode,support in enumerate(supports):
            ti=np.asarray([target_idx[x["measurement_id"]] for x in support]);qi=np.asarray([target_idx[x["measurement_id"]] for x in query]);ty=np.asarray([int(x["label"]) for x in support]);qy=np.asarray([int(x["label"]) for x in query]);codes=[x["bearing_code"] for x in query]
            x=np.vstack([embedding[:ns],embedding[ti]]);y=np.concatenate([labels,ty]);sc=StandardScaler().fit(x);head=LogisticRegression(C=.1,max_iter=4000,solver="lbfgs",random_state=20260825+episode);head.fit(sc.transform(x),y,sample_weight=np.concatenate([np.ones(ns),np.full(len(ti),10.)]));pred=head.predict(sc.transform(embedding[qi]));value=metrics(qy,pred,codes);value["recall_N"],value["recall_IR"],value["recall_OR"]=value["recall"];episode_rows.append({"variant":name,"episode":episode,"value":value})
        del model;torch.cuda.empty_cache()
    controlled={name:aggregate([x for x in episode_rows if x["variant"]==name],"value") for name in RUN_VARIANTS}
    fair=json.loads(Path("outputs/staged/FAIR_BASELINE_PU/result.json").read_text());locked={"A0_STFT_CNN":fair["summaries"]["stft_cnn"]["source_support_linear"],"A5_LOCKED_M2":fair["locked_method_comparators"]["M2"],"A6_UNGATED":fair["locked_method_comparators"]["PU_UNGATED_IO"],"A7_LCGA_S":fair["locked_method_comparators"]["LCGA_S"]}
    result={"experiment":"CORE_COMPONENT_ABLATION","query_used_for_selection":False,"controlled_meta":meta,"controlled":controlled,"registered_locked_stages":locked,
            "controlled_A5_status":"NOT_ESTIMABLE_nonfinite_complex_counterfactual_gradient_before_target_metrics",
            "comparability_warning":"A0-A4 controlled share source CE; locked M2 adds counterfactual/episodic/speed training"}
    write_csv(args.output/"training_history.csv",histories);path=args.output/"result.json";path.write_text(json.dumps(result,indent=2),encoding="utf-8");(args.output/"evaluation.lock").write_text(json.dumps({"completed":True,"result_sha256":sha256(path)},indent=2),encoding="utf-8");print(json.dumps(result,indent=2))


if __name__=="__main__":main()
