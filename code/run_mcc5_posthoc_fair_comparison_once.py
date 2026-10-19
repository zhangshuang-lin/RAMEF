"""One separately locked, post-target MCC5 fairness event with frozen heads.

No .fit method or training step is called. The original target artifacts and
the preparation-state packets remain unchanged. This event cannot create a
new untouched confirmation cohort.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from capacity_matched_baselines import WidthMatchedWDCNN
from fair_baseline_models import make_model as make_wdcnn
from mcc5_confirmatory_data import sha256
from run_mcc5_capacity_dimension_p0 import infer_blocks, metric, write_csv
from run_mcc5_ramef_confirmatory import SEEDS, load_model

ROOT=Path(__file__).resolve().parent
BASE=ROOT/"outputs/staged/MCC5_RAMEF_CONFIRMATORY"
T2=ROOT/"outputs/staged/MCC5_TRANSFERRED_EVIDENCE_PACKAGE/T2_benchmark"
P0=ROOT/"outputs/staged/MCC5_CAPACITY_DIMENSION_P0"
PREP=ROOT/"outputs/staged/MCC5_POSTHOC_FAIR_COMPARISON_PREPARED"
OUT=ROOT/"outputs/staged/MCC5_POSTHOC_FAIR_COMPARISON"
AMENDMENT=ROOT/"MCC5_POSTHOC_FAIR_COMPARISON_EXECUTION_AMENDMENT.md"
VIEWS=("carrier1024_C01","carrier128_common_C01","carrier128_equivalent_C08",
       "carrier_sourcePCA128_C01","wdcnn_original128_C01","wdcnn_widthMatched128_C01",
       "fusion1087_C01","fusion191_equivalent_C01")
CONTRASTS=(
    ("unique_carrier_minus_widthmatched",VIEWS[1],VIEWS[5]),
    ("expanded_minus_original_wdcnn",VIEWS[5],VIEWS[4]),
    ("full_fusion_minus_widthmatched",VIEWS[6],VIEWS[5]),
    ("full_fusion_minus_original_carrier",VIEWS[6],VIEWS[0]),
    ("pca_minus_unique_carrier",VIEWS[3],VIEWS[1]),
)


def read_csv(path):
    with path.open(encoding="utf-8-sig",newline="") as stream:
        return list(csv.DictReader(stream))


def validate_query(roles,ids,truth,manifest):
    selected={r["archive_path"]:r for r in manifest if r["role"]=="target_query"}
    if len(ids)!=27 or len(set(ids.tolist()))!=27 or set(roles.tolist())!={"target_query"}:
        raise RuntimeError("Query roles or unique run count changed")
    if set(ids.tolist())!=set(selected): raise RuntimeError("Query identity manifest mismatch")
    mapping={"N":0,"IR":1,"OR":2}
    if Counter(truth.tolist())!={0:13,1:7,2:7}: raise RuntimeError("Query class counts changed")
    if any(int(label)!=mapping[selected[str(run)]["localisation_class"]] for run,label in zip(ids,truth)):
        raise RuntimeError("Query truth/manifest mismatch")


def pca_transform_blocks(blocks,components,mean):
    # Same operation order as sklearn 1.8 PCA._transform with whiten=False.
    values=blocks.reshape(-1,1024) @ components.T
    values -= mean.reshape(1,-1) @ components.T
    return values.reshape(len(blocks),8,128).mean(1)


def all_references():
    paths=[BASE/"run/target_evaluation.lock",BASE/"run/result.json",BASE/"run/predictions.csv",
           T2/"target_evaluation.lock",T2/"result.json",T2/"predictions.csv"]
    return {str(p.relative_to(ROOT)):sha256(p) for p in paths}


def vector_metrics(pred,truth):
    # Each row is one sampled encoder on the same sampled run set.
    acc=np.mean(pred==truth[None],axis=1)
    f1=[]
    for c in (0,1,2):
        tp=np.sum((pred==c)&(truth[None]==c),axis=1)
        den=np.sum(pred==c,axis=1)+np.sum(truth==c)
        f1.append(np.divide(2*tp,den,out=np.zeros_like(tp,dtype=float),where=den!=0))
    return float(acc.mean()),float(np.mean(f1))


def run():
    registry_path=PREP/"preparation_complete.json"
    registry=json.loads(registry_path.read_text())
    if registry["status"]!="PREPARED_QUERY_EXECUTION_DISABLED": raise RuntimeError("Unexpected preparation status")
    if len(registry["heads"])!=40: raise RuntimeError("Incomplete frozen head grid")
    for line in (PREP/"MANIFEST.sha256").read_text().splitlines():
        expected,rel=line.split(" *",1)
        if sha256(PREP/rel)!=expected: raise RuntimeError(f"Preparation hash changed: {rel}")
    if not AMENDMENT.is_file(): raise RuntimeError("Separate event activation is absent")
    if sha256(ROOT/"MCC5_POSTHOC_FAIR_COMPARISON_PROTOCOL.md")!=registry["protocol_sha256"]:
        raise RuntimeError("Original prepared protocol changed")
    for path,value in registry["model_hashes"].items():
        if sha256(ROOT/path)!=value: raise RuntimeError("Frozen model hash changed")
    head_rows={(int(r["seed"]),r["view"]):r for r in registry["heads"]}
    if set(head_rows)!={(s,v) for s in SEEDS for v in VIEWS}: raise RuntimeError("Head identities changed")
    references=all_references()
    OUT.mkdir(parents=True,exist_ok=True)
    lock_path=OUT/"target_evaluation.lock"
    event={"started_utc":datetime.now(timezone.utc).isoformat(),"event":"POST_TARGET_SUPPLEMENTARY_FAIRNESS",
           "prepared_registry_sha256":sha256(registry_path),"amendment_sha256":sha256(AMENDMENT),
           "runner_sha256":sha256(Path(__file__)),"seeds":list(SEEDS),"views":list(VIEWS),
           "original_artifact_hashes":references,"model_selection_allowed":False}
    # Exclusive creation prevents both a second event and concurrent execution.
    with lock_path.open("x",encoding="utf-8") as stream: json.dump(event,stream,indent=2)
    query_path=BASE/"cache/query_after_selection.npz"
    query_hash=sha256(query_path)
    with np.load(query_path,allow_pickle=False) as cache:
        if not bool(cache["query_cache"]): raise RuntimeError("Expected original query cache")
        ids,roles,y=cache["ids"],cache["roles"],cache["labels"]
        validate_query(roles,ids,y,read_csv(BASE/"protocol/protocol_run_manifest.csv"))
        waves,rpms,physical=cache["distributed"],cache["rpm_distributed"],cache["physical"]
        contexts,severities,torques=cache["contexts"],cache["severities"],cache["torques"]
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); torch.set_num_threads(4)
    predictions,metrics,token_rows,equivalence=[],[],[],[]
    grid={}; probabilities={}
    for seed in SEEDS:
        u2=load_model(BASE/f"run/checkpoints/u2hz_seed{seed}.pt",device)
        blocks,_=infer_blocks(u2,waves,rpms,device,u2=True); del u2
        carrier=blocks.mean(1); unique=carrier[:,:128]
        difference=float(np.max(np.abs(blocks.reshape(27,8,8,128)-blocks.reshape(27,8,8,128)[:,:,:1])))
        token_rows.append({"seed":seed,"max_token_difference":difference})
        model=make_wdcnn("wdcnn").to(device)
        model.load_state_dict(torch.load(T2/f"checkpoints/wdcnn_seed{seed}.pt",map_location="cpu",weights_only=False)["state"])
        original,_=infer_blocks(model,waves,rpms,device); del model
        payload=torch.load(P0/f"checkpoints/wdcnn_widthMatched_seed{seed}.pt",map_location="cpu",weights_only=False)
        model=WidthMatchedWDCNN(payload["width"]).to(device); model.load_state_dict(payload["state"])
        expanded,_=infer_blocks(model,waves,rpms,device); del model
        for view in VIEWS:
            row=head_rows[(seed,view)]; packet=joblib.load(PREP/row["file"])
            if packet["query_execution_enabled"] is not False: raise RuntimeError("Prepared state was overwritten")
            kind=packet["preprocess"]
            if kind=="carrier1024": x=carrier
            elif kind=="first_carrier_token": x=unique
            elif kind=="source_block_pca128": x=pca_transform_blocks(blocks,packet["pca_components"],packet["pca_mean"])
            elif kind=="wdcnn_original": x=original.mean(1)
            elif kind=="wdcnn_width_matched": x=expanded.mean(1)
            elif kind=="full_fusion": x=np.hstack([carrier,physical])
            elif kind=="compact_fusion": x=np.hstack([unique,physical])
            else: raise RuntimeError(kind)
            z=packet["scaler"].transform(x)
            scale=packet["carrier_scale_after_standardization"]
            if scale!=1.: z[:,:128]*=scale
            prob=packet["head"].predict_proba(z); pred=packet["head"].classes_[prob.argmax(1)]
            grid[(seed,view)]=pred; probabilities[(seed,view)]=prob
            metrics.append({"seed":seed,"view":view,"n_query_runs":27,**metric(y,pred)})
            for i,label in enumerate(pred):
                predictions.append({"seed":seed,"view":view,"run_id":str(ids[i]),"context":str(contexts[i]),
                                    "severity":str(severities[i]),"torque":int(torques[i]),"true":int(y[i]),
                                    "prediction":int(label),"correct":int(label==y[i])})
        for a,b in [(VIEWS[0],VIEWS[2]),(VIEWS[6],VIEWS[7])]:
            equivalence.append({"seed":seed,"original":a,"equivalent":b,"n_query_runs":27,
                                "class_disagreements":int(np.sum(grid[(seed,a)]!=grid[(seed,b)])),
                                "max_probability_difference":float(np.max(np.abs(probabilities[(seed,a)]-probabilities[(seed,b)])))})
        print(f"seed={seed}: all eight frozen views scored",flush=True)
    write_csv(OUT/"predictions.csv",predictions); write_csv(OUT/"seed_metrics.csv",metrics)
    write_csv(OUT/"token_checks.csv",token_rows); write_csv(OUT/"equivalence_checks.csv",equivalence)
    summary=[]
    for view in VIEWS:
        part=[r for r in metrics if r["view"]==view]
        row={"view":view,"seeds":5,"query_runs":27}
        for key in ("accuracy","macro_f1","recall_NRF","recall_IR","recall_OR"):
            values=[100*r[key] for r in part]
            row[key+"_mean_percent"]=float(np.mean(values)); row[key+"_sample_sd_pp"]=float(np.std(values,ddof=1))
        summary.append(row)
    write_csv(OUT/"summary.csv",summary)
    paired=[]; matrices={v:np.stack([grid[(s,v)] for s in SEEDS]) for v in VIEWS}
    for name,a,b in CONTRASTS:
        for seed in SEEDS:
            correct_a=grid[(seed,a)]==y; correct_b=grid[(seed,b)]==y
            ma=next(r for r in metrics if r["seed"]==seed and r["view"]==a)
            mb=next(r for r in metrics if r["seed"]==seed and r["view"]==b)
            paired.append({"contrast":name,"seed":seed,"method_a":a,"method_b":b,
                           "corrected_errors":int(np.sum(correct_a & ~correct_b)),
                           "introduced_errors":int(np.sum(~correct_a & correct_b)),
                           "accuracy_delta_pp":100*(ma["accuracy"]-mb["accuracy"]),
                           "macro_f1_delta_pp":100*(ma["macro_f1"]-mb["macro_f1"])})
    write_csv(OUT/"paired_errors.csv",paired)
    rng=np.random.default_rng(20261005); reps={name:[] for name,_,_ in CONTRASTS}
    groups=[np.flatnonzero(y==c) for c in (0,1,2)]
    for _ in range(10000):
        seeds=rng.integers(0,5,5)
        runs=np.concatenate([rng.choice(g,size=len(g),replace=True) for g in groups])
        for name,a,b in CONTRASTS:
            am=vector_metrics(matrices[a][np.ix_(seeds,runs)],y[runs])
            bm=vector_metrics(matrices[b][np.ix_(seeds,runs)],y[runs])
            reps[name].append([(am[0]-bm[0])*100,(am[1]-bm[1])*100])
    intervals=[]
    np.savez_compressed(OUT/"bootstrap_replicates.npz",**{name:np.asarray(values) for name,values in reps.items()})
    for name,a,b in CONTRASTS:
        values=np.asarray(reps[name]); part=[r for r in paired if r["contrast"]==name]
        intervals.append({"contrast":name,"method_a":a,"method_b":b,
                          "observed_accuracy_delta_pp":float(np.mean([r["accuracy_delta_pp"] for r in part])),
                          "observed_macro_f1_delta_pp":float(np.mean([r["macro_f1_delta_pp"] for r in part])),
                          "accuracy_percentile_low_pp":float(np.quantile(values[:,0],.025)),
                          "accuracy_percentile_high_pp":float(np.quantile(values[:,0],.975)),
                          "macro_f1_percentile_low_pp":float(np.quantile(values[:,1],.025)),
                          "macro_f1_percentile_high_pp":float(np.quantile(values[:,1],.975)),
                          "corrected_seed_run_errors":sum(r["corrected_errors"] for r in part),
                          "introduced_seed_run_errors":sum(r["introduced_errors"] for r in part)})
    write_csv(OUT/"contrast_summary.csv",intervals)
    replay=[]
    for path,column,label,view in [(BASE/"run/predictions.csv","variant","U2-Hz-Distributed-M8",VIEWS[0]),
                                   (BASE/"run/predictions.csv","variant","RAMEF-F",VIEWS[6]),
                                   (T2/"predictions.csv","method","WDCNN-DM8",VIEWS[4])]:
        ref={(int(r["seed"]),r["run_id"]):int(r["prediction"]) for r in read_csv(path) if r[column]==label}
        if len(ref)!=135: raise RuntimeError("Original reference prediction grid changed")
        differences=sum(int(grid[(s,view)][i])!=ref[(s,str(run))] for s in SEEDS for i,run in enumerate(ids))
        replay.append({"view":view,"archived_comparisons":135,"class_disagreements":differences})
    if references!=all_references(): raise RuntimeError("An original target artifact was modified")
    for line in (PREP/"MANIFEST.sha256").read_text().splitlines():
        expected,rel=line.split(" *",1)
        if sha256(PREP/rel)!=expected: raise RuntimeError("Prepared archive was modified")
    result={"status":"COMPLETE_POST_TARGET_SUPPLEMENTARY_EVENT","query_runs":27,"prediction_rows":len(predictions),
            "query_cache_sha256":query_hash,"summary":summary,"contrasts":intervals,"reference_replay":replay,
            "equivalence_checks":equivalence,"token_checks":token_rows,
            "prepared_heads_unchanged":True,"original_target_artifacts_unchanged":True,
            "model_fitting_performed":False,"cohort_status":"Previously evaluated fixed MCC5 runs; not a fresh external test"}
    (OUT/"result.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    event.update({"completed_utc":datetime.now(timezone.utc).isoformat(),"result_sha256":sha256(OUT/"result.json"),
                  "predictions_sha256":sha256(OUT/"predictions.csv"),"query_cache_sha256":query_hash})
    lock_path.write_text(json.dumps(event,indent=2),encoding="utf-8")
    with (OUT/"MANIFEST.sha256").open("w",encoding="utf-8") as stream:
        for p in sorted(p for p in OUT.rglob("*") if p.is_file() and p.name!="MANIFEST.sha256"):
            stream.write(f"{sha256(p)} *{p.relative_to(OUT).as_posix()}\n")
    print("COMPLETE_POST_TARGET_SUPPLEMENTARY_EVENT",flush=True)


if __name__=="__main__":
    with threadpool_limits(limits=4): run()
