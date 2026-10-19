"""T3–T5 frozen MCC5 error, uncertainty and context/physics analyses."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from mcc5_confirmatory_data import sha256
from run_mcc5_ramef_confirmatory import SEEDS, load_npz


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "outputs" / "staged" / "MCC5_RAMEF_CONFIRMATORY"
PACKAGE = ROOT / "outputs" / "staged" / "MCC5_TRANSFERRED_EVIDENCE_PACKAGE"
OUT3 = PACKAGE / "T3_run_errors"; OUT4 = PACKAGE / "T4_bootstrap"; OUT5 = PACKAGE / "T5_context_physics"
PROTOCOL = ROOT / "MCC5_TRANSFERRED_EVIDENCE_PACKAGE_PROTOCOL.md"
PRED = BASE / "run" / "predictions.csv"
QUERY = BASE / "cache" / "query_after_selection.npz"
DEVELOPMENT = BASE / "cache" / "development.npz"
VARIANTS = {"U2-Hz-Local-M8": "U2-Hz-LM8", "U2-Hz-Distributed-M8": "U2-Hz-DM8", "RAMEF-F": "RAMEF"}
LABELS = {0: "NRF", 1: "IR", 2: "OR"}


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream: return list(csv.DictReader(stream))


def write_csv(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not rows: Path(path).write_text("", encoding="utf-8"); return
    with Path(path).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def nuisance(run_id: str) -> str:
    value = run_id.lower()
    if "broken_bar" in value: return "broken_bar"
    if "dynamic_eccentricity" in value: return "dynamic_eccentricity"
    if "static_eccentricity" in value: return "static_eccentricity"
    if "winding" in value: return "winding"
    if "voltage_unbalance" in value: return "voltage_unbalance"
    if "bend" in value: return "bend"
    if "health" in value: return "health"
    return "pure_race"


def load_predictions():
    rows = []
    for row in read_csv(PRED):
        if row["variant"] not in VARIANTS: continue
        rows.append({"seed": int(row["seed"]), "method": VARIANTS[row["variant"]], "run_id": row["run_id"],
                     "context": row["context"], "severity": row["severity"], "torque": int(row["torque"]),
                     "true": int(row["true"]), "prediction": int(row["prediction"]), "correct": int(row["correct"]),
                     "nuisance": nuisance(row["run_id"])})
    return rows


def t3(rows):
    OUT3.mkdir(parents=True, exist_ok=True); write_csv(OUT3 / "seed_run_correctness.csv", rows)
    lookup = {(r["seed"], r["method"], r["run_id"]): r for r in rows}
    run_ids = sorted({r["run_id"] for r in rows}); summary = []
    for run_id in run_ids:
        base = next(r for r in rows if r["run_id"] == run_id)
        item = {k: base[k] for k in ("run_id", "context", "severity", "torque", "true", "nuisance")}
        item["class"] = LABELS[item.pop("true")]
        for method in ("U2-Hz-LM8", "U2-Hz-DM8", "RAMEF"):
            item[f"{method}_correct_seeds"] = sum(lookup[(s, method, run_id)]["correct"] for s in SEEDS)
        item["dm8_fixes_lm8"] = sum(not lookup[(s,"U2-Hz-LM8",run_id)]["correct"] and lookup[(s,"U2-Hz-DM8",run_id)]["correct"] for s in SEEDS)
        item["dm8_new_errors"] = sum(lookup[(s,"U2-Hz-LM8",run_id)]["correct"] and not lookup[(s,"U2-Hz-DM8",run_id)]["correct"] for s in SEEDS)
        item["ramef_fixes_dm8"] = sum(not lookup[(s,"U2-Hz-DM8",run_id)]["correct"] and lookup[(s,"RAMEF",run_id)]["correct"] for s in SEEDS)
        item["ramef_new_errors"] = sum(lookup[(s,"U2-Hz-DM8",run_id)]["correct"] and not lookup[(s,"RAMEF",run_id)]["correct"] for s in SEEDS)
        summary.append(item)
    write_csv(OUT3 / "run_correction_summary.csv", summary)
    totals = {key: int(sum(r[key] for r in summary)) for key in ("dm8_fixes_lm8","dm8_new_errors","ramef_fixes_dm8","ramef_new_errors")}
    result = {"experiment":"MCC5_T3_RUN_LEVEL_ERROR_AUDIT","status":"COMPLETE_POST_HOC_FROZEN",
              "runs":len(run_ids),"seeds":list(SEEDS),"totals":totals,"protocol_sha256":sha256(PROTOCOL),"prediction_sha256":sha256(PRED)}
    (OUT3 / "result.json").write_text(json.dumps(result,indent=2),encoding="utf-8"); return summary, lookup


def score(y, pred):
    return float(np.mean(y==pred)), float(f1_score(y,pred,labels=[0,1,2],average="macro",zero_division=0))


def t4(rows):
    OUT4.mkdir(parents=True, exist_ok=True)
    lookup = {(r["seed"],r["method"],r["run_id"]):r["prediction"] for r in rows}
    truth = {r["run_id"]:r["true"] for r in rows}; by_class=defaultdict(list)
    for run_id,y in truth.items(): by_class[y].append(run_id)
    rng=np.random.default_rng(20261220); reps=[]
    comparisons=(("U2-Hz-DM8","U2-Hz-LM8","DM8_minus_LM8"),("RAMEF","U2-Hz-DM8","RAMEF_minus_DM8"))
    for rep in range(10000):
        seeds=rng.choice(np.asarray(SEEDS),size=len(SEEDS),replace=True)
        sampled=[]
        for c in (0,1,2): sampled.extend(rng.choice(by_class[c],size=len(by_class[c]),replace=True).tolist())
        y=np.asarray([truth[r] for s in seeds for r in sampled])
        for left,right,name in comparisons:
            lp=np.asarray([lookup[(int(s),left,r)] for s in seeds for r in sampled])
            rp=np.asarray([lookup[(int(s),right,r)] for s in seeds for r in sampled])
            la,lf=score(y,lp);ra,rf=score(y,rp)
            reps.append({"replicate":rep,"comparison":name,"accuracy_delta":la-ra,"macro_f1_delta":lf-rf})
    write_csv(OUT4 / "bootstrap_replicates.csv",reps)
    summary={}
    for _,_,name in comparisons:
        local=[r for r in reps if r["comparison"]==name];summary[name]={}
        for key in ("accuracy_delta","macro_f1_delta"):
            v=np.asarray([r[key] for r in local]);summary[name][key]={"mean":float(v.mean()),"ci95":[float(np.quantile(v,.025)),float(np.quantile(v,.975))],"positive_probability":float(np.mean(v>0))}
    result={"experiment":"MCC5_T4_HIERARCHICAL_PAIRED_BOOTSTRAP","status":"COMPLETE_POST_HOC_FROZEN","replicates":10000,"seed":20261220,
            "resampling_unit":"five encoder seeds and query runs stratified within NRF/IR/OR","summary":summary,"protocol_sha256":sha256(PROTOCOL),"prediction_sha256":sha256(PRED)}
    (OUT4/"result.json").write_text(json.dumps(result,indent=2),encoding="utf-8");return result


def group_metrics(rows, field):
    output=[]
    groups=sorted({str(r[field]) for r in rows})
    for group in groups:
        run_ids=sorted({r["run_id"] for r in rows if str(r[field])==group})
        if len(run_ids)<2: continue
        for method in ("U2-Hz-DM8","RAMEF"):
            local=[r for r in rows if r["run_id"] in run_ids and r["method"]==method]
            output.append({"group_type":field,"group":group,"runs":len(run_ids),"method":method,
                           "accuracy":float(np.mean([r["correct"] for r in local])),"predictions":len(local)})
    return output


def physical_rows(cache, role):
    rows=[]
    for i,run_id in enumerate(cache["ids"]):
        p=cache["physical"][i]
        rows.append({"role":role,"run_id":str(run_id),"class":LABELS[int(cache["labels"][i])],"label":int(cache["labels"][i]),
                     "context":str(cache["contexts"][i]),"severity":str(cache["severities"][i]),"torque":int(cache["torques"][i]),"nuisance":nuisance(str(run_id)),
                     "bpfo_mean":float(p[48]),"bpfo_median":float(p[49]),"bpfi_mean":float(p[53]),"bpfi_median":float(p[54]),
                     "bpfo_minus_bpfi_mean":float(p[58]),"bpfo_occurrence":float(p[51]),"bpfi_occurrence":float(p[56])})
    return rows


def t5(pred_rows):
    OUT5.mkdir(parents=True,exist_ok=True);query=load_npz(QUERY);dev=load_npz(DEVELOPMENT)
    phys=physical_rows(query,"query")+physical_rows({k:v[dev["roles"]=="target_support"] if getattr(v,"ndim",0)>0 and len(v)==len(dev["roles"]) else v for k,v in dev.items()},"support")
    write_csv(OUT5/"physical_evidence_by_run.csv",phys)
    metrics=[]
    for field in ("true","context","severity","torque","nuisance"): metrics.extend(group_metrics(pred_rows,field))
    write_csv(OUT5/"group_performance.csv",metrics)
    group_phys=[]
    for role in ("support","query"):
        for field in ("class","context","severity","torque","nuisance"):
            groups=sorted({str(r[field]) for r in phys if r["role"]==role})
            for group in groups:
                local=[r for r in phys if r["role"]==role and str(r[field])==group]
                if len(local)<2:continue
                item={"role":role,"group_type":field,"group":group,"runs":len(local)}
                for key in ("bpfo_mean","bpfi_mean","bpfo_minus_bpfi_mean","bpfo_occurrence","bpfi_occurrence"):
                    v=np.asarray([r[key] for r in local]);item[key+"_mean"]=float(v.mean());item[key+"_std"]=float(v.std(ddof=1))
                group_phys.append(item)
    write_csv(OUT5/"group_physical_evidence.csv",group_phys)
    result={"experiment":"MCC5_T5_CONTEXT_SEVERITY_PHYSICS","status":"COMPLETE_POST_HOC_DESCRIPTIVE","minimum_group_runs":2,
            "query_runs":27,"support_runs":15,"protocol_sha256":sha256(PROTOCOL),"query_cache_sha256":sha256(QUERY)}
    (OUT5/"result.json").write_text(json.dumps(result,indent=2),encoding="utf-8");return result


def main():
    for p in (OUT3,OUT4,OUT5):p.mkdir(parents=True,exist_ok=True)
    rows=load_predictions();summary,lookup=t3(rows);boot=t4(rows);context=t5(rows)
    print(json.dumps({"T3":json.loads((OUT3/"result.json").read_text()),"T4":boot,"T5":context},indent=2))


if __name__=="__main__":main()
