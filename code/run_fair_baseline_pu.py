"""Train conventional source encoders and evaluate the frozen PU fair benchmark."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from bottleneck_deconvolution import load_contract, load_vibration, sha256
from fair_baseline_models import make_model
from prepare_splits import scan_records
from run_c2of import metrics
from run_lcga_fresh_p0 import digest_text
from run_lcga_paper_main import aggregate
from run_mc_sbt_p0 import SOURCE_DOMAINS, write_csv
from run_paper_main import PROTOCOL_MANIFEST_SHA256, WINDOW_LENGTH, load_contract as load_paper_contract


MODEL_NAMES = ("wdcnn", "resnet1d", "stft_cnn", "wdcnn_dann")


def centre_item(row):
    available = int(row.get("signal_length", 256001)) - WINDOW_LENGTH
    start = int(round(max(0, available) * .5))
    return {**row, "start": start, "end": start + WINDOW_LENGTH}


def waveform(row):
    signal = load_vibration(row["path"], row.get("signal_key", "signal")); start, end = int(row["start"]), int(row["end"])
    value = np.asarray(signal[start:end], dtype=np.float32).copy()
    if len(value) != WINDOW_LENGTH: raise RuntimeError("Invalid PU baseline crop")
    return (value - value.mean()) / max(float(value.std()), 1e-6)


def build_cache(records, query, supports, path):
    source = sorted([centre_item(x) for x in records if x["domain"] in SOURCE_DOMAINS], key=lambda x: x["measurement_id"])
    target_map = {x["measurement_id"]: x for x in query}
    for episode in supports:
        for row in episode: target_map.setdefault(row["measurement_id"], row)
    target = sorted(target_map.values(), key=lambda x: x["measurement_id"])
    ids = [x["measurement_id"] for x in source + target]
    if path.exists():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload["measurement_ids"] != ids: raise RuntimeError("Stale fair-baseline waveform cache")
        return source, target, payload["waveforms"].float()
    values = torch.from_numpy(np.stack([waveform(x) for x in source + target])).half()
    torch.save({"measurement_ids": ids, "waveforms": values}, path)
    return source, target, values.float()


def fixed_source_split(source):
    train, val = [], []
    groups = {}
    for i, row in enumerate(source): groups.setdefault((row["domain"], int(row["label"])), []).append(i)
    for key, indices in groups.items():
        indices = sorted(indices, key=lambda i: source[i]["measurement_id"])
        for position, index in enumerate(indices): (val if position % 5 == 0 else train).append(index)
    return np.asarray(sorted(train)), np.asarray(sorted(val))


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def infer(model, waves, device, batch=64):
    model.eval(); output = []
    for start in range(0, len(waves), batch): output.append(model(waves[start:start+batch].to(device))["embedding"].cpu())
    return torch.cat(output).numpy()


def train_model(name, waves, labels, domains, train_idx, val_idx, device, output):
    checkpoint = output / f"{name}.pt"
    if checkpoint.exists():
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False); model = make_model(name); model.load_state_dict(payload["state"]); return model.to(device), payload["best_val_macro_f1"], payload["history"]
    seed_all(20260824); model = make_model(name).to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    dataset = TensorDataset(waves[train_idx], torch.tensor(labels[train_idx]), torch.tensor(domains[train_idx]))
    loader = DataLoader(dataset, batch_size=32, shuffle=True, generator=torch.Generator().manual_seed(20260824))
    best, best_state, history = -1., None, []
    for epoch in range(12):
        model.train(); losses = []
        progress = epoch / 11 if 11 else 1.; grl = 2 / (1 + np.exp(-10 * progress)) - 1
        for x, y, d in loader:
            x, y, d = x.to(device), y.to(device), d.to(device); optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                result = model(x, float(grl)); loss = F.cross_entropy(result["class_logits"], y)
                if "domain_logits" in result: loss = loss + .2 * F.cross_entropy(result["domain_logits"], d)
            loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
        model.eval(); pred = []
        with torch.no_grad():
            for start in range(0, len(val_idx), 64):
                idx = val_idx[start:start+64]; pred.extend(model(waves[idx].to(device))["class_logits"].argmax(1).cpu().tolist())
        score = float(f1_score(labels[val_idx], pred, labels=[0,1,2], average="macro", zero_division=0))
        history.append({"model": name, "epoch": epoch, "train_loss": float(np.mean(losses)), "val_macro_f1": score})
        if score > best: best = score; best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best_state); torch.save({"state": best_state, "best_val_macro_f1": best, "history": history}, checkpoint)
    return model, best, history


def diagonal_logits(support, y, query):
    means=[]; variances=[]
    for c in range(3):
        value=support[y==c]; means.append(value.mean(0)); variances.append(value.var(0)+.05)
    mean=np.stack(means); var=np.stack(variances); return -(((query[:,None]-mean[None])**2/var[None]).mean(2))


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--output",type=Path,default=Path("outputs/staged/FAIR_BASELINE_PU")); parser.add_argument("--protocol",type=Path,default=Path("FAIR_BASELINE_PU_PROTOCOL.md")); args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/"evaluation.lock").exists(): raise RuntimeError("FAIR PU benchmark already locked")
    frozen=args.output/"protocol_frozen.md"
    if frozen.exists() and digest_text(frozen)!=digest_text(args.protocol): raise RuntimeError("Protocol mismatch")
    if not frozen.exists(): shutil.copy2(args.protocol,frozen)
    paper=Path("paper/evidence/protocol/PAPER_MAIN_V1")
    if sha256(paper/"MANIFEST.sha256")!=PROTOCOL_MANIFEST_SHA256: raise RuntimeError("PAPER_MAIN mismatch")
    query,supports,_=load_paper_contract(paper); records=scan_records(Path("D:/Paderborn"))
    source,target,waves=build_cache(records,query,supports,args.output/"waveforms.pt")
    nsource=len(source); labels=np.asarray([int(x["label"]) for x in source]); domain_map={x:i for i,x in enumerate(SOURCE_DOMAINS)}; domains=np.asarray([domain_map[x["domain"]] for x in source])
    train_idx,val_idx=fixed_source_split(source); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_index={x["measurement_id"]:nsource+i for i,x in enumerate(target)}; source_codes=[x["bearing_code"] for x in source]
    episode_rows=[]; predictions=[]; histories=[]; model_meta={}
    for name in MODEL_NAMES:
        model,best,history=train_model(name,waves[:nsource],labels,domains,train_idx,val_idx,device,args.output); histories.extend(history)
        embedding=infer(model,waves,device); model_meta[name]={"best_source_val_macro_f1":best,"parameters":sum(p.numel() for p in model.parameters())}
        source_x=embedding[:nsource]
        for episode,support in enumerate(supports):
            ti=np.asarray([target_index[x["measurement_id"]] for x in support]); qi=np.asarray([target_index[x["measurement_id"]] for x in query]); ty=np.asarray([int(x["label"]) for x in support]); qy=np.asarray([int(x["label"]) for x in query]); qcodes=[x["bearing_code"] for x in query]
            heads={"target_diagonal_proto":diagonal_logits(embedding[ti],ty,embedding[qi])}
            train_x=np.vstack([source_x,embedding[ti]]); train_y=np.concatenate([labels,ty]); scaler=StandardScaler().fit(train_x)
            linear=LogisticRegression(C=.1,max_iter=4000,solver="lbfgs",random_state=20260824+episode)
            linear.fit(scaler.transform(train_x),train_y,sample_weight=np.concatenate([np.ones(nsource),np.full(len(ti),10.)])); heads["source_support_linear"]=linear.decision_function(scaler.transform(embedding[qi]))
            for head,logits in heads.items():
                pred=logits.argmax(1); value=metrics(qy,pred,qcodes); value["recall_N"],value["recall_IR"],value["recall_OR"]=value["recall"]
                episode_rows.append({"model":name,"head":head,"episode":episode,"value":value})
                for item,y,p in zip(query,qy,pred): predictions.append({"model":name,"head":head,"episode":episode,"measurement_id":item["measurement_id"],"bearing_code":item["bearing_code"],"true":int(y),"prediction":int(p),"correct":int(y==p)})
        del model; torch.cuda.empty_cache()
    summaries={}
    for name in MODEL_NAMES:
        summaries[name]={}
        for head in ("target_diagonal_proto","source_support_linear"):
            rows=[x for x in episode_rows if x["model"]==name and x["head"]==head]; summaries[name][head]=aggregate(rows,"value")
    locked={"M2":json.loads(Path("outputs/staged/PAPER_MAIN_V1_B2/summary.json").read_text())["aggregate"],
            "LCGA_S":json.loads(Path("outputs/staged/LCGA_PAPER_MAIN_V1/summary.json").read_text())["candidate"],
            "SMRC_LCGA":json.loads(Path("outputs/staged/SMRC_LCGA/target_result.json").read_text())["candidate"],
            "PU_UNGATED_IO":json.loads(Path("outputs/staged/PU_UNGATED_IO/result.json").read_text())["ungated"]}
    result={"experiment":"FAIR_BASELINE_PU","epistemic_status":"retrospective fair benchmark","query_used_for_selection":False,"source_split":{"train":len(train_idx),"validation":len(val_idx)},"models":model_meta,"summaries":summaries,"locked_method_comparators":locked}
    write_csv(args.output/"training_history.csv",histories); write_csv(args.output/"predictions.csv",predictions)
    path=args.output/"result.json"; path.write_text(json.dumps(result,indent=2),encoding="utf-8"); (args.output/"evaluation.lock").write_text(json.dumps({"completed":True,"result_sha256":sha256(path)},indent=2),encoding="utf-8")
    print(json.dumps({"source_split":result["source_split"],"models":model_meta,"summaries":summaries},indent=2))


if __name__=="__main__": main()
