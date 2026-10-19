"""Query-free check of duplicated U2-Hz tokens and equivalent linear heads."""
import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from mcc5_confirmatory_data import sha256
from run_mcc5_capacity_dimension_p0 import metric, validate_development_roles, write_csv
from run_mcc5_ramef_confirmatory import SEEDS, balanced_source_weights

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "outputs/staged/MCC5_RAMEF_CONFIRMATORY"
P0 = ROOT / "outputs/staged/MCC5_CAPACITY_DIMENSION_P0"
OUT = ROOT / "outputs/staged/MCC5_CARRIER_DIMENSION_EQUIVALENCE_P0B"
PROTOCOL = ROOT / "MCC5_DIMENSION_EQUIVALENCE_P0B_PROTOCOL.md"


def probabilities(x, y, q, weights, c, seed, carrier_scale=1.):
    scaler = StandardScaler().fit(x)
    train, test = scaler.transform(x), scaler.transform(q)
    if carrier_scale != 1.:
        train[:, :128] *= carrier_scale
        test[:, :128] *= carrier_scale
    head = LogisticRegression(C=c, solver="lbfgs", max_iter=4000, random_state=seed)
    head.fit(train, y, sample_weight=weights)
    return head.predict_proba(test)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/"audit.lock").exists():
        raise RuntimeError("Existing P0B archive is preserved")
    (OUT/"audit.lock").write_text(json.dumps({"started_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha256(PROTOCOL), "runner_sha256": sha256(Path(__file__)),
        "query_evaluation_authorized": False}, indent=2), encoding="utf-8")
    shutil.copyfile(PROTOCOL, OUT/"protocol_frozen.md")
    expected = {}
    for line in (P0/"MANIFEST.sha256").read_text().splitlines():
        value, path = line.split(" *",1); expected[path] = value
    cache_path = BASE/"cache/development.npz"
    selection = json.loads((BASE/"run/selection_complete.json").read_text())
    assert sha256(cache_path) == selection["development_cache_sha256"]
    with np.load(cache_path, allow_pickle=False) as cache:
        validate_development_roles(cache["roles"], cache["ids"])
        ids, roles = cache["ids"], cache["roles"]
        physical = cache["physical"]
    source = np.flatnonzero(roles == "source")
    val = np.flatnonzero(roles == "source_validation")
    support = np.flatnonzero(roles == "target_support")
    rows, pairs, token_checks = [], [], []
    for seed in SEEDS:
        path = P0/f"development_embeddings_seed{seed}.npz"
        assert sha256(path) == expected[path.name]
        with np.load(path, allow_pickle=False) as archive:
            validate_development_roles(archive["roles"], archive["ids"])
            assert np.array_equal(archive["ids"], ids)
            x, y = archive["U2_Hz_1024"], archive["labels"]
        tokens = x.reshape(99, 8, 128)
        delta = float(np.max(np.abs(tokens-tokens[:, :1])))
        token_checks.append({"seed": seed, "max_absolute_token_difference": delta,
                             "nominal_dimensions": 1024, "unique_token_dimensions": 128})
        first = tokens[:, 0]
        views = {
            "Carrier1024_C0.1": (x, .1, 1.),
            "Carrier128_C0.1": (first, .1, 1.),
            "Carrier128_C0.8_equivalent": (first, .8, 1.),
            "Fusion1087_C0.1": (np.hstack([x, physical]), .1, 1.),
            "Fusion191_scaled_equivalent": (np.hstack([first, physical]), .1, np.sqrt(8.)),
        }
        weights = balanced_source_weights(y[source])
        results = {}
        for name, (features, c, scale) in views.items():
            for condition in ("source_validation_source_only", "source_validation_source_plus_support", "held_target_support_loo"):
                if condition == "held_target_support_loo":
                    index = support; probs = []
                    for held in range(len(support)):
                        keep = np.delete(support, held)
                        train = np.concatenate([source, keep])
                        w = np.concatenate([weights, np.full(len(keep),10.)])
                        probs.append(probabilities(features[train],y[train],features[support[[held]]],w,c,seed,scale)[0])
                    prob = np.asarray(probs)
                else:
                    index = val
                    train = source if condition.endswith("source_only") else np.concatenate([source,support])
                    w = weights if len(train) == len(source) else np.concatenate([weights,np.full(len(support),10.)])
                    prob = probabilities(features[train], y[train], features[val], w, c, seed, scale)
                results[(name,condition)] = prob
                rows.append({"seed":seed,"view":name,"condition":condition,"n_runs":len(index),**metric(y[index],prob.argmax(1))})
        for original, equivalent in [("Carrier1024_C0.1", "Carrier128_C0.8_equivalent"),
                                     ("Fusion1087_C0.1", "Fusion191_scaled_equivalent")]:
            for condition in ("source_validation_source_only","source_validation_source_plus_support","held_target_support_loo"):
                a,b = results[(original,condition)],results[(equivalent,condition)]
                pairs.append({"seed":seed,"original":original,"equivalent":equivalent,"condition":condition,
                              "n_predictions":len(a),"prediction_disagreements":int(np.sum(a.argmax(1)!=b.argmax(1))),
                              "max_probability_difference":float(np.max(np.abs(a-b)))})
    write_csv(OUT/"seed_metrics.csv",rows)
    write_csv(OUT/"equivalence_checks.csv",pairs)
    write_csv(OUT/"token_checks.csv",token_checks)
    summaries=[]
    for name,condition in sorted({(r["view"],r["condition"]) for r in rows}):
        part=[r for r in rows if (r["view"],r["condition"])==(name,condition)]
        row={"view":name,"condition":condition}
        for key in ("accuracy","macro_f1"):
            values=[r[key]*100 for r in part]
            row[key+"_mean_percent"]=float(np.mean(values))
            row[key+"_sample_sd_pp"]=float(np.std(values,ddof=1))
        summaries.append(row)
    write_csv(OUT/"summary.csv",summaries)
    result={"status":"COMPLETE_QUERY_FREE_EQUIVALENCE_AUDIT","token_checks":token_checks,
            "comparisons":pairs,"total_compared_predictions":sum(r["n_predictions"] for r in pairs),
            "total_prediction_disagreements":sum(r["prediction_disagreements"] for r in pairs),
            "max_probability_difference":max(r["max_probability_difference"] for r in pairs),
            "summaries":summaries,"target_query_loaded":False}
    (OUT/"result.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    with (OUT/"MANIFEST.sha256").open("w",encoding="utf-8") as stream:
        for p in sorted(p for p in OUT.rglob("*") if p.is_file() and p.name!="MANIFEST.sha256"):
            stream.write(f"{sha256(p)} *{p.relative_to(OUT).as_posix()}\n")
    print(json.dumps({k:result[k] for k in ("status","total_compared_predictions","total_prediction_disagreements","max_probability_difference")},indent=2))


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
