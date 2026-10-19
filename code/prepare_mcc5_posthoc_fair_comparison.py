"""Freeze all development-fitted heads; target execution is disabled."""
import json
import platform
import shutil
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from mcc5_confirmatory_data import sha256
from run_mcc5_capacity_dimension_p0 import validate_development_roles
from run_mcc5_ramef_confirmatory import SEEDS, balanced_source_weights

ROOT=Path(__file__).resolve().parent
P0=ROOT/"outputs/staged/MCC5_CAPACITY_DIMENSION_P0"
BASE=ROOT/"outputs/staged/MCC5_RAMEF_CONFIRMATORY"
OUT=ROOT/"outputs/staged/MCC5_POSTHOC_FAIR_COMPARISON_PREPARED"
PROTOCOL=ROOT/"MCC5_POSTHOC_FAIR_COMPARISON_PROTOCOL.md"


def main():
    if (OUT/"preparation_complete.json").exists():
        raise RuntimeError("Prepared archive already exists")
    OUT.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(PROTOCOL,OUT/"protocol_frozen.md")
    selection=json.loads((BASE/"run/selection_complete.json").read_text())
    cache_path=BASE/"cache/development.npz"
    assert sha256(cache_path)==selection["development_cache_sha256"]
    with np.load(cache_path,allow_pickle=False) as cache:
        ids,roles,y=cache["ids"],cache["roles"],cache["labels"]
        validate_development_roles(roles,ids)
        physical=cache["physical"]
    source=np.flatnonzero(roles=="source"); support=np.flatnonzero(roles=="target_support")
    fit=np.concatenate([source,support])
    weights=np.concatenate([balanced_source_weights(y[source]),np.full(len(support),10.)])
    records=[]
    for seed in SEEDS:
        feature_path=P0/f"development_embeddings_seed{seed}.npz"
        with np.load(feature_path,allow_pickle=False) as features:
            assert np.array_equal(features["ids"],ids)
            validate_development_roles(features["roles"],features["ids"])
            carrier=features["U2_Hz_1024"]
            unique=carrier[:,:128]
            assert np.max(np.abs(carrier.reshape(99,8,128)-unique[:,None]))==0
            views={
                "carrier1024_C01":(carrier,.1,1.,"carrier1024"),
                "carrier128_common_C01":(unique,.1,1.,"first_carrier_token"),
                "carrier128_equivalent_C08":(unique,.8,1.,"first_carrier_token"),
                "carrier_sourcePCA128_C01":(features["U2_Hz_sourcePCA128"],.1,1.,"source_block_pca128"),
                "wdcnn_original128_C01":(features["WDCNN_128"],.1,1.,"wdcnn_original"),
                "wdcnn_widthMatched128_C01":(features["WDCNN_widthMatched_128"],.1,1.,"wdcnn_width_matched"),
                "fusion1087_C01":(np.hstack([carrier,physical]),.1,1.,"full_fusion"),
                "fusion191_equivalent_C01":(np.hstack([unique,physical]),.1,np.sqrt(8.),"compact_fusion"),
            }
            for name,(x,c,scale,kind) in views.items():
                scaler=StandardScaler().fit(x[fit])
                z=scaler.transform(x[fit])
                if scale!=1.: z[:,:128]*=scale
                head=LogisticRegression(C=c,solver="lbfgs",max_iter=4000,random_state=seed)
                head.fit(z,y[fit],sample_weight=weights)
                packet={"seed":seed,"view":name,"preprocess":kind,"scaler":scaler,"head":head,
                        "carrier_scale_after_standardization":scale,
                        "pca_components":features["pca_components"],"pca_mean":features["pca_mean"],
                        "query_execution_enabled":False}
                path=OUT/f"head_{seed}_{name}.joblib"
                joblib.dump(packet,path)
                records.append({"seed":seed,"view":name,"dimension":x.shape[1],"C":c,
                                "file":path.name,"sha256":sha256(path),
                                "development_feature_sha256":sha256(feature_path)})
    original_registry=json.loads((P0/"source_audit.lock").read_text())
    model_hashes=original_registry["original_checkpoints"].copy()
    for seed in SEEDS:
        path=P0/f"checkpoints/wdcnn_widthMatched_seed{seed}.pt"
        model_hashes[str(path.relative_to(ROOT))]=sha256(path)
    result={"status":"PREPARED_QUERY_EXECUTION_DISABLED","protocol_sha256":sha256(PROTOCOL),
            "preparer_sha256":sha256(Path(__file__)),"heads":records,"model_hashes":model_hashes,
            "development_roles":{"source":42,"source_validation":42,"target_support":15},
            "query_read":False,"target_evaluation_performed":False,
            "software":{"python":platform.python_version(),"numpy":np.__version__,"scikit_learn":sklearn.__version__}}
    assert len(records)==40
    (OUT/"preparation_complete.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    with (OUT/"MANIFEST.sha256").open("w",encoding="utf-8") as stream:
        for path in sorted(p for p in OUT.rglob("*") if p.is_file() and p.name!="MANIFEST.sha256"):
            stream.write(f"{sha256(path)} *{path.relative_to(OUT).as_posix()}\n")
    print("PREPARED_QUERY_EXECUTION_DISABLED: 40 source/support-fitted heads frozen")


if __name__=="__main__":
    with threadpool_limits(limits=4): main()
