"""Frozen feature selectors for the RACP fair ablation."""
from __future__ import annotations
import hashlib
import numpy as np

BLOCK4=(0,2,5,7)

def static_indices()->np.ndarray:
    indices=[]
    # Three bands, four evidence columns, retain mean and median only.
    for band in range(3):
        base=band*16
        for column in range(4):indices.extend([base+column*4,base+column*4+1])
    # Cross-band best OR, best IR, and OR-minus-IR: mean and median.
    for base in (48,53,58):indices.extend([base,base+1])
    return np.asarray(indices,dtype=np.int64)

def deterministic_nuisance(measurement_ids:list[str],dimensions:int=63,
                           seed:int=20260821)->np.ndarray:
    rows=[]
    for measurement_id in measurement_ids:
        digest=hashlib.sha256(f'{seed}:{measurement_id}'.encode()).digest()
        local_seed=int.from_bytes(digest[:8],'little',signed=False)
        rows.append(np.random.default_rng(local_seed).standard_normal(dimensions))
    return np.asarray(rows,dtype=np.float32)
