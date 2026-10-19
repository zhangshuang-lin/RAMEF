"""Generate auditable record-first PU LOCO splits."""
from __future__ import annotations
import argparse, csv, json, random, re
from collections import defaultdict
from pathlib import Path
from data_loader_pu import CONDITIONS, NAME_TO_DOMAIN, infer_default_label, load_vibration


FILE_RE = re.compile(r"(?P<condition>N\d+_M\d+_F\d+)_(?P<bearing>[^_]+)_(?P<record>\d+)\.mat$", re.I)


def read_labels(root: Path):
    path = root / "labels.csv"
    if not path.exists(): return {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        return {row["bearing_code"].strip(): int(row["fault_label"]) for row in csv.DictReader(f)}


def scan_records(root: Path):
    explicit = read_labels(root); records = []
    for path in root.rglob("*.mat"):
        m = FILE_RE.match(path.name)
        if not m or m.group("condition") not in NAME_TO_DOMAIN: continue
        code, domain = m.group("bearing"), NAME_TO_DOMAIN[m.group("condition")]
        label = explicit.get(code, infer_default_label(code))
        if label is None: continue
        records.append({"path": str(path.resolve()), "bearing_code": code,
                        "measurement_id": path.stem, "record": int(m.group("record")),
                        "condition": m.group("condition"), "domain": domain,
                        "domain_index": list(CONDITIONS).index(domain), "label": label,
                        "rpm": CONDITIONS[domain]["rpm"],
                        # Official PU recordings contain 256001 vibration points.
                        # Recording this here avoids repeatedly loading ~8 MB MAT
                        # files merely to discover their length during split design.
                        "signal_length": 256001})
    return records


def candidate_windows(record, length, stride):
    n = record.get("signal_length") or len(load_vibration(record["path"]))
    return [(s, s + length) for s in range(0, n - length + 1, stride)]


def build_split(records, target, shot, seed, length=8192, train_per=100,
                val_per=20, query_per=80, gap=16384):
    rng = random.Random(seed); grouped = defaultdict(list)
    for r in records: grouped[(r["domain"], r["label"])].append(r)
    windows = []
    labels = sorted({r["label"] for r in records})
    for domain in CONDITIONS:
        for label in labels:
            recs = grouped[(domain, label)][:]
            rng.shuffle(recs)
            if not recs: continue
            if domain != target:
                # Independent MAT files are the primary leakage boundary.
                val_count = max(1, min(4, len(recs) // 5))
                val_recs, train_recs = recs[:val_count], recs[val_count:]
                if not train_recs: train_recs, val_recs = recs, recs[:1]
                pools = [("train", train_recs, length // 2, train_per),
                         ("val", val_recs, length, val_per)]
                for role, chosen, stride, limit in pools:
                    candidates = [(r, s, e) for r in chosen for s, e in candidate_windows(r, length, stride)]
                    rng.shuffle(candidates)
                    for r, s, e in candidates[:limit]: windows.append({**r, "role": role, "start": s, "end": e})
            else:
                # Prefer disjoint measurement files for support and query.
                n_support_records = min(max(shot, 1), max(1, len(recs) // 2))
                support_recs, query_recs = recs[:n_support_records], recs[n_support_records:]
                if not query_recs:
                    # Fallback within records uses separated head/tail regions.
                    support_recs = query_recs = recs
                support_candidates, query_candidates = [], []
                same_records = set(x["path"] for x in support_recs) & set(x["path"] for x in query_recs)
                for r in support_recs:
                    cand = candidate_windows(r, length, length)
                    cutoff = r.get("signal_length", 256001) // 3 if r["path"] in same_records else 10**18
                    support_candidates += [(r, s, e) for s, e in cand if e <= cutoff]
                for r in query_recs:
                    cand = candidate_windows(r, length, length)
                    cutoff = 2 * r.get("signal_length", 256001) // 3 if r["path"] in same_records else 0
                    query_candidates += [(r, s, e) for s, e in cand if s >= cutoff + (gap if cutoff else 0)]
                rng.shuffle(support_candidates); rng.shuffle(query_candidates)
                for r, s, e in support_candidates[:shot]: windows.append({**r, "role": "support", "start": s, "end": e})
                for r, s, e in query_candidates[:query_per]: windows.append({**r, "role": "query", "start": s, "end": e})
    return {"version": 1, "target_domain": target, "shot": shot, "seed": seed,
            "window_length": length, "classes": labels, "windows": windows}


def validate_split(split):
    by_path_role = defaultdict(list)
    for w in split["windows"]: by_path_role[(w["path"], w["role"])].append((w["start"], w["end"]))
    forbidden = [("train", "val"), ("support", "query")]
    for path in {k[0] for k in by_path_role}:
        for a, b in forbidden:
            for x in by_path_role.get((path, a), []):
                for y in by_path_role.get((path, b), []):
                    if max(x[0], y[0]) < min(x[1], y[1]): raise AssertionError(f"Leakage in {path}: {a}/{b}")


def validate_independent_measurement_split(split):
    """Stricter contract used by the locked DS 5-shot protocol.

    A measurement may belong to only one role among train/val and support/query.
    In particular, every target support shot is an independent MAT measurement
    and no window from that measurement is allowed to enter the query pool.
    """
    validate_split(split)
    role_by_measurement = defaultdict(set)
    for w in split["windows"]:
        role_by_measurement[w["measurement_id"]].add(w["role"])
    forbidden = ({"train", "val"}, {"support", "query"})
    for measurement, roles in role_by_measurement.items():
        for pair in forbidden:
            if pair <= roles:
                raise AssertionError(f"Measurement leakage in {measurement}: {sorted(roles)}")

    target = split["target_domain"]
    labels = split["classes"]
    for label in labels:
        support = [w for w in split["windows"]
                   if w["domain"] == target and w["role"] == "support" and w["label"] == label]
        query = [w for w in split["windows"]
                 if w["domain"] == target and w["role"] == "query" and w["label"] == label]
        if len(support) != split["shot"]:
            raise AssertionError(f"Class {label}: expected {split['shot']} support, got {len(support)}")
        if len({w["measurement_id"] for w in support}) != split["shot"]:
            raise AssertionError(f"Class {label}: support shots are not independent measurements")
        if len(query) != split.get("query_per_class", 80):
            raise AssertionError(f"Class {label}: wrong query count {len(query)}")


def main():
    p=argparse.ArgumentParser(); p.add_argument("--data-root", default=r"D:\Paderborn")
    p.add_argument("--output", default="splits"); p.add_argument("--shots", nargs="+", type=int, default=[1,5,10])
    p.add_argument("--seeds", type=int, default=10); p.add_argument("--window-length", type=int, default=8192)
    args=p.parse_args(); root=Path(args.data_root); out=Path(args.output); out.mkdir(parents=True, exist_ok=True)
    records=scan_records(root)
    labels=sorted({r["label"] for r in records})
    if labels != [0, 1, 2]:
        raise ValueError(f"Locked main protocol requires exactly N/I/O labels [0,1,2], found {labels}")
    for target in CONDITIONS:
        for shot in args.shots:
            for seed in range(args.seeds):
                split=build_split(records,target,shot,seed,args.window_length); validate_split(split)
                (out/f"{target}_{shot}shot_seed{seed:02d}.json").write_text(json.dumps(split,indent=2),encoding="utf-8")
    print(f"Generated splits from {len(records)} measurement files in {out.resolve()}")
if __name__ == "__main__": main()
