"""Create deterministic, non-overlapping CWRU eight-window bag episodes."""
from __future__ import annotations
import csv,hashlib,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from bottleneck_deconvolution import sha256

POOL=Path('cwru_10class_window_pool.csv');OUTPUT=Path('cwru_ramef_bag_splits.json')

def local_seed(target,episode,record):return int.from_bytes(hashlib.sha256(f'{target}:{episode}:{record}'.encode()).digest()[:8],'little')

def main():
 if OUTPUT.exists():raise RuntimeError('CWRU RAMEF bag split already exists')
 with POOL.open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
 by_record=defaultdict(list)
 for row in rows:by_record[row['record_id']].append(row)
 episodes=[]
 for target in range(4):
  for episode in range(10):
   bags=[]
   for record,items in sorted(by_record.items()):
    items=sorted(items,key=lambda x:int(x['raw_window_index']));rng=np.random.default_rng(local_seed(target,episode,record));chosen=[items[i] for i in rng.permutation(len(items))];load=int(items[0]['load_hp']);count=64 if load==target else 40
    if len(chosen)<count:raise RuntimeError(f'{record} has only {len(chosen)} windows')
    chosen=chosen[:count]
    for bag_index in range(count//8):
     role=('source' if load!=target else ('support' if bag_index<5 else 'query'));block=chosen[bag_index*8:(bag_index+1)*8];bags.append({'bag_id':f'T{target}_S{episode}_{record}_{role}_{bag_index:02d}','record_id':record,'label':int(items[0]['label']),'class_name':items[0]['class_name'],'load_hp':load,'role':role,'window_ids':[x['window_id'] for x in block]})
   support={w for b in bags if b['role']=='support' for w in b['window_ids']};query={w for b in bags if b['role']=='query' for w in b['window_ids']}
   counts={role:sum(b['role']==role for b in bags) for role in ('source','support','query')}
   if counts!={'source':150,'support':50,'query':30} or support&query:raise RuntimeError('bag protocol violation')
   episodes.append({'episode_id':f'T{target}_S{episode}','target_load':target,'seed':episode,'bags':bags})
 payload={'protocol':'CWRU_RAMEF_BAG_V1','pool_sha256':sha256(POOL),'bag_size':8,'source_bags_per_record':5,'target_support_bags_per_class':5,'target_query_bags_per_class':3,'episodes':episodes};OUTPUT.write_text(json.dumps(payload,indent=2),encoding='utf-8');print(json.dumps({'output':str(OUTPUT),'sha256':sha256(OUTPUT),'episodes':len(episodes)},indent=2))
if __name__=='__main__':main()
