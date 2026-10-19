"""Three editorial refinements only, retaining the previous vector release."""
from pathlib import Path
import hashlib
import importlib.util
import json

import numpy as np
import pandas as pd
from matplotlib.collections import PathCollection

ROOT=Path(__file__).resolve().parents[3]
spec=importlib.util.spec_from_file_location('base_repair',Path(__file__).with_name('repair_figures_6_8_11_20261014.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.OUT=ROOT/'paper/figures/revised_20261014_v2'
m.QA=ROOT/'tmp/figure_repair_20261014_v2'
m.OUT.mkdir(parents=True,exist_ok=True);m.QA.mkdir(parents=True,exist_ok=True)
base_finish=m.finish
CHECKS={}

def refined_finish(fig,name):
    if name.startswith('Fig6_'):
        ax=fig.axes[0]
        for col in ax.collections:
            if isinstance(col,PathCollection):
                col.set_offsets(col.get_offsets()[:,::-1])
        locations={'KA16':(1.11,.97),'KA22':(1.04,.80),'KA30':(1.04,.53)}
        for a in ax.texts:
            if a.get_gid() and a.get_gid().startswith('code-label-'):
                a.xy=a.xy[::-1]
                a.set_position(locations[a.get_text()])
        ax.set_xlabel('Best-band BPFO evidence')
        ax.set_ylabel('Best-band BPFI evidence')
        values=json.loads(m.PHYS.read_text(encoding='utf-8'))
        for cls,col in zip((0,1,2),ax.collections):
            expected=[[p['bpfo'],p['bpfi']] for p in values if p['class']==cls]
            assert np.array_equal(col.get_offsets(),expected)
        m.RECORD['checks']['figure6'].pop('all_scatter_coordinates_unchanged',None)
        m.RECORD['checks']['figure6']['evidence_values_unchanged']=True
        m.RECORD['checks']['figure6']['axis_order']={'x':'BPFO','y':'BPFI'}
        CHECKS['figure6_axis_and_label_coordinates_verified']=True
        name='Fig6_PU_physical_evidence_V2'
    elif name.startswith('Fig8_'):
        ax=fig.axes[1]
        ax.set_ylim(0,118)
        ax.set_yticks([0,20,40,60,80,100])
        assert all(p.get_y()==0 for p in ax.patches)
        CHECKS['figure8b_zero_baseline']=True
        name='Fig8_MCC5_evidence_V2'
    elif name.startswith('Fig11_'):
        fig.axes[0].set_title('Support LOO mean ± sample SD',pad=33)
        b=fig.axes[1]
        b.set_title('Leave-two-out seed summaries',pad=33)
        b.text(.5,-.25,'105 folds; 210 predictions per seed',transform=b.transAxes,
               ha='center',va='top',fontsize=8.5)
        ax=fig.axes[3]
        group=pd.read_csv(m.PACKAGE/'T5_context_physics/group_performance.csv')
        g=group[(group.group_type=='nuisance')&(group.runs>=2)]
        groups=sorted(g.group.unique())
        counts=[]
        for i,group_name in enumerate(groups):
            ns=g[g.group==group_name].runs.unique();assert len(ns)==1
            n=int(ns[0]);counts.append(n)
            ax.text(i,106,f'n={n}',ha='center',va='center',fontsize=8.5)
            assert (g[g.group==group_name].predictions==5*n).all()
        ax.set_ylim(0,115)
        ax.set_yticks([0,20,40,60,80,100])
        assert counts==[2,4,3,6,5,2,4]
        m.RECORD['checks']['figure11']['group_run_counts']=counts
        m.RECORD['checks']['figure11']['predictions_per_method_per_group']=[5*n for n in counts]
        m.RECORD['checks']['figure11']['axis_order']={'x':'BPFO','y':'BPFI'}
        CHECKS['figure11_group_denominators_verified']=True
        name='Fig11_MCC5_support_physics_V2'
    base_finish(fig,name)

def main():
    m.finish=refined_finish
    m.RECORD['sources']={str(p.relative_to(ROOT)):m.digest(p) for p in m.FILES}
    m.figure6();m.figure8();m.figure11()
    assert m.RECORD['sources']=={str(p.relative_to(ROOT)):m.digest(p) for p in m.FILES}
    old=json.loads((ROOT/'paper/figures/revised_20261014/layout_and_data_integrity.json').read_text(encoding='utf-8'))
    assert m.RECORD['sources']==old['sources']
    # Quantitative arrays remain identical to the prior release.
    assert m.RECORD['checks']['figure8']==old['checks']['figure8']
    for key in ['points','centre_accuracy','ramef_accuracy']:
        assert m.RECORD['checks']['figure6'][key]==old['checks']['figure6'][key]
    for key in ['support','query_points','groups','dm8_accuracy','ramef_accuracy']:
        assert m.RECORD['checks']['figure11'][key]==old['checks']['figure11'][key]
    CHECKS['all_quantitative_arrays_equal_to_previous_release']=True
    CHECKS['source_hashes_equal_to_previous_release']=True
    m.RECORD['checks']['source_files_unchanged']=True
    m.RECORD['editorial_checks']=CHECKS
    (m.OUT/'layout_and_data_integrity.json').write_text(json.dumps(m.RECORD,indent=2),encoding='utf-8')
    print(m.OUT);print(json.dumps(CHECKS,indent=2))

if __name__=='__main__':main()
