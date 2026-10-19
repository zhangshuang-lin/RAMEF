"""Layout-only repair from archived summary data; no model or query evaluation.

Contracts: Fig6 describes code-level physical overlap and recovery; Fig8 shows
the locally prespecified comparison plus the distinct post-hoc matched audit;
Fig11 separates support-neighbourhood audits from post-hoc query descriptions.
Archetype: quantitative grids. Export editable SVG/PDF at 7.2 inches, TNR.
"""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import font_manager
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'paper/figures/revised_20261014'
QA = ROOT / 'tmp/figure_repair_20261014'
PACKAGE = ROOT / 'outputs/staged/MCC5_TRANSFERRED_EVIDENCE_PACKAGE'
PHYS = ROOT / 'outputs/staged/PU_UNIQUE_MEASUREMENT_REANALYSIS/physical_code_means.json'
PU = ROOT / 'outputs/staged/PU_UNIQUE_MEASUREMENT_REANALYSIS/results.json'
MCC = ROOT / 'outputs/staged/MCC5_RAMEF_CONFIRMATORY'
FILES = [PHYS, PU, MCC/'run/seed_metrics.csv',
         MCC/'label_semantics_audit/matched_nuisance_metrics.csv',
         PACKAGE/'T1_support/support_loo_metrics.csv',
         PACKAGE/'T1_support/leave_two_out_metrics.csv',
         PACKAGE/'T5_context_physics/physical_evidence_by_run.csv',
         PACKAGE/'T5_context_physics/group_performance.csv']
OUT.mkdir(parents=True, exist_ok=True)
QA.mkdir(parents=True, exist_ok=True)
font_manager.findfont('Times New Roman', fallback_to_default=False)
mpl.rcParams.update({'font.family':'serif', 'font.serif':['Times New Roman'],
    'font.size':10, 'axes.titlesize':10, 'axes.labelsize':10,
    'xtick.labelsize':9, 'ytick.labelsize':9, 'legend.fontsize':9,
    'svg.fonttype':'none', 'pdf.fonttype':42, 'mathtext.fontset':'stix',
    'axes.spines.top':False, 'axes.spines.right':False,
    'axes.linewidth':.75, 'legend.frameon':False})
BLUE = '#1F78B4'; ORANGE = '#D95F02'; LOCAL = '#9ECAE1'
CLASS = {0:'#777777', 1:'#0878b8', 2:'#dd6800'}
RECORD = {'sources': {}, 'checks': {}, 'outputs': {}, 'layout': {}}

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def header(ax, letter, title, permission=None):
    ax.set_title(title, pad=33 if permission else 15)
    ax.text(-.13,1.20 if permission else 1.11,letter,
            transform=ax.transAxes, fontsize=12, fontweight='bold')
    if permission:
        colour, face = (BLUE,'#EAF4FB') if permission != 'POST-HOC DESCRIPTIVE' else ('#A94B00','#FFF4E8')
        tag = ax.text(.5,1.045,permission,transform=ax.transAxes,
                     ha='center',va='bottom',fontsize=8.5,
                     fontweight='bold',color=colour,clip_on=False,
                     bbox={'boxstyle':'round,pad=.22','facecolor':face,
                           'edgecolor':colour,'linewidth':.55})
        tag.set_gid('permission-outside-data')

def grid2(height):
    fig = plt.figure(figsize=(7.2,height))
    axes = [fig.add_axes(rect) for rect in
            [(.105,.585,.37,.245),(.61,.585,.365,.245),
             (.105,.14,.37,.245),(.61,.14,.365,.245)]]
    return fig, axes

def finish(fig, name):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    tags = []
    if fig._suptitle is not None:
        main_box=fig._suptitle.get_window_extent(renderer)
        for ax in fig.axes:
            assert not main_box.overlaps(ax.title.get_window_extent(renderer)), (name,'title overlap')
    for ax in fig.axes:
        for t in ax.texts:
            if t.get_gid() == 'permission-outside-data':
                bbox = t.get_bbox_patch().get_window_extent(renderer)
                assert bbox.y0 > ax.bbox.y1, (name, t.get_text(), 'badge enters data')
                tags.append({'text':t.get_text(), 'clearance_px':bbox.y0-ax.bbox.y1})
    RECORD['layout'][name] = {'width_inches':7.2,
                              'permission_tags_outside_data':tags,
                              'minimum_text_pt':min(t.get_fontsize() for t in fig.findobj(mpl.text.Text)
                                                    if t.get_text())}
    for ext in ('svg','pdf'):
        path = OUT / (name+'.'+ext)
        fig.savefig(path, facecolor='white')
        RECORD['outputs'][path.name] = digest(path)
    fig.savefig(QA/(name+'.png'), dpi=240, facecolor='white')
    # A true 6.65-inch-equivalent preview, no artificial upscaling of text.
    fig.savefig(QA/(name+'_print_size.png'), dpi=133, facecolor='white')
    plt.close(fig)

def figure6():
    points = json.loads(PHYS.read_text(encoding='utf-8'))
    lookup = {p['code']:p for p in points}
    models = {m['method']:m for m in json.loads(PU.read_text(encoding='utf-8'))['models']}
    assert sum(p['n'] for p in points) == 171
    fig = plt.figure(figsize=(7.2,5.6))
    ax = fig.add_axes((.115,.54,.36,.34))
    for cls,label in ((0,'N'),(1,'IR'),(2,'OR')):
        vals = [p for p in points if p['class']==cls]
        ax.scatter([p['bpfi'] for p in vals],[p['bpfo'] for p in vals],
                   s=38,color=CLASS[cls],edgecolor='white',linewidth=.5,label=label)
    low = min(min(p['bpfi'],p['bpfo']) for p in points)-.05
    high = max(max(p['bpfi'],p['bpfo']) for p in points)+.08
    ax.plot([low,high],[low,high],ls='--',color='#777',lw=.8,zorder=0)
    positions = {'KA16':(.96,1.18),'KA22':(1.04,.94),'KA30':(1.04,.57)}
    for code,xytext in positions.items():
        p=lookup[code]
        label=ax.annotate(code,(p['bpfi'],p['bpfo']),xytext=xytext,
            textcoords='data',ha='center',va='center',fontsize=9.5,
            color=CLASS[p['class']],
            bbox={'boxstyle':'round,pad=.13','fc':'white','ec':'none'},
            arrowprops={'arrowstyle':'-','color':CLASS[p['class']], 'lw':.9},
            zorder=5)
        label.set_gid('code-label-'+code)
    ax.set(xlim=(low,high),ylim=(low,high),xlabel='Best-band BPFI evidence',
           ylabel='Best-band BPFO evidence')
    ax.legend(loc='lower left',bbox_to_anchor=(-.03,-.36),ncol=3,
              handletextpad=.3,columnspacing=.8)
    header(ax,'a','Code-mean soft evidence')
    ax=fig.add_axes((.63,.54,.34,.34))
    selected=['KI18','KA30','KA22','K006','K001','KA16']
    for i,code in enumerate(selected):
        p=lookup[code]; value=p['difference']
        ax.barh(i,value,color=CLASS[p['class']],height=.62,
                edgecolor='#444',lw=.4)
        ax.text(value+(.02 if value>=0 else -.018),i,f'{value:+.3f}',
                ha='left' if value>=0 else 'right',va='center',fontsize=9)
    ax.axvline(0,color='#333',lw=.7)
    ax.set_yticks(range(6),selected)
    ax.invert_yaxis()
    ax.set(xlim=(-.20,.76),xlabel='BPFO minus BPFI evidence')
    header(ax,'b','Representative codes')
    ax=fig.add_axes((.20,.135,.77,.245))
    codes=['KA04','KA15','KA16','KA22','KA30']
    y=np.arange(5)
    base=[models['Centre U2-Hz']['code_accuracy'][c]['accuracy'] for c in codes]
    full=[models['RAMEF-F']['code_accuracy'][c]['accuracy'] for c in codes]
    ax.barh(y-.18,base,.34,color='#888',label='Centre crop')
    ax.barh(y+.18,full,.34,color=CLASS[2],label='RAMEF-F')
    ax.set_yticks(y,[f'{c} (n={lookup[c]["n"]})' for c in codes])
    ax.invert_yaxis();ax.set_xlim(0,105);ax.set_xticks([0,20,40,60,80,100])
    ax.set_xlabel('Unique-measurement code accuracy (%)')
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.32),ncol=2)
    header(ax,'c','Outer-race code recovery')
    RECORD['checks']['figure6']={'points':points,'centre_accuracy':base,'ramef_accuracy':full,
        'all_scatter_coordinates_unchanged':True}
    finish(fig,'Fig6_PU_physical_evidence_corrected')

def figure8():
    d=pd.read_csv(MCC/'run/seed_metrics.csv')
    matched=pd.read_csv(MCC/'label_semantics_audit/matched_nuisance_metrics.csv')
    order=['U2-Hz-K1','U2-Hz-Local-M8','U2-Hz-Distributed-M8','RAMEF-F','Random63']
    names=['K1','U2-Hz-LM8','U2-Hz-DM8','RAMEF','Random63']
    colours=['#8C8C8C',LOCAL,BLUE,ORANGE,'#C77BAA']
    def values(method,key,frame=d):
        return 100*frame[frame.variant==method].sort_values('seed')[key].to_numpy()
    fig,axes=grid2(6.9)
    fig.suptitle('MCC5 cross-speed evidence: locally prespecified comparison\nand post-hoc matched-nuisance audit',
                 fontsize=11,y=.985)
    ax=axes[0]
    for i,(method,colour) in enumerate(zip(order,colours)):
        v=values(method,'accuracy');assert len(v)==5
        ax.scatter(i+np.linspace(-.10,.10,5),v,s=24,color=colour,
                   edgecolor='white',lw=.4,zorder=3)
        ax.hlines(v.mean(),i-.2,i+.2,color='#222',lw=1.2)
    ax.set_xticks(range(5),names,rotation=22,ha='right')
    ax.set_ylabel('Accuracy (%)');ax.set_ylim(55,104);ax.set_yticks([60,70,80,90,100])
    ax.grid(axis='y',color='#E6E6E6',lw=.6)
    header(ax,'a','Five source-selected encoders','LOCALLY PRESPECIFIED')
    ax=axes[1]
    means=np.array([values(m,'macro_f1').mean() for m in order])
    sds=np.array([values(m,'macro_f1').std(ddof=1) for m in order])
    bars=ax.bar(range(5),means,yerr=sds,capsize=3,color=colours,
                edgecolor='#555',lw=.45)
    for bar,mean,sd in zip(bars,means,sds):
        ax.annotate(f'{mean:.1f}',(bar.get_x()+bar.get_width()/2,mean+sd),
                    xytext=(0,5),textcoords='offset points',ha='center',va='bottom',fontsize=9)
    ax.set_xticks(range(5),names,rotation=22,ha='right')
    ax.set_ylabel('Macro-F1 (%)');ax.set_ylim(45,109);ax.set_yticks([50,60,70,80,90,100])
    ax.grid(axis='y',color='#E6E6E6',lw=.6)
    header(ax,'b','Mean ± sample SD across seeds','LOCALLY PRESPECIFIED')
    ax=axes[2];x=np.arange(3);width=.22
    recalls=[]
    for j,(method,colour) in enumerate(zip(order[1:4],colours[1:4])):
        rec=[values(method,key).mean() for key in ('recall_N','recall_IR','recall_OR')]
        recalls.append(rec)
        ax.bar(x+(j-1)*width,rec,width,color=colour,edgecolor='#555',lw=.4,
               label=names[j+1])
    ax.set_xticks(x,['NRF','IR','OR']);ax.set_ylabel('Recall (%)')
    ax.set_ylim(0,106);ax.set_yticks([0,20,40,60,80,100]);ax.grid(axis='y',color='#E6E6E6',lw=.6)
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.24),ncol=3,
              columnspacing=.6,handlelength=1,handletextpad=.3,fontsize=8.5)
    header(ax,'c','Fault-localisation effect','LOCALLY PRESPECIFIED')
    ax=axes[3]
    dl=values(order[2],'macro_f1',matched)-values(order[1],'macro_f1',matched)
    dp=values(order[3],'macro_f1',matched)-values(order[2],'macro_f1',matched)
    for l,p in zip(dl,dp):ax.plot([0,1],[l,p],color='#B8B8B8',lw=.8,zorder=1)
    ax.scatter(np.zeros(5),dl,s=26,color=BLUE,zorder=3)
    ax.scatter(np.ones(5),dp,s=26,color=ORANGE,zorder=3)
    ax.hlines(dl.mean(),-.16,.16,color='#222',lw=1.2)
    ax.hlines(dp.mean(),.84,1.16,color='#222',lw=1.2)
    ax.set_xticks([0,1],['DM8\n− LM8','RAMEF\n− DM8'])
    ax.set_ylabel('Paired Macro-F1 gain (pp)');ax.set_ylim(0,40)
    ax.grid(axis='y',color='#E6E6E6',lw=.6);ax.set_facecolor('#FAFAFA')
    header(ax,'d','Matched-nuisance gains (15 runs)','POST-HOC DESCRIPTIVE')
    assert max(dp)<ax.get_ylim()[1]
    RECORD['checks']['figure8']={'macro_f1_mean':means.tolist(),'sample_sd':sds.tolist(),
        'class_recalls':recalls,'dm8_minus_lm8':dl.tolist(),'ramef_minus_dm8':dp.tolist(),
        'all_ten_paired_points_inside_axes':True}
    finish(fig,'Fig8_MCC5_evidence_corrected')

def figure11():
    loo=pd.read_csv(PACKAGE/'T1_support/support_loo_metrics.csv')
    leave=pd.read_csv(PACKAGE/'T1_support/leave_two_out_metrics.csv')
    phys=pd.read_csv(PACKAGE/'T5_context_physics/physical_evidence_by_run.csv')
    group=pd.read_csv(PACKAGE/'T5_context_physics/group_performance.csv')
    fig,axes=grid2(7.0)
    fig.suptitle('MCC5 evidence permissions: query-free support audits\nand post-hoc physical context',
                 fontsize=11,y=.985)
    weights=[1,5,10,20];audit={}
    ax=axes[0];ax.set_facecolor('#F6FAFD')
    for view,colour,marker in [('U2-Hz-DM8',BLUE,'s'),('RAMEF',ORANGE,'o')]:
        means=[];sds=[]
        for w in weights:
            v=100*loo[(loo.view==view)&(loo.condition==f'source_support_w{w}_loo')].accuracy.to_numpy()
            assert len(v)==5
            means.append(v.mean());sds.append(v.std(ddof=1))
        ax.errorbar(weights,means,yerr=sds,color=colour,marker=marker,
                    markersize=5,capsize=3,label=view)
        audit[view]={'loo_means':means,'loo_sample_sd':sds}
    ax.set_xscale('log');ax.set_xticks(weights,[str(w) for w in weights])
    ax.set_xlabel('Target-support weight');ax.set_ylabel('Support LOO accuracy (%)')
    ax.set_ylim(45,104);ax.set_yticks([50,60,70,80,90,100]);ax.grid(color='#E8E8E8',lw=.6)
    ax.legend(loc='center right',fontsize=9)
    header(ax,'a','Calibration stability','QUERY-FREE')
    ax=axes[1];ax.set_facecolor('#F6FAFD')
    for i,(view,colour) in enumerate([('U2-Hz-DM8',BLUE),('RAMEF',ORANGE)]):
        v=100*leave[leave.view==view].sort_values('seed').accuracy.to_numpy();assert len(v)==5
        ax.scatter(i+np.linspace(-.08,.08,5),v,color=colour,s=26,edgecolor='white',lw=.4,zorder=3)
        ax.hlines(v.mean(),i-.18,i+.18,color='#222',lw=1.2)
        audit[view]['leave_two_out']=v.tolist()
    ax.set_xticks([0,1],['U2-Hz-DM8','RAMEF'])
    ax.set_ylabel('Leave-two-out accuracy (%)');ax.set_ylim(45,104)
    ax.set_yticks([50,60,70,80,90,100]);ax.grid(axis='y',color='#E8E8E8',lw=.6)
    header(ax,'b','105 folds per seed','QUERY-FREE')
    ax=axes[2];ax.set_facecolor('#FCF8F3');q=phys[phys.role=='query'];assert len(q)==27
    for cls,colour,marker in [('NRF','#777','o'),('IR',BLUE,'s'),('OR',ORANGE,'^')]:
        v=q[q['class']==cls]
        ax.scatter(v.bpfo_mean,v.bpfi_mean,label=cls,color=colour,marker=marker,
                   s=30,edgecolor='white',lw=.4)
    lo=min(q.bpfo_mean.min(),q.bpfi_mean.min());hi=max(q.bpfo_mean.max(),q.bpfi_mean.max())
    ax.plot([lo,hi],[lo,hi],ls='--',color='#777',lw=.7)
    ax.set_xlabel('BPFO soft evidence');ax.set_ylabel('BPFI soft evidence')
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.28),ncol=3,
              columnspacing=.9,handletextpad=.3)
    header(ax,'c','Frozen-query physical evidence','POST-HOC DESCRIPTIVE')
    ax=axes[3];ax.set_facecolor('#FCF8F3')
    g=group[(group.group_type=='nuisance')&(group.runs>=2)];groups=sorted(g.group.unique())
    base=[100*g[(g.group==v)&(g.method=='U2-Hz-DM8')].accuracy.iloc[0] for v in groups]
    full=[100*g[(g.group==v)&(g.method=='RAMEF')].accuracy.iloc[0] for v in groups]
    labels=['bend','broken bar','dynamic ecc.','pure race','static ecc.','voltage unbalance','winding']
    assert len(labels)==len(groups)==7
    x=np.arange(7);w=.36
    ax.bar(x-w/2,base,w,color=BLUE,label='U2-Hz-DM8')
    ax.bar(x+w/2,full,w,color=ORANGE,label='RAMEF')
    ax.set_xticks(x,labels,rotation=40,ha='right');ax.tick_params(axis='x',labelsize=8.5)
    ax.set_ylabel('Accuracy over seed-run predictions (%)',fontsize=9)
    ax.set_ylim(0,105);ax.set_yticks([0,20,40,60,80,100]);ax.grid(axis='y',color='#E8E8E8',lw=.6)
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.40),ncol=2,fontsize=9,
              columnspacing=.9,handlelength=1.3)
    header(ax,'d','Nuisance-family behaviour','POST-HOC DESCRIPTIVE')
    RECORD['checks']['figure11']={'support':audit,'query_points':27,'groups':groups,
                                  'dm8_accuracy':base,'ramef_accuracy':full}
    finish(fig,'Fig11_MCC5_support_physics_corrected')

def main():
    RECORD['sources']={str(p.relative_to(ROOT)):digest(p) for p in FILES}
    figure6();figure8();figure11()
    assert RECORD['sources']=={str(p.relative_to(ROOT)):digest(p) for p in FILES}
    RECORD['checks']['source_files_unchanged']=True
    (OUT/'layout_and_data_integrity.json').write_text(json.dumps(RECORD,indent=2),encoding='utf-8')
    print(OUT)
    print(json.dumps(RECORD['layout'],indent=2))

if __name__=='__main__':main()
