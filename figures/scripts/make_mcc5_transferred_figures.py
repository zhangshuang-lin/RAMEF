"""Python-only publication figures for MCC5 transferred evidence package.

Figure contracts
----------------
1. Fair methods (quantitative grid): RAMEF should be located against frozen
   equal-input conventional and few-shot baselines, with seed spread visible.
2. Run corrections (quantitative heatmap): external gains should be distributed
   across runs/seeds rather than created by rejection or a few cherry-picked runs.
3. Support/context/physics (mixed quantitative grid): geometry evidence should
   stabilize target-support calibration and explain class/context behavior.

Exports are SVG and PDF only, as requested.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
PACKAGE = ROOT / "outputs" / "staged" / "MCC5_TRANSFERRED_EVIDENCE_PACKAGE"
OUT = PACKAGE / "figures"; OUT.mkdir(parents=True, exist_ok=True)
SEEDS = [20261101,20261102,20261103,20261104,20261105]
COLORS = {"neutral":"#8C8C8C","baseline":"#9ECAE1","u2":"#1F78B4","ramef":"#D95F02","physics":"#009E73","random":"#C77BAA"}

mpl.rcParams.update({"font.family":"serif","font.serif":["Times New Roman","Times","DejaVu Serif"],
                     "mathtext.fontset":"stix","svg.fonttype":"none","pdf.fonttype":42,
                     "font.size":8.2,"axes.titlesize":9.2,
                     "axes.labelsize":8.5,"xtick.labelsize":8.0,"ytick.labelsize":8.0,
                     "legend.fontsize":8.0,
                     "axes.spines.top":False,"axes.spines.right":False,"axes.linewidth":.7,
                     "legend.frameon":False})


def enforce_typography(fig, minimum=8.0):
    for artist in fig.findobj(match=lambda obj:isinstance(obj,mpl.text.Text)):
        artist.set_fontfamily("Times New Roman")
        if artist.get_fontsize()<minimum: artist.set_fontsize(minimum)


def export(fig,name):
    enforce_typography(fig)
    fig.savefig(OUT/f"{name}.svg",bbox_inches="tight")
    fig.savefig(OUT/f"{name}.pdf",bbox_inches="tight")
    fig.savefig(OUT/f"{name}.png",dpi=360,bbox_inches="tight",facecolor="white")
    plt.close(fig)


def panel(ax,label): ax.text(-.12,1.06,label,transform=ax.transAxes,fontweight="bold",fontsize=10.5,va="top")


def evidence_badge(ax, text, face, edge):
    ax.text(.985,.965,text,transform=ax.transAxes,ha="right",va="top",fontsize=5.4,
            fontweight="bold",color=edge,
            bbox={"boxstyle":"round,pad=.22","facecolor":face,
                  "edgecolor":edge,"linewidth":.55})


def fair_methods():
    d=pd.read_csv(PACKAGE/"T2_benchmark"/"seed_metrics.csv")
    order=["ProtoNet-DM8","ResNet1D-DM8","STFT-CNN-DM8","WDCNN-DM8","WDCNN-DANN-DM8","U2-Hz-DM8","RAMEF"]
    labels=["ProtoNet","ResNet1D","STFT-CNN","WDCNN","DANN","U2-Hz","RAMEF"]
    # Fixed method palette: conventional/few-shot comparators are deliberately
    # distinguishable instead of being collapsed into one grey category.
    # The hues follow a colour-blind-safe, print-compatible palette and are
    # shared by panels a and b so that method identity remains consistent.
    color=["#7F7F7F", "#8C6BB1", "#009E73", "#E6AB02", "#CC79A7",
           COLORS["u2"], COLORS["ramef"]]
    fig,axes=plt.subplots(2,2,figsize=(7.2,5.75),constrained_layout=True)
    ax=axes[0,0]
    for i,m in enumerate(order):
        v=100*d[d.method==m].accuracy.to_numpy();x=i+np.linspace(-.1,.1,len(v))
        ax.scatter(x,v,s=20,color=color[i],edgecolor="white",linewidth=.4,zorder=3)
        ax.hlines(v.mean(),i-.18,i+.18,color="#222",lw=1.2)
    ax.set_xticks(range(len(order)),labels,rotation=25,ha="right");ax.set_ylabel("Accuracy (%)");ax.set_ylim(30,103)
    ax.set_title("Fixed-query performance across five seeds");ax.grid(axis="y",color="#E8E8E8",lw=.6);panel(ax,"a")

    ax=axes[0,1];means=[];stds=[]
    for m in order:
        v=100*d[d.method==m].macro_f1.to_numpy();means.append(v.mean());stds.append(v.std(ddof=1))
    bars=ax.bar(range(len(order)),means,yerr=stds,color=color,edgecolor="#555",linewidth=.4,
                capsize=2,error_kw={"ecolor":"#111111","elinewidth":1.0,"capthick":1.0})
    ax.set_xticks(range(len(order)),labels,rotation=25,ha="right");ax.set_ylabel("Macro-F1 (%)");ax.set_ylim(25,108)
    ax.set_title("Mean ± sample SD across seeds");ax.grid(axis="y",color="#E8E8E8",lw=.6);panel(ax,"b")
    # Place labels above the upper error-bar cap, not above the bar itself.
    for bar,val,sd in zip(bars,means,stds):
        ax.text(bar.get_x()+bar.get_width()/2,val+sd+1.4,f"{val:.1f}",
                ha="center",va="bottom",fontsize=5.5)

    ax=axes[1,0];mat=[]
    for m in order: mat.append([100*d[d.method==m][k].mean() for k in ("recall_NRF","recall_IR","recall_OR")])
    im=ax.imshow(mat,aspect="auto",vmin=0,vmax=100,cmap=mpl.colors.LinearSegmentedColormap.from_list("rec",["#F7FBFF",COLORS["u2"]]))
    ax.set_xticks(range(3),["NRF","IR","OR"]);ax.set_yticks(range(len(order)),labels);ax.set_title("Class recall (%)")
    for i in range(len(order)):
        for j in range(3): ax.text(j,i,f"{mat[i][j]:.1f}",ha="center",va="center",fontsize=5.5,color="white" if mat[i][j]>60 else "#222")
    panel(ax,"c")

    ax=axes[1,1];ramef=d[d.method=="RAMEF"].sort_values("seed")
    compare=[m for m in order if m!="RAMEF"]
    for i,m in enumerate(compare):
        other=d[d.method==m].sort_values("seed");delta=100*(ramef.macro_f1.to_numpy()-other.macro_f1.to_numpy())
        jitter=np.linspace(-.08,.08,len(delta));ax.scatter(i+jitter,delta,s=18,color=COLORS["ramef"],edgecolor="white",linewidth=.4)
        ax.hlines(delta.mean(),i-.17,i+.17,color="#222",lw=1.2)
    ax.axhline(0,color="#777",lw=.7);ax.set_xticks(range(len(compare)),[labels[order.index(m)] for m in compare],rotation=25,ha="right")
    ax.set_ylabel("RAMEF − comparator Macro-F1 (pp)");ax.set_title("RAMEF gain over each frozen comparator",pad=8);ax.grid(axis="y",color="#E8E8E8",lw=.6);panel(ax,"d")
    ax.text(.99,.97,"RAMEF = 0 reference\n(self-comparison omitted)",transform=ax.transAxes,
            ha="right",va="top",fontsize=5.6,color="#555555")
    fig.suptitle("MCC5 post-hoc frozen equal-input method benchmark",fontsize=10,y=.995)
    fig.get_layout_engine().set(rect=(0,0,1,.94))
    export(fig,"Fig_MCC5_fair_method_benchmark")


def run_corrections():
    d=pd.read_csv(PACKAGE/"T3_run_errors"/"seed_run_correctness.csv")
    summary=pd.read_csv(PACKAGE/"T3_run_errors"/"run_correction_summary.csv")
    class_order={"NRF":0,"IR":1,"OR":2};summary=summary.sort_values(["class","nuisance","run_id"],key=lambda s:s.map(class_order) if s.name=="class" else s)
    run_ids=summary.run_id.tolist(); counts={}; short=[]
    for _, row in summary.iterrows():
        base=f"{row['class']} | {str(row['nuisance']).replace('_',' ')} | {int(row['torque'])}Nm"
        counts[base]=counts.get(base,0)+1; short.append(base+(f" #{counts[base]}" if counts[base]>1 else ""))
    fig=plt.figure(figsize=(7.2,7.0),constrained_layout=True);gs=fig.add_gridspec(2,3,height_ratios=[4.5,1.0])
    methods=["U2-Hz-LM8","U2-Hz-DM8","RAMEF"];cmaps=[ListedColormap(["#F2F2F2","#9ECAE1"]),ListedColormap(["#F2F2F2",COLORS["u2"]]),ListedColormap(["#F2F2F2",COLORS["ramef"]])]
    for j,(method,cmap) in enumerate(zip(methods,cmaps)):
        ax=fig.add_subplot(gs[0,j]);mat=np.zeros((len(run_ids),5))
        for i,rid in enumerate(run_ids):
            local=d[(d.method==method)&(d.run_id==rid)].sort_values("seed");mat[i]=local.correct.to_numpy()
        ax.imshow(mat,aspect="auto",vmin=0,vmax=1,cmap=cmap);ax.set_xticks(range(5),[str(s)[-2:] for s in SEEDS]);ax.set_xlabel("Seed suffix")
        ax.set_title(method);ax.set_yticks(range(len(run_ids)),short if j==0 else []);panel(ax,chr(97+j))
        for y in [sum(summary["class"]=="NRF")-.5,sum(summary["class"].isin(["NRF","IR"]))-.5]:ax.axhline(y,color="#222",lw=.8)
    fig.legend(handles=[Patch(facecolor="#7F7F7F",edgecolor="#555555",label="Correct"),
                        Patch(facecolor="#F2F2F2",edgecolor="#777777",label="Incorrect")],
               loc="upper center",bbox_to_anchor=(.5,.955),ncol=2,fontsize=6.2,
               frameon=False,handlelength=1.2,columnspacing=1.3)
    ax=fig.add_subplot(gs[1,:]);keys=["dm8_fixes_lm8","dm8_new_errors","ramef_fixes_dm8","ramef_new_errors"]
    vals=[summary[k].sum() for k in keys];labs=["DM8 fixes LM8","DM8 new errors","RAMEF fixes DM8","RAMEF new errors"]
    bars=ax.bar(range(4),vals,color=[COLORS["u2"],"#D8A6A6",COLORS["ramef"],"#D8A6A6"],edgecolor="#555",linewidth=.4)
    ax.set_xticks(range(4),labs);ax.set_ylabel("Paired seed-run transitions");ax.set_ylim(0,max(vals)+5);ax.grid(axis="y",color="#E8E8E8",lw=.6);panel(ax,"d")
    for b,v in zip(bars,vals):ax.text(b.get_x()+b.get_width()/2,v+.5,str(int(v)),ha="center",fontsize=7)
    fig.suptitle("MCC5 run-level correction stability (27 runs × 5 seeds)",fontsize=10,y=.995)
    fig.get_layout_engine().set(rect=(0,0,1,.91))
    export(fig,"Fig_MCC5_run_level_corrections")


def support_context_physics():
    loo=pd.read_csv(PACKAGE/"T1_support"/"support_loo_metrics.csv")
    leave=pd.read_csv(PACKAGE/"T1_support"/"leave_two_out_metrics.csv")
    phys=pd.read_csv(PACKAGE/"T5_context_physics"/"physical_evidence_by_run.csv")
    group=pd.read_csv(PACKAGE/"T5_context_physics"/"group_performance.csv")
    fig,axes=plt.subplots(2,2,figsize=(7.2,6.05),constrained_layout=True)
    ax=axes[0,0];weights=[1,5,10,20]
    ax.set_facecolor("#F6FAFD")
    for view,color,marker in [("U2-Hz-DM8",COLORS["u2"],"s"),("RAMEF",COLORS["ramef"],"o")]:
        means=[];stds=[]
        for w in weights:
            v=100*loo[(loo.view==view)&(loo.condition==f"source_support_w{w}_loo")].accuracy.to_numpy();means.append(v.mean());stds.append(v.std(ddof=1))
        ax.errorbar(weights,means,yerr=stds,color=color,marker=marker,label=view,capsize=2)
    ax.set_xscale("log");ax.set_xticks(weights,weights);ax.set_xlabel("Target-support weight");ax.set_ylabel("Support LOO accuracy (%)");ax.set_ylim(45,103);ax.legend();ax.grid(color="#E8E8E8",lw=.6);ax.set_title("Calibration stability");evidence_badge(ax,"QUERY-FREE","#EAF4FB","#1F78B4");panel(ax,"a")

    ax=axes[0,1]
    ax.set_facecolor("#F6FAFD")
    for i,(view,color) in enumerate([("U2-Hz-DM8",COLORS["u2"]),("RAMEF",COLORS["ramef"])]):
        v=100*leave[leave.view==view].accuracy.to_numpy();ax.scatter(np.full(len(v),i)+np.linspace(-.08,.08,len(v)),v,color=color,s=20,edgecolor="white",linewidth=.4);ax.hlines(v.mean(),i-.18,i+.18,color="#222",lw=1.2)
    ax.set_xticks([0,1],["U2-Hz-DM8","RAMEF"]);ax.set_ylabel("Leave-two-out accuracy (%)");ax.set_ylim(45,103);ax.set_title("105 folds per seed");ax.grid(axis="y",color="#E8E8E8",lw=.6);evidence_badge(ax,"QUERY-FREE","#EAF4FB","#1F78B4");panel(ax,"b")

    ax=axes[1,0];q=phys[phys.role=="query"]
    ax.set_facecolor("#FCF8F3")
    for cls,color,marker in [("NRF","#777","o"),("IR",COLORS["u2"],"s"),("OR",COLORS["ramef"],"^")]:
        local=q[q["class"]==cls];ax.scatter(local.bpfo_mean,local.bpfi_mean,label=cls,color=color,marker=marker,s=28,edgecolor="white",linewidth=.4)
    lo=min(q.bpfo_mean.min(),q.bpfi_mean.min());hi=max(q.bpfo_mean.max(),q.bpfi_mean.max());ax.plot([lo,hi],[lo,hi],ls="--",color="#777",lw=.7)
    ax.set_xlabel("BPFO soft evidence");ax.set_ylabel("BPFI soft evidence");ax.legend();ax.set_title("Frozen-query physical evidence");evidence_badge(ax,"POST-HOC DESCRIPTIVE","#FFF4E8","#A94B00");panel(ax,"c")

    ax=axes[1,1];g=group[(group.group_type=="nuisance")&(group.runs>=2)];groups=sorted(g.group.unique());x=np.arange(len(groups));w=.36
    ax.set_facecolor("#FCF8F3")
    base=[100*g[(g.group==v)&(g.method=="U2-Hz-DM8")].accuracy.iloc[0] for v in groups]
    ramef=[100*g[(g.group==v)&(g.method=="RAMEF")].accuracy.iloc[0] for v in groups]
    ax.bar(x-w/2,base,w,color=COLORS["u2"],label="U2-Hz-DM8");ax.bar(x+w/2,ramef,w,color=COLORS["ramef"],label="RAMEF")
    tick_labels=["bend","broken\nbar","dynamic\necc.","pure\nrace","static\necc.","voltage\nunbalance","winding"]
    ax.set_xticks(x,tick_labels,rotation=0);ax.set_ylabel("Accuracy over seed-run predictions (%)");ax.set_ylim(0,105);ax.legend(fontsize=8,loc="upper center",bbox_to_anchor=(.5,-.24),ncol=2,borderaxespad=0);ax.set_title("Nuisance-family behavior");ax.grid(axis="y",color="#E8E8E8",lw=.6);evidence_badge(ax,"POST-HOC DESCRIPTIVE","#FFF4E8","#A94B00");panel(ax,"d")
    fig.suptitle("MCC5 evidence permissions: query-free support audit and post-hoc physical context",fontsize=10,y=.995)
    fig.get_layout_engine().set(rect=(0,0,1,.93))
    export(fig,"Fig_MCC5_support_context_physics")


def main(): fair_methods();run_corrections();support_context_physics();print(OUT)
if __name__=="__main__":main()
