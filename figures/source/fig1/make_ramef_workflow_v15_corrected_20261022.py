"""Draw the implementation-grounded RAMEF workflow from existing source traces."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / 'tmp/fig1_revision_20261022'
sys.path.insert(0, str(WORK / 'python_packages'))
os.environ['MPLCONFIGDIR'] = str(WORK / 'mplconfig')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np

DATA = ROOT / 'paper/figures/Fig1_RAMEF_CleanAxes_20261020'
OUT = ROOT / 'paper/figures/Fig1_RAMEF_Revised_20261022'
W, H = 177.8, 198.0
INK, GREY, BORDER = '#26333d', '#61707b', '#c5cdd2'
BLUE, WARM, GREEN = '#487ba1', '#ab7951', '#4e826b'
BLUE_BG, WARM_BG, GREEN_BG = '#f0f5f9', '#faf5f0', '#e8f2ec'
TEXT, BOXES, ARROWS = [], [], []


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def text(ax, x, y, value, *, size=10.5, bold=False, color=INK,
         ha='center', va='center', box=None, rotation=0):
    artist = ax.text(x, y, value, fontsize=size, weight='bold' if bold else 'normal',
                     color=color, ha=ha, va=va, linespacing=1.05,
                     rotation=rotation, zorder=5)
    TEXT.append((artist, box))
    return artist


def box(ax, name, x, y, w, h, value, *, color=BORDER, fill='#f6f7f8', size=10.5):
    patch = FancyBboxPatch((x,y), w,h, boxstyle='round,pad=0,rounding_size=0.7',
                          ec=color, fc=fill, lw=.8, zorder=2)
    ax.add_patch(patch)
    BOXES.append((name,patch))
    if value:
        text(ax,x+w/2,y+h/2,value,size=size,box=patch)
    return patch


def arrow(ax, points, *, color=GREY, dashed=False):
    if len(points)>2:
        x,y=zip(*points[:-1])
        ax.plot(x,y,color=color,lw=.9,ls=(0,(3,2)) if dashed else '-',zorder=3)
    p=FancyArrowPatch(points[-2],points[-1],arrowstyle='-|>',mutation_scale=9,
                      color=color,lw=.9,linestyle=(0,(3,2)) if dashed else '-',
                      shrinkA=0,shrinkB=0,zorder=3)
    ax.add_patch(p)
    ARROWS.append({'points':points,'dashed':dashed})


def inset(fig, x, y, w, h):
    # Logical drawing coordinates have an additional 3-mm bottom margin.
    ax=fig.add_axes([x/W,(y+3)/H,w/W,h/H])
    for side in ('top','right'):
        ax.spines[side].set_visible(False)
    for side in ('left','bottom'):
        ax.spines[side].set_color(INK)
        ax.spines[side].set_linewidth(.75)
    ax.tick_params(labelsize=10.5,length=2.4,pad=1.5,width=.7,
                   color=INK,labelcolor=INK)
    return ax


def main(*, panel_a_builder=None, panel_b_title=None):
    OUT.mkdir(parents=True,exist_ok=True)
    WORK.mkdir(parents=True,exist_ok=True)
    matplotlib.rcParams.update({
        'font.family':'Times New Roman','font.size':10.5,
        'mathtext.fontset':'custom','mathtext.rm':'Times New Roman',
        'mathtext.it':'Times New Roman:italic','mathtext.bf':'Times New Roman:bold',
        'mathtext.fallback':'stix','pdf.fonttype':42,'ps.fonttype':42,
        'svg.fonttype':'none','axes.unicode_minus':True,
        'savefig.pad_inches':0,'path.simplify':False,
    })
    assert 'times' in font_manager.findfont('Times New Roman').lower()
    data_hashes={}
    for name in ('overview_bin_ranges.csv','raw_local_detail.csv'):
        data_hashes[name]=sha(DATA/name)
        shutil.copy2(DATA/name,OUT/name)
    with (DATA/'overview_bin_ranges.csv').open(newline='',encoding='utf-8-sig') as f:
        overview=list(csv.DictReader(f))
    with (DATA/'raw_local_detail.csv').open(newline='',encoding='utf-8-sig') as f:
        local=list(csv.DictReader(f))
    assert len(local)==640

    fig=plt.figure(figsize=(W/25.4,H/25.4),facecolor='white')
    ax=fig.add_axes([0,3/H,1,195/H]);ax.set_xlim(0,W);ax.set_ylim(0,195);ax.axis('off')

    text(ax,4,189,'Source-only representation learning',size=12.2,bold=True,ha='left')
    box(ax,'source_inputs',4,171,50,12,'PU: centre crops\nMCC5: distributed blocks',fill='white')
    box(ax,'source_train',60,171,25,12,'Train\nU2-Hz',fill='white')
    box(ax,'source_validate',91,171,47,12,'Select on\nsource validation',fill='white')
    box(ax,'source_freeze',144,171,30,12,'Freeze\nencoder',fill='white')
    for start,end in ((54,60),(85,91),(138,144)):
        arrow(ax,[(start,177),(end,177)])
    ax.plot([4,174],[167,167],color='#d9dfe3',lw=.65)

    panel_a_extra={}
    if panel_a_builder is not None:
        full,detail,panel_a_extra=panel_a_builder(fig,ax,overview,local)
    else:
        text(ax,4,160,'a   Distributed observation',size=12.2,bold=True,ha='left')
        text(ax,14,154,'Source record',ha='left')
        text(ax,108,154,'Block 4: first 10 ms',ha='left')
        full=inset(fig,14,137,60,14)
        full.vlines([float(r['time_bin_centre_s'])for r in overview],
                    [float(r['minimum'])for r in overview],
                    [float(r['maximum'])for r in overview],color=INK,lw=.28)
        full.set_xlim(0,4);full.set_ylim(-2,2)
        full.set_xticks([0,2,4]);full.set_yticks([-2,0,2])
        detail=inset(fig,108,137,62,14)
        detail.plot([float(r['elapsed_ms'])for r in local],
                    [float(r['recorded_amplitude'])for r in local],color=INK,lw=.45)
        detail.lines[0].get_path().should_simplify=False
        detail.set_xlim(0,10);detail.set_ylim(-.8,.8)
        detail.set_xticks([0,5,10]);detail.set_yticks([-.5,0,.5],labels=['−0.5','0','0.5'])
        text(ax,44,129,'Time (s)')
        text(ax,139,129,'Time (ms)')
        text(ax,4,144,'Amplitude',rotation=90)
        text(ax,98,144,'Amplitude',rotation=90)
        for i in range(8):
            ax.add_patch(Rectangle((9+i*6,122.5),2.1,3,ec=BLUE,fc=BLUE_BG,lw=.65))
        text(ax,60,124,'8 non-overlapping blocks; one decision per record',ha='left')
        ax.plot([4,174],[118,118],color='#d9dfe3',lw=.65)

    text(ax,4,112,panel_b_title or 'b   Frozen dual-path evidence extraction',size=12.2,bold=True,ha='left')
    text(ax,4,106,'Carrier path',size=11,bold=True,color=BLUE,ha='left')
    box(ax,'carrier_preprocess',4,88.5,44,15,'Standardise blocks\nHz log-STFT',color=BLUE,fill=BLUE_BG)
    box(ax,'carrier_encode_mean',55,88.5,79,15,
        'Frozen U2-Hz; mean over 8 blocks\n'+r'$\overline{\mathbf{z}}_i$: 1024 stored (128 unique)',
        color=BLUE,fill=BLUE_BG)
    arrow(ax,[(48,96),(55,96)],color=BLUE)
    arrow(ax,[(134,96),(146,96)],color=BLUE)
    box(ax,'late_fusion',146,80,28,22,'Late concat.\n'+r'$\mathbf{r}_i=[\overline{\mathbf{z}}_i;\mathbf{p}_i]$')
    text(ax,160,75,'1087 stored')

    text(ax,4,83,'Physical path',size=11,bold=True,color=WARM,ha='left')
    box(ax,'speed_geometry',62,81,50,6,'Block RPM + orders',fill='white')
    arrow(ax,[(87,81),(87,79)],color=GREY)
    box(ax,'bandpass_waveform',4,63,30,16,'Band-pass\nwaveform',color=WARM,fill=WARM_BG)
    box(ax,'soft_order_evidence',41,63,63,16,
        'Hilbert envelope → FFT power\nSoft BPFO/BPFI evidence\nGaussian core / local background',
        color=WARM,fill=WARM_BG)
    box(ax,'physical_statistics',111,63,29,16,'Cross-block\nstatistics\n'+r'$\mathbf{p}_i\in\mathbb{R}^{63}$',
        color=WARM,fill=WARM_BG)
    arrow(ax,[(34,71),(41,71)],color=WARM)
    arrow(ax,[(104,71),(111,71)],color=WARM)
    arrow(ax,[(140,71),(143,71),(143,84),(146,84)],color=WARM)
    box(ax,'filtered_statistics',40,49,64,10,
        'Filtered-waveform statistics\nKurtosis + log(1+RMS)',color=WARM,fill='white')
    arrow(ax,[(19,63),(19,54),(40,54)],color=WARM)
    arrow(ax,[(104,54),(125.5,54),(125.5,63)],color=WARM)
    ax.plot([4,174],[45,45],color='#d9dfe3',lw=.65)

    text(ax,4,39,'c   Support calibration and query inference',size=12.2,bold=True,ha='left')
    box(ax,'labelled_features',4,19,50,12,'Source + support features\n5 labelled supports/class')
    box(ax,'fit_calibration',61,19,64,12,
        'Fit StandardScaler\n+ weighted multinomial LR')
    box(ax,'freeze_calibration',136,19,38,12,'Freeze scaler\nand classifier')
    arrow(ax,[(54,25),(61,25)])
    arrow(ax,[(125,25),(136,25)])
    box(ax,'query_record',4,.5,24,12,'Query\nrecord',fill='white')
    box(ax,'frozen_extraction',35,.5,53,12,'Same frozen extractor\nPanels a–b',fill='white')
    box(ax,'query_inference',95,.5,40,12,'Query inference\nFrozen calibration',color=GREEN,fill=GREEN_BG)
    box(ax,'prediction',142,.5,32,12,'PU: N/IR/OR\nMCC5: NRF/IR/OR',fill='white')
    for start,end in ((28,35),(88,95),(135,142)):
        arrow(ax,[(start,6.5),(end,6.5)])
    arrow(ax,[(155,19),(155,16),(115,16),(115,12.5)],dashed=True)

    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    issues=[]
    for artist,patch in TEXT:
        bbox=artist.get_window_extent(renderer)
        if patch is not None:
            outer=patch.get_window_extent(renderer)
            if not (bbox.x0>outer.x0+2 and bbox.x1<outer.x1-2 and
                    bbox.y0>outer.y0+2 and bbox.y1<outer.y1-2):
                issues.append({'kind':'text_box_fit','text':artist.get_text(),
                               'text_bbox':list(bbox.bounds),'box_bbox':list(outer.bounds)})
    alltexts=TEXT+[(t,None)for a in (full,detail)for t in a.get_xticklabels()+a.get_yticklabels()]
    for i,(a,_) in enumerate(alltexts):
        aa=a.get_window_extent(renderer)
        for b,_ in alltexts[i+1:]:
            bb=b.get_window_extent(renderer)
            if aa.overlaps(bb):
                overlap=aa.intersection(aa,bb)
                if overlap.width>1 and overlap.height>1:
                    issues.append({'kind':'text_overlap','a':a.get_text(),'b':b.get_text()})
    for ext in ('svg','pdf'):
        fig.savefig(OUT/f'Fig1_RAMEF_Revised.{ext}',facecolor='white')
    fig.savefig(OUT/'Fig1_RAMEF_Revised.png',dpi=600,facecolor='white')
    fig.savefig(OUT/'Fig1_RAMEF_Revised_preview.png',dpi=200,facecolor='white')
    plt.close(fig)
    assert all(sha(OUT/name)==value for name,value in data_hashes.items())
    report={
        'canvas_mm':[W,H],'embedded_width_mm':177.8,'minimum_text_pt':10.5,
        'font':'Times New Roman','source_training_PU':'centre crops',
        'source_training_MCC5':'distributed blocks',
        'filtered_statistics_tapped_before_envelope':True,
        'scaler_sample_weighted':False,'multinomial_LR_sample_weighted':True,
        'source_traces_changed':False,'source_data_sha256':data_hashes,
        'no_new_query_read':True,'no_model_fitting':True,'layout_issues':issues,
        'solid_arrows':'processing/data transfer',
        'dashed_arrow':'frozen calibration parameters',
        'all_text_pt':sorted(set(float(t.get_fontsize())for t,_ in alltexts)),
        'status':'PASS' if not issues else 'REVIEW',
        **panel_a_extra,
    }
    (OUT/'figure_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
