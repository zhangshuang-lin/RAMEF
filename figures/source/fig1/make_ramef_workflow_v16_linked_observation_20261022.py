"""Connect the source trace, registered blocks and raw detail in Figure 1a."""
from __future__ import annotations

import json
from pathlib import Path

import make_ramef_workflow_v15_corrected_20261022 as base
from matplotlib.patches import Rectangle

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'paper/figures/Fig1_RAMEF_Linked_Observation_20261022'


def linked_panel(fig,ax,overview,local):
    provenance=json.loads((base.DATA/'clean_axes_audit.json').read_text(encoding='utf-8'))['source_example']
    fs=provenance['sample_rate_hz']
    starts=provenance['block_starts']
    length=provenance['block_length']
    assert len(starts)==8
    assert int(local[0]['original_sample_index'])==starts[3]
    assert len(local)==640
    assert [int(row['original_sample_index'])for row in local]==list(range(starts[3],starts[3]+640))
    base.text(ax,4,160,'a   Distributed observation: 8 non-overlapping blocks',
              size=12.2,bold=True,ha='left')
    base.text(ax,14,154,'Source record',ha='left')
    base.text(ax,108,154,'Block 4: first 10 ms',ha='left',color=base.WARM)
    full=base.inset(fig,14,137,60,14)
    full.vlines([float(r['time_bin_centre_s'])for r in overview],
                [float(r['minimum'])for r in overview],
                [float(r['maximum'])for r in overview],color=base.INK,lw=.28)
    full.set_xlim(0,4);full.set_ylim(-2,2)
    full.set_xticks([0,2,4]);full.set_yticks([-2,0,2])
    start4=starts[3]/fs
    full.add_patch(Rectangle((start4,-2),length/fs,4,fc='none',ec=base.WARM,
                            lw=.9,linestyle=(0,(2.5,1.5)),zorder=4))
    detail=base.inset(fig,108,137,62,14)
    detail.plot([float(r['elapsed_ms'])for r in local],
                [float(r['recorded_amplitude'])for r in local],color=base.INK,lw=.45)
    detail.lines[0].get_path().should_simplify=False
    detail.set_xlim(0,10);detail.set_ylim(-.8,.8)
    detail.set_xticks([0,5,10]);detail.set_yticks([-.5,0,.5],labels=['−0.5','0','0.5'])
    base.text(ax,44,130.5,'Time (s)')
    base.text(ax,139,129,'Time (ms)')
    base.text(ax,4,144,'Amplitude',rotation=90)
    base.text(ax,98,144,'Amplitude',rotation=90)

    # This strip uses the same temporal coordinate map as the full-record plot.
    group=Rectangle((13,121.8),62,6.5,fc='white',ec=base.BORDER,lw=.6,zorder=1)
    ax.add_patch(group)
    coordinates=[]
    for i,start in enumerate(starts):
        x=14+60*(start/fs)/4
        width=60*(length/fs)/4
        selected=i==3
        r=Rectangle((x,122.8),width,4.5,ec=base.WARM if selected else base.GREY,
                    fc=base.WARM_BG if selected else 'white',lw=.9 if selected else .65,zorder=2)
        ax.add_patch(r)
        base.text(ax,x+width/2,125.05,str(i+1),color=base.WARM if selected else base.INK)
        coordinates.append([x,width])
    # A source-record-to-group arrow makes the extraction operation explicit.
    base.arrow(ax,[(74.5,145),(82,145),(82,125.05),(75,125.05)])
    base.text(ax,87,136,'Extract',rotation=90,color=base.GREY)
    # The detail comes from block 4; the other blocks still feed the model.
    midpoint=coordinates[3][0]+coordinates[3][1]/2
    base.arrow(ax,[(midpoint,122.8),(midpoint,120),(174.5,120),(174.5,144),(170,144)],color=base.WARM)
    base.text(ax,139,124,'All 8 blocks → both paths in b',color=base.GREY)
    ax.plot([4,174],[118,118],color='#d9dfe3',lw=.65)
    return full,detail,{
        'source_record_to_eight_block_group_connected':True,
        'strip_positions_match_source_time_axis':True,
        'block_4_matched_outline_and_detail_connector':True,
        'eight_blocks_feed_both_paths_not_only_block_4':True,
        'detail_is_illustrative_not_model_input':True,
        'selected_block_1_based':4,
        'selected_block_start_sample':starts[3],
        'selected_block_start_s':start4,
        'selected_block_length_samples':length,
        'detail_raw_samples':len(local),
    }


def main():
    base.OUT=OUT
    base.main(panel_a_builder=linked_panel,
              panel_b_title='b   Frozen dual-path extraction of the same 8 blocks')


if __name__=='__main__':
    main()
