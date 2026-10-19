# Script index

`code/` contains the complete static local-import closure of the selected experiment entrypoints. Historical names remain unchanged. Only the three scripts under `scripts/` documented as public launchers have portable paths; the other experiment runners retain their original stage-layout prerequisites.

| Entry | Role | Execution status |
|---|---|---|
| scripts/verify_bundle.py | SHA-256 and static checks | Standard library; read-only |
| scripts/replay_paper_results.py | Recompute all packaged prediction groups | Standard library; new output only |
| scripts/mcc5_pipeline.py | prepare / train / evaluate | Explicit raw-data and fresh work paths; ML dependencies |
| code/u2_stft_model.py | Adopted U2-Hz architecture and inactive historical branches | Implementation module |
| code/racp_or.py | PU eight-block coordinates and fixed 63D physical descriptor | Implementation module |
| code/data_loader_pu.py | Official PU vibration-channel loading | Implementation module |
| code/mcc5_confirmatory_data.py | MCC5 role-aware data preparation and fixed physical63 | Use scripts/mcc5_pipeline.py |
| code/run_mcc5_ramef_confirmatory.py | Original MCC5 train/evaluate runner; use portable launcher | Use scripts/mcc5_pipeline.py |
| code/run_ramef_fiveseed_rerun.py | PU source-freeze/evaluate five-seed archive; stage prerequisites | Historical stage-layout runner; inspect prerequisites |
| code/prepare_racp_measurement_final.py | Historical PU frozen calibration preparation | Historical stage-layout runner; inspect prerequisites |
| code/run_racp_measurement_final_once.py | Historical PU 240-entry evaluation; not a default replay command | Historical stage-layout runner; inspect prerequisites |
| code/run_pu_eightblock_fair_baselines.py | PU equal-input encoder implementations | Historical stage-layout runner; inspect prerequisites |
| code/run_pu_multiseed_stability.py | Earlier PU initialisation audit | Historical stage-layout runner; inspect prerequisites |
| code/run_support_calibration_direct_ablation.py | Source/support-only and prototype head attribution | Historical stage-layout runner; inspect prerequisites |
| code/run_racp_linear_support_stability.py | Historical support-deletion fit; see exact comparator in docs | Historical stage-layout runner; inspect prerequisites |
| code/run_u2hz_freeze_support_audit.py | Query-free frozen versus neural-adaptation audit | Historical stage-layout runner; inspect prerequisites |
| code/run_ramef_design_sensitivity.py | Query-free classifier and weight audit | Historical stage-layout runner; inspect prerequisites |
| code/run_ramef_threshold_sensitivity.py | Query-free occurrence-statistic audit | Historical stage-layout runner; inspect prerequisites |
| code/audit_mcc5_full_dataset.py | MCC5 structural and keyphase inventory | Historical stage-layout runner; inspect prerequisites |
| code/prepare_mcc5_localisation_protocol.py | MCC5 run-role manifest construction | Historical stage-layout runner; inspect prerequisites |
| code/run_mcc5_t1_support_stability.py | MCC5 registered support-neighbourhood audit | Historical stage-layout runner; inspect prerequisites |
| code/run_mcc5_t2_fair_benchmark.py | MCC5 WDCNN/ResNet/STFT/DANN/ProtoNet comparison | Historical stage-layout runner; inspect prerequisites |
| code/run_mcc5_t3_t5_analysis.py | MCC5 paired run errors, pooled bootstrap and physical context | Historical stage-layout runner; inspect prerequisites |
| code/run_mcc5_capacity_dimension_p0.py | MCC5 source-only dimension/capacity preparation | Historical stage-layout runner; inspect prerequisites |
| code/run_mcc5_carrier_dimension_equivalence_p0b.py | MCC5 repeated-coordinate equivalence audit | Historical stage-layout runner; inspect prerequisites |
| code/prepare_mcc5_posthoc_fair_comparison.py | Prepare frozen post-target head comparisons | Historical stage-layout runner; inspect prerequisites |
| code/run_mcc5_posthoc_fair_comparison_once.py | Historical separately locked dimension/capacity scoring | Historical stage-layout runner; inspect prerequisites |
| code/run_mcc5_encoder_physics_factorial.py | Historical post-target same-physics 2x2 | Historical stage-layout runner; inspect prerequisites |
| code/audit_mcc5_label_semantics.py | Pure/compound and matched-nuisance descriptions | Historical stage-layout runner; inspect prerequisites |
| code/audit_mcc5_physical_only.py | MCC5 physical-only descriptive control | Historical stage-layout runner; inspect prerequisites |
| code/run_sq_ramef_carrier.py | SQ native carrier ceiling task | Historical stage-layout runner; inspect prerequisites |
| code/prepare_sq_ramef_p0.py | SQ run split and speed-region preparation | Historical stage-layout runner; inspect prerequisites |
| code/run_cwru_ramef_cross_load.py | CWRU ten-class native cross-load ceiling task | Historical stage-layout runner; inspect prerequisites |
| code/create_cwru_ramef_bag_splits.py | CWRU parent-record-aware bag split | Historical stage-layout runner; inspect prerequisites |

## Dependency modules

angular_cycle_topology, audit_mcc5_ramef_p0, audit_x0b_phase_topology, bottleneck_deconvolution, capacity_matched_baselines, coordinate_carrier_order, fair_baseline_models, geometry_static_evidence, mc_sbt, prepare_splits, racp_ablation, run_a1_u2hz_confirmation, run_c2of, run_core_component_ablation, run_cwru_10class_baseline, run_cwru_native_a1_multiseed, run_cwru_native_u2, run_fair_baseline_pu, run_lcga_fresh_p0, run_lcga_paper_main, run_mc_sbt_p0, run_paper_main, run_racp_or_information_p0, run_tser_hb_support_p0, sq_data, tser_hb

## Important distinctions

- PU training uses source centre crops; eight distributed blocks are aggregated after frozen encoding. Internal blocks are not target shots.
- The later PU five-seed event uses 20260911-20260915. Its original 240-entry metrics must be collapsed to 171 IDs before comparing with the manuscript.
- MCC5 trains on 42 runs and chooses checkpoints on 42 validation runs; its original final calibration uses 42 source-training plus 15 supports.
- DANN uses source-domain labels only; ProtoNet is episodic WDCNN, not an ordinary prototype head attached to U2-Hz.
- The support comparison in the paper is DM8 versus full fusion; not every historical support script has those same comparators.
- File-address portability in code/mcc5_confirmatory_data.py changes only disk lookup; original run IDs and their SHA-derived random controls are unchanged.
- Original checkpoint/target-event hashes describe the original archives. Adapted/new computational runs receive new records; they do not replace those events.
