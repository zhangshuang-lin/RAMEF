# SQ_RAMEF_STRICT_V1.4 pre-training amendment

Parents: SQ protocol V1 and amendments V1.1–V1.3.

This amendment is frozen after the final P0 PASS and before neural training.

## Source-class balance

Every frozen outer fold contains 3/12/12 source REC for N/IR/OR. To prevent
this record-count imbalance from becoming a class prior:

1. SQ-native U2-Hz source training uses a class-balanced block sampler while
   retaining every source REC.
2. The source feature bank retains every source REC. Source sample weights are
   inverse-frequency balanced across the three classes and then normalised to
   mean one over source REC.
3. Each target-support REC retains the registered weight 10.
4. The same weights are used for Single, Local-M8, Distributed-M8, RAMEF-F and
   random controls.

No query count, feature or outcome informs this rule. All other V1–V1.3 clauses
remain unchanged.

