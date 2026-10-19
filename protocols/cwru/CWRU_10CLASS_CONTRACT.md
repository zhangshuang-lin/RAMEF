# CWRU ten-class data contract

Frozen taxonomy: 2026-08-17

## Canonical local subset

- Local path is `D:/CRWU` (directory spelling retained); dataset name is CWRU.
- Use only **12 kHz drive-end bearing fault records** and the drive-end (`DE`)
  accelerometer channel.
- Use healthy plus 0.007, 0.014 and 0.021 inch EDM faults at the inner race,
  ball and outer race.
- For outer-race faults use the **6 o'clock, load-zone-centred** records only.
- Exclude 0.028/0.040 inch faults, 48 kHz records, fan-end fault datasets,
  outer-race 3/12 o'clock variants, and FE/BA channels from the main task.

## Label map

| Label | Class | Physical meaning |
|---:|---|---|
| 0 | N | Healthy |
| 1 | IR007 | Inner race, 0.007 in |
| 2 | IR014 | Inner race, 0.014 in |
| 3 | IR021 | Inner race, 0.021 in |
| 4 | B007 | Ball, 0.007 in |
| 5 | B014 | Ball, 0.014 in |
| 6 | B021 | Ball, 0.021 in |
| 7 | OR007 | Outer race at 6:00, 0.007 in |
| 8 | OR014 | Outer race at 6:00, 0.014 in |
| 9 | OR021 | Outer race at 6:00, 0.021 in |

The ordering is hierarchical and fixed: healthy, then location IR/B/OR, then
increasing diameter. It does not imply ordinal cross-entropy targets.

## Operating-condition domains

| Domain | Load | Official approximate speed |
|---|---:|---:|
| L0 | 0 hp | 1797 rpm |
| L1 | 1 hp | 1772 rpm |
| L2 | 2 hp | 1750 rpm |
| L3 | 3 hp | 1730 rpm |

The canonical grid contains exactly 40 records: ten classes x four loads. Four
leave-one-load-out tasks will later evaluate `L{others} -> L{target}`.

## Scientific boundary

CWRU faults are single-point EDM defects, whereas PU includes real damage; the
datasets validate different aspects of the method. In CWRU, the four load
records for a fault type/diameter correspond to the same seeded fault bearing.
Consequently, leave-one-load-out is a valid **cross-condition** task but not an
unseen-bearing task.

There is only one target recording per class/load. A later “5-shot” CWRU
protocol must therefore mean five labelled, non-overlapping **windows** from
that recording, never five independent measurements. Support/query windows
must occupy separated temporal blocks and the paper must state this limitation.

## Local-data corrections

`normal_2.mat` contains both duplicated `X098` variables and the required
`X099` record. The contract explicitly selects `X099_DE_time`. Selected normal
records for loads 1 and 2 lack matching RPM variables; their official nominal
1772/1750 rpm metadata are used and marked as fallbacks in the manifest.
