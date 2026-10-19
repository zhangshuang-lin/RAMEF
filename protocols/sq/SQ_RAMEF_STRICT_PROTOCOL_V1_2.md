# SQ_RAMEF_STRICT_V1.2 pre-training amendment

Parents: `SQ_RAMEF_STRICT_PROTOCOL.md` and
`SQ_RAMEF_STRICT_PROTOCOL_V1_1.md`.

This amendment is frozen after the V1.1 signal-only P0 failed and before any
feature learning or classification.

## Signal-only finding

The three NC records reserved for the source role reach approximately
2367–2374 rpm. They provide enough high-speed acceleration/deceleration data,
but the V1.1 steady-speed slope constraint leaves fewer than eight registered
source blocks. Reusing target-support REC would violate the strict role
isolation and is prohibited.

## Override

- Keep the source speed interval at 2000–3400 rpm.
- Remove the source slope constraint; high-speed acceleration, plateau and
  deceleration samples are all eligible source observations.
- Keep the target interval at 300–2000 rpm, so source and target speeds remain
  non-overlapping.
- Keep six outer folds, five support REC per class and complete REC isolation.

All other V1/V1.1 clauses and positive-result gates remain unchanged.

