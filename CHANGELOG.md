# Changes

## 2026.10.23-readme-final

- Rewrote the English homepage and Chinese README around method, quick start, tasks/results, data-dependent execution and file access.
- Repaired the missing .gitattributes and outdated .gitignore detected in the uploaded snapshot e7052d5.
- Reconciled current distribution checksums with exact delivered bytes; scientific records and historical experiment hashes remain unchanged.
- Public uploaded snapshot tests and prediction replay passed; its distribution verification had failed before these repairs. New local checks apply to this package, not to a future public commit.


## 2026.10.23-submission-sync — locally prepared update

Based on the public `main` snapshot `1c75f7eab840a1b7d48801a08c29beadb4bb974f`.
This package version is not a claim that a new public tag or archival DOI exists.

- Unified reader-facing documentation under RAMEF. The historical RAMEF-F alias,
  original source filenames and prediction fields remain intact.
- Preserved the parent snapshot's model/experiment code, configurations,
  protocols, prediction CSVs and reference evidence byte-for-byte. No new model
  fitting, raw waveform inference or result selection was performed.
- Updated manuscript/supplement bindings and the real repository URL. The current
  manuscript title is retained; suggested alternative titles were not adopted.
- Added the exact current Figure 1 v16/v15 source chain and its non-waveform
  provenance metadata. Raw source waveform CSVs remain external prerequisites.
- Added the exact final manuscript SVG artwork, including Fig. S1 and the
  graphical abstract, with checksums and source mapping.
- Added `.gitattributes` to preserve stored bytes across Windows and Unix
  checkouts. Historical protocols and evaluation/model hashes are not
  renormalised or rewritten.
- Strengthened the distribution verifier's manifest coverage and path checks,
  and added synthetic packaging tests. Numerical replay/metric code is unchanged.
- Added an unpublished-manuscript citation, upload instructions and explicit
  local-version metadata. No licence, copyright ownership, publication DOI or
  software-author role was invented.

Executed checks and runtime details are recorded in `docs/LOCAL_PACKAGE_CHECKS.json`.
Remote upload, a remote fresh-clone check and release creation occur afterwards.
