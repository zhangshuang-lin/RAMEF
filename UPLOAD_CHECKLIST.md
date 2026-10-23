# Author upload checklist

Repository: https://github.com/zhangshuang-lin/RAMEF

## Already preserved by package construction

- The parent public snapshot is recorded in `docs/MANUSCRIPT_VERSION.json`.
- Historical experiment/model source, configurations, protocols, predictions and
  reference evidence are preserved byte-for-byte. Internal blocks are not shots.
- PU reporting uses 171 distinct IDs; historical 240-entry ledgers remain intact.
- MCC5 has 27 query runs per seed, not five independent cohorts.
- Post-target controls do not replace the original 96.30% MCC5 result.
- New documentation uses RAMEF, with the RAMEF-F historical alias explained.
- Current final SVGs and Fig. 1's v16/v15 source chain are included. Waveform
  CSVs, checkpoints, fitted classifiers and manuscript PDFs are excluded.

## Before uploading

- [ ] Confirm contributor permission and code/data redistribution conditions.
- [ ] Resolve software licensing with the rights holders; this update does not
      select a licence. See `LICENSE_STATUS.md` and `THIRD_PARTY_NOTICES.md`.
- [ ] Compare remote changes made after the recorded parent commit.
- [ ] Inspect `docs/LOCAL_PACKAGE_CHECKS.json` and upload only the prepared patch
      contents, retaining relative directories and dotfiles.

## After uploading

- [ ] Download a fresh public copy and run the verifier, tests and prediction
      replay described in `UPLOAD_INSTRUCTIONS.md`.
- [ ] Record the actual public commit and, after checks, create a release/tag.
- [ ] Cite that actual version in the submission record; do not invent a DOI.
- [ ] If archiving externally, use the exact tested version and wait for the
      real identifier before citing it.
- [ ] Keep local complete-model archives distinct from publicly downloadable
      files. No raw-data end-to-end reproduction is certified by metric replay.

中文核对：技术整理和本地验证不等于线上发布或授权确认。本轮未选择许可证，
未分配DOI，也未在GitHub推送、建tag或创建release。
