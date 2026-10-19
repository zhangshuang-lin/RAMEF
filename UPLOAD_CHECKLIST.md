# Author upload checklist

This checklist concerns the prepared code bundle, not an instruction to upload the full working project.

## Before creating the public repository

- [ ] Confirm author/repository-owner identity and all contributors' permission to publish the included code.
- [ ] Resolve `LICENSE_STATUS.md`; choose an appropriate licence only after checking third-party provenance. Do not describe unlicensed code as permissively licensed.
- [ ] Complete `THIRD_PARTY_NOTICES.md` where source provenance or permissions remain unverified.
- [ ] Check dataset-provider terms for derived prediction/source-data redistribution. The bundle excludes third-party waveforms and feature caches.
- [ ] Review the file list: no `.pt`, `.pth`, `.joblib`, `.pkl`, feature `.npz`, raw `.mat`/waveform CSVs, manuscript PDFs, reference PDFs, credentials, private configuration, or local build paths should be uploaded inadvertently.
- [ ] Ensure source comments and environment notes do not expose personal contact details or machine-specific credentials. Measurement IDs, bearing codes, dataset-relative paths and SHA-256 values are not authentication secrets.

## Local package checks

```text
python scripts/verify_bundle.py
python -m unittest discover -s tests -v
python scripts/replay_paper_results.py --output-dir runs/pre_upload_replay
```

- [ ] All bundle hashes match the manifest and no forbidden/oversized files are present.
- [ ] All declared in-project Python dependencies are packaged or explicitly documented.
- [ ] Prediction replay reproduces the reference values without requiring data downloads or ML dependencies.
- [ ] PU is scored on 171 unique measurement IDs; the historical 240-entry ledger is not presented as 240 independent measurements.
- [ ] MCC5 metrics use 27 unique query runs per seed, with the same five seeds; five seeds are not counted as five independent cohorts.
- [ ] The original 63-coordinate MCC5 result is not replaced by a later compact-fusion/factorial result.
- [ ] README correctly distinguishes runnable prediction replay, data-dependent training, historical scripts with cache prerequisites, and archived figure source.
- [ ] Any training or raw-data workflow not actually exercised is labelled as unverified rather than claimed to have passed.

## Publish and link from the paper

- [ ] Create the author-approved GitHub repository and upload **only this prepared bundle**; do not upload the project root recursively.
- [ ] Run the verification/replay checks from a clean clone and record the tested Python/runtime details.
- [ ] Select a release tag and create an immutable release. Record its commit ID and SHA-256 manifest.
- [ ] If using an archival service, deposit that exact release and wait for a real assigned DOI. A planned DOI is not an existing identifier.
- [ ] Insert the actual repository URL and, once issued, the version-specific DOI into the manuscript code-availability section.
- [ ] Check that public links work without a private account, and that the repository describes the same method/version as the manuscript.
- [ ] Keep original experimental locks and historical archive manifests intact; portable copies must retain their provenance and adaptation log.

## 中文核对

当前待作者确认：公开账户/作者身份、代码与数值数据授权、GitHub地址、正式发布标签、归档DOI。整理、打包和本地校验均不等于已经上传；仓库链接真实可访问后，才能在论文中写为公开可用。
