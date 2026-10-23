# Updating the public repository

Repository: https://github.com/zhangshuang-lin/RAMEF

This update was prepared from commit
`1c75f7eab840a1b7d48801a08c29beadb4bb974f`. Compare any subsequent remote changes
before uploading so that newer edits are not overwritten.

1. In the delivered archive, open `UPLOAD_TO_GITHUB` and upload its **contents**
   to the repository root using Add file → Upload files. Do not upload the ZIP
   as a source file or add an extra `UPLOAD_TO_GITHUB/` directory in the repo.
2. Include the dotfiles `.gitattributes` and `.gitignore` and retain all relative
   subdirectories. The separate `COMPLETE_REPOSITORY` is an alternative full
   checkout for local checking, not a second folder to upload alongside the patch.
3. Review the proposed changes, preferably on a new branch, before merging.
   No old files need deletion. Original `code/`, `configs/`, `protocols/`,
   `evidence/` and `reference/` files must retain their existing bytes.
4. Download a fresh ZIP/checkout from the updated GitHub repository and run:

```text
python scripts/verify_bundle.py
python -m unittest discover -s tests -v
python scripts/replay_paper_results.py --output-dir runs/after_upload01
```

The output directory must not already exist. This verifies the distribution and
replays stored class predictions; it is not raw-data training or original-model
inference.

After verification, create an actual release/tag and record that public identity
in the paper's Code availability. Do not fill a DOI until one is actually issued.
The source package does not contain `.git` or an invented release commit.

## Line endings

`.gitattributes` uses `* -text` to preserve exact bytes. If a previous Windows
checkout has already converted line endings, make a fresh checkout:

```text
git clone --config core.autocrlf=false https://github.com/zhangshuang-lin/RAMEF.git RAMEF-fresh
```

Do not fix verification by rewriting historical protocol/model hashes or running
bulk normalisation on scientific records. When intentionally changing mutable
documentation or code later, update the **current distribution** manifest for
that new version, not the historical experimental hashes.

## Rights

No software licence is assigned by this update. The rights holders must confirm
licensing and any third-party sharing conditions. See `LICENSE_STATUS.md` and
`THIRD_PARTY_NOTICES.md`. Raw waveform assets, model weights and fitted
classifiers have not been added.
