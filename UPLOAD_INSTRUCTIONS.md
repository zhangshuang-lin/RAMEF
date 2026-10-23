# Final repository update

This package is based on public commit e7052d54a1f71b9822ebc92968617bfff83abdf0.
Upload the contents of UPLOAD_TO_GITHUB to the repository root, preserving paths.
Do not upload its outer directory, the ZIP, COMPLETE_REPOSITORY, or local audit files.
No deletion of original repository files is required.

The patch includes final English/Chinese README files, version/provenance metadata,
an updated distribution manifest, and the two dotfiles missing or outdated in the
audited upload. Ensure .gitattributes and .gitignore are included; do not assume a
browser folder drag includes hidden files. Their exact paths start with a dot at
the repository root, not a nested upload folder. Use the files provided rather
than retyping them, since the distribution verifies exact bytes.

After upload, download a fresh copy and run:

```text
python scripts/verify_bundle.py
python -m unittest discover -s tests -v
python scripts/replay_paper_results.py --output-dir runs/final_public_check01
```

The output directory must not exist. If verification fails, inspect the named
file; do not normalise scientific records or rewrite historical hashes. A Git
client can preserve exact bytes with the supplied .gitattributes; use a fresh
checkout if an older checkout already converted line endings. These local files
do not create a public release/tag/DOI or a software licence. Check remote changes
after the parent commit before replacing files. Do not upload raw recordings,
feature caches, model weights, fitted classifiers or manuscript PDFs.
