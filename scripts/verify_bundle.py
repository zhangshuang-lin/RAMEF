"""Verify the distribution without importing experiment code or opening data."""
from pathlib import Path
import argparse
import ast
import csv
import hashlib
import json
import re


def digest(path):
    h=hashlib.sha256()
    with path.open('rb')as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def verify(root):
    root=root.resolve();manifest=root/'MANIFEST.sha256';count=0
    if not manifest.is_file():raise ValueError('Missing MANIFEST.sha256')
    for line in manifest.read_text(encoding='utf-8').splitlines():
        expected,relative=line.split(' *',1)
        path=(root/relative).resolve()
        if root not in path.parents or not path.is_file():raise ValueError('Missing/unsafe manifest file: '+relative)
        if digest(path)!=expected:raise ValueError('SHA-256 mismatch: '+relative)
        count+=1
    forbidden={'.mat','.pt','.pth','.npz','.npy','.joblib','.pkl','.pickle','.docx','.pdf','.7z','.zip'}
    bad=[p.relative_to(root).as_posix()for p in root.rglob('*')if p.is_file()and p.suffix.lower()in forbidden]
    if bad:raise ValueError('Non-code/private/heavy artifacts found: '+', '.join(bad))
    code=list((root/'code').glob('*.py'));names={p.stem for p in code};missing=[]
    for path in code:
        tree=ast.parse(path.read_text(encoding='utf-8-sig'),filename=str(path))
        for node in ast.walk(tree):
            modules=[a.name for a in node.names]if isinstance(node,ast.Import)else[node.module]if isinstance(node,ast.ImportFrom)and node.module else[]
            for module in modules:
                if module.split('.')[0]in names and not(root/'code'/(module.split('.')[0]+'.py')).is_file():missing.append(module)
    with(root/'docs/dependency_graph.json').open(encoding='utf-8')as stream:graph=json.load(stream)
    for module,dependencies in graph['local_imports'].items():
        for dependency in dependencies:
            if dependency not in names:missing.append(f'{module} -> {dependency}')
    if missing:raise ValueError('Missing local modules: '+', '.join(missing))
    parsed=0
    patterns=[r'gh[pousr]_[A-Za-z0-9]{25,}',r'sk-[A-Za-z0-9]{25,}',
              r'C:[\\/]Users[\\/](?:Charlotte|dev-admin)']
    for path in root.rglob('*.py'):
        source=path.read_text(encoding='utf-8-sig')
        ast.parse(source,filename=str(path));parsed+=1
        if any(re.search(pattern,source)for pattern in patterns):
            raise ValueError('Potential credential or personal path: '+path.relative_to(root).as_posix())
    report={'status':'PASS','manifest_files':count,'code_modules':len(code),
            'python_sources_parsed':parsed,'high_confidence_privacy_scan_hits':0,
            'forbidden_binary_artifacts':bad,'imports_executed':False,
            'models_loaded':False,'target_inference_performed':False,
            'scope':'File integrity and static source checks, not scientific or external-registration certification'}
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root',type=Path,default=Path(__file__).resolve().parents[1])
    args=parser.parse_args()
    try:print(json.dumps(verify(args.repo_root),indent=2))
    except(ValueError,OSError)as error:parser.exit(2,str(error)+'\n')
