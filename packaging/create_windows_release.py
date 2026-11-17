"""Create a portable Windows ZIP and a SHA-256 checksum for a built folder."""
from pathlib import Path
import hashlib
import sys
import zipfile

root = Path(sys.argv[1]).resolve()
folder = root / 'TomatoTracker-0.7.36'
target = root / 'TomatoTracker-0.7.36-Windows-x64.zip'
if not (folder / 'TomatoTracker-0.7.36.exe').is_file():
    raise SystemExit('The built executable was not found')
with zipfile.ZipFile(target, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=4, allowZip64=True) as archive:
    for path in sorted(folder.rglob('*')):
        if path.is_file():
            archive.write(path, path.relative_to(root).as_posix())
with zipfile.ZipFile(target) as archive:
    bad = archive.testzip()
    if bad:
        raise SystemExit(f'ZIP integrity check failed: {bad}')
with target.open('rb') as stream:
    checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
(root / 'SHA256SUMS.txt').write_text(f'{checksum}  {target.name}\n', encoding='ascii')
print(f'{target.name}: {checksum}')
