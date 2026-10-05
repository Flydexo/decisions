"""Share APFS blocks of nearly identical checkpoints, preserving every byte."""
from pathlib import Path
import hashlib
import os
import subprocess
import numpy as np


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def share(source, target):
    if source.stat().st_size != target.stat().st_size:
        return False
    changes = []
    offset = 0
    with source.open('rb') as left, target.open('rb') as right:
        while block := left.read(4 * 2**20):
            wanted = right.read(len(block))
            differences = np.flatnonzero(np.frombuffer(block, dtype='u1') != np.frombuffer(wanted, dtype='u1'))
            if len(differences) > 4096:
                return False  # Different weights: no worthwhile sharing.
            if len(differences):
                breaks = np.flatnonzero(np.diff(differences) > 1) + 1
                for group in np.split(differences, breaks):
                    start, end = int(group[0]), int(group[-1]) + 1
                    changes.append((offset + start, wanted[start:end]))
            offset += len(block)
    expected = digest(target)
    source_expected = digest(source)
    temporary = target.with_suffix('.shared.tmp')
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        subprocess.run(['cp', '-c', str(source), str(temporary)], check=True)
        with temporary.open('r+b') as output:
            for offset, block in changes:
                output.seek(offset)
                output.write(block)
        if digest(temporary) != expected or digest(source) != source_expected:
            raise RuntimeError('Checkpoint byte-preservation check failed')
        os.replace(temporary, target)
        assert digest(target) == expected
        print(f'{target}: shared storage; original SHA-256 preserved ({expected})', flush=True)
        return True
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == '__main__':
    for name in ['pilot', 'pilot_corrected', 'pilot_cross_entropy',
                 'pilot_streaming', 'pilot_streaming_cross_entropy']:
        root = Path(__file__).resolve().parents[1] / 'outputs' / name
        if (root / 'last.pt').exists() and (root / 'best.pt').exists():
            share(root / 'last.pt', root / 'best.pt')
