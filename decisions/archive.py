"""Bounded HTTP ranges for large published ZIP datasets, with no full-download fallback."""
from __future__ import annotations

import io
import json
import zipfile

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class RangeReader(io.RawIOBase):
    def __init__(self, url, size, max_transfer_bytes=128 * 2**20, block_size=256 * 2**10):
        self.url, self.size = url, size
        self.limit, self.block_size = max_transfer_bytes, block_size
        self.position, self.transferred = 0, 0
        self.block_start, self.block = -1, b""
        self.session = requests.Session()
        retries = Retry(total=4, connect=4, read=4, status=4, backoff_factor=2,
                        backoff_max=30, status_forcelist=[429, 500, 502, 503, 504],
                        allowed_methods=["GET"], respect_retry_after_header=False)
        self.session.mount("https://", HTTPAdapter(max_retries=retries))

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        position = offset if whence == io.SEEK_SET else self.position + offset if whence == io.SEEK_CUR else self.size + offset
        if position < 0:
            raise ValueError("Negative seek")
        self.position = position
        return position

    def read(self, size=-1):
        # zipfile requests small fixed headers/blocks. Never service an unbounded read.
        if size < 0:
            size = self.size - self.position
        if size > self.block_size:
            raise ValueError("Unbounded/large remote ZIP reads are forbidden")
        size = min(size, max(0, self.size - self.position))
        chunks = []
        while size:
            if not self.block_start <= self.position < self.block_start + len(self.block):
                start = self.position // self.block_size * self.block_size
                end = min(self.size, start + self.block_size) - 1
                requested = end - start + 1
                if self.transferred + requested > self.limit:
                    raise RuntimeError(f"Remote archive transfer limit reached ({self.limit} bytes); "
                                       "increase source.archive.max_transfer_bytes for a longer streaming run")
                with self.session.get(self.url, headers={"Range": f"bytes={start}-{end}",
                                                       "Accept-Encoding": "identity"},
                                      stream=True, timeout=60) as response:
                    expected = f"bytes {start}-{end}/{self.size}"
                    if response.status_code != 206 or response.headers.get("Content-Range") != expected:
                        raise RuntimeError("Archive server did not honor the exact byte range; full download is forbidden")
                    response.raw.decode_content = False
                    block = response.raw.read(requested + 1)
                    if len(block) != requested:
                        raise RuntimeError("Invalid HTTP byte range length")
                    self.transferred += len(block)
                    self.block_start, self.block = start, block
            offset = self.position - self.block_start
            count = min(size, len(self.block) - offset)
            chunks.append(self.block[offset:offset + count])
            size -= count
            self.position += count
        return b"".join(chunks)

    def close(self):
        self.session.close()
        super().close()


def archive_rows(archive: dict, split: str):
    with RangeReader(archive["url"], archive["size"], archive["max_transfer_bytes"],
                     block_size=archive.get("block_size_bytes", 256 * 2**10)) as remote:
        with zipfile.ZipFile(remote) as zipped:
            for member in archive["members"][split]:
                with zipped.open(member) as handle:
                    for line in handle:
                        if line.strip():
                            yield json.loads(line)
