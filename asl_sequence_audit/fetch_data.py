"""Fetch a pinned candidate plus original metadata; never download the 14-GB archive.

No credentials or third-party executable code are used. All payloads are size
bounded and SHA-256 verified before acceptance. Existing files are also checked.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile
import urllib.request

REPO = "Kibalama/ChicagoFSWild-Landmarks"
REVISION = "c738cce88fcae8bcdb0811738c44a0b41701ad4d"
FILES = [
    ("train-00000-of-00001.parquet", 302818625, "4cb8e73f9a4a3889f001c55a5afc2b1948d0184bce5a607f62b8e42831730edc"),
    ("validation-00000-of-00001.parquet", 58906243, "396c849dd03d644924c8e696bf35e241d09b283ecb523131ee0d684137942533"),
    ("test-00000-of-00001.parquet", 52489504, "848a12d531a91a62f84615545204d31105897edaebfcfdc3542272b987bdddef"),
]
ARCHIVE_URL = "https://dl.ttic.edu/ChicagoFSWild.tgz"
PREFIX_SIZE = 1048576
PREFIX_SHA = "7c5f533098bb12eb4b3fb1e1eea384ce4a8ce752a56e33d281df567b808f4525"
METADATA_SHA = "ed0698ae6a606742030bb6130daddef115a2032414efc34a92000029ddd17e61"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as inp:
        for block in iter(lambda: inp.read(1024**2), b""):
            h.update(block)
    return h.hexdigest()


def download(url, dest, size, expected_sha, headers=None, prefix=False):
    if dest.exists():
        if dest.stat().st_size != size or digest(dest) != expected_sha:
            raise ValueError(f"Existing file differs from pinned source: {dest}")
        print(f"Verified cached {dest.name}", flush=True)
        return
    part = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers=headers or {})
    total = 0
    try:
        with urllib.request.urlopen(req, timeout=45) as response, part.open("wb") as out:
            # Prefix mode closes after exactly the allowed prefix, even if the
            # server ignores Range. It never follows the 14-GB payload to EOF.
            while total < size:
                chunk = response.read(min(1024**2, size - total))
                if not chunk:
                    break
                out.write(chunk)
                total += len(chunk)
            if not prefix and response.read(1):
                raise ValueError("Payload exceeds declared size")
        if total != size or digest(part) != expected_sha:
            raise ValueError(f"Size/hash mismatch for {dest.name}")
        part.replace(dest)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    print(f"Downloaded and verified {dest.name}: {size} bytes", flush=True)


def fetch_metadata(cache):
    source = cache / "source"
    source.mkdir(parents=True, exist_ok=True)
    prefix = source / "official_archive_prefix.bin"
    download(ARCHIVE_URL, prefix, PREFIX_SIZE, PREFIX_SHA,
             {"Range": f"bytes=0-{PREFIX_SIZE - 1}"}, prefix=True)
    wanted = {"ChicagoFSWild/README": "README", "ChicagoFSWild/ChicagoFSWild.csv": "ChicagoFSWild.csv"}
    found = set()
    with tarfile.open(fileobj=io.BytesIO(prefix.read_bytes()), mode="r|gz") as archive:
        for member in archive:
            if member.name in wanted:
                if not member.isfile() or member.size > 2 * 1024**2:
                    raise ValueError("Unexpected metadata member")
                content = archive.extractfile(member).read(member.size + 1)
                if len(content) != member.size:
                    raise ValueError("Truncated metadata member")
                destination = source / wanted[member.name]
                if destination.exists() and destination.read_bytes() != content:
                    raise ValueError("Existing metadata differs; refusing overwrite")
                destination.write_bytes(content)
                found.add(member.name)
            if found == set(wanted):
                break
            if member.size > 4 * 1024**2:
                raise ValueError("Metadata not found before large archive member")
    if found != set(wanted) or digest(source / "ChicagoFSWild.csv") != METADATA_SHA:
        raise ValueError("Official metadata verification failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("cache"))
    args = parser.parse_args()
    cache = args.cache_dir.resolve()
    (cache / "parquet").mkdir(parents=True, exist_ok=True)
    fetch_metadata(cache)
    manifest = []
    for name, size, sha in FILES:
        url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/data/{name}?download=true"
        download(url, cache / "parquet" / name, size, sha)
        manifest.append({"name": name, "url": url, "bytes": size, "sha256": sha, "revision": REVISION})
    (cache / "source/download_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Complete: 414,214,372 Parquet bytes + 1 MiB official archive prefix.")


if __name__ == "__main__":
    main()
