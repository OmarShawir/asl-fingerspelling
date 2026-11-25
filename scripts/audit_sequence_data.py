"""Index-only Kaggle audit. Never downloads Parquet; preserves every phrase verbatim."""

import argparse
import base64
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import string
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections import Counter, defaultdict
from io import BytesIO

COMPETITION = "asl-fingerspelling"
API = "https://www.kaggle.com/api/v1/competitions/data/"
LETTERS = string.ascii_lowercase


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def auth_headers():
    token = os.environ.get("KAGGLE_API_TOKEN")
    if token:
        return {"Authorization": "Bearer " + token}
    username, key = os.environ.get("KAGGLE_USERNAME"), os.environ.get("KAGGLE_KEY")
    config = Path(os.environ.get("KAGGLE_CONFIG_DIR", str(Path.home() / ".kaggle")))
    if not (username and key) and (config / "kaggle.json").is_file():
        credentials = json.loads((config / "kaggle.json").read_text())
        username, key = credentials.get("username"), credentials.get("key")
    if username and key:
        encoded = base64.b64encode(f"{username}:{key}".encode()).decode()
        return {"Authorization": "Basic " + encoded}
    if (config / "access_token").is_file():
        return {"Authorization": "Bearer " + (config / "access_token").read_text().strip()}
    return {}


def fetch(url):
    request = urllib.request.Request(url, headers=auth_headers())
    with urllib.request.urlopen(request, timeout=45) as response:
        data = response.read(64 * 1024**2 + 1)
        if len(data) > 64 * 1024**2:
            raise ValueError("Index-only response exceeds 64 MiB safety limit")
        return data


def download_indexes(directory):
    attempts = []
    listing = directory / "files.json"
    if not listing.exists():
        try:
            files, token, seen = [], None, set()
            while True:
                url = API + "list/" + COMPETITION
                if token:
                    url += "?" + urllib.parse.urlencode({"pageToken": token})
                page = json.loads(fetch(url))
                if isinstance(page, list):
                    files.extend(page)
                    break
                entries = page.get("files", page.get("competitionFiles"))
                if not isinstance(entries, list):
                    raise ValueError("Unknown file-list response schema")
                files.extend(entries)
                token = page.get("nextPageToken")
                if not token:
                    break
                if token in seen:
                    raise ValueError("Repeated pagination token")
                seen.add(token)
            dump(listing, files)
            attempts.append(dict(file="files.json", status="downloaded", count=len(files)))
        except (OSError, ValueError) as e:
            attempts.append(dict(file="files.json", status="blocked", error=str(e)))
    for name in ("train.csv", "character_to_prediction_index.json"):
        if (directory / name).exists():
            continue
        try:
            data = fetch(API + "download/" + COMPETITION + "/" + name)
            if zipfile.is_zipfile(BytesIO(data)):
                with zipfile.ZipFile(BytesIO(data)) as archive:
                    candidates = [n for n in archive.namelist() if Path(n).name == name]
                    if len(candidates) != 1:
                        raise ValueError("Unexpected index archive")
                    if archive.getinfo(candidates[0]).file_size > 64 * 1024**2:
                        raise ValueError("Expanded index too large")
                    data = archive.read(candidates[0])
            if name.endswith(".json"):
                obj = json.loads(data)
                if not isinstance(obj, dict) or not all(
                    len(k) == 1 and isinstance(v, int) for k, v in obj.items()
                ):
                    raise ValueError("Not a character mapping")
            elif "participant_id" not in data[:4096].decode("utf-8-sig"):
                raise ValueError("Not the expected CSV (possibly login HTML)")
            (directory / name).write_bytes(data)
            attempts.append(dict(file=name, status="downloaded", bytes=len(data)))
        except (OSError, ValueError, zipfile.BadZipFile) as e:
            attempts.append(dict(file=name, status="blocked", error=str(e)))
    return attempts


def summarize(rows):
    chars = Counter(c for row in rows for c in row["phrase"])
    lengths = Counter(len(row["phrase"]) for row in rows)
    folded = Counter()
    sequences = Counter()
    for row in rows:
        # Case folding is for reporting only. The target remains untouched.
        folded.update(c.lower() for c in row["phrase"] if c in string.ascii_letters)
        sequences.update(set(c.lower() for c in row["phrase"] if c in string.ascii_letters))
    return dict(
        sequences=len(rows),
        participants=len({r["participant_id"] for r in rows}),
        phrase_length_histogram=dict(sorted(lengths.items())),
        phrase_length_min=min(lengths, default=None),
        phrase_length_max=max(lengths, default=None),
        phrase_length_mean=sum(k * v for k, v in lengths.items()) / len(rows) if rows else None,
        exact_character_counts=dict(sorted(chars.items())),
        ascii_letter_coverage={
            c: dict(
                lowercase=chars[c],
                uppercase=chars[c.upper()],
                case_insensitive=folded[c],
                sequences=sequences[c],
            )
            for c in LETTERS
        },
        missing_letters=[c.upper() for c in LETTERS if not folded[c]],
        digits={c: chars[c] for c in string.digits if chars[c]},
        whitespace={c: n for c, n in chars.items() if c.isspace()},
        symbols={c: n for c, n in chars.items() if not c.isalnum() and not c.isspace()},
        non_ascii={c: n for c, n in chars.items() if not c.isascii()},
        empty_phrases=lengths[0],
    )


def file_sizes(raw):
    entries = raw if isinstance(raw, list) else raw.get("files", raw.get("competitionFiles", []))
    result = {}
    for item in entries:
        name = item.get("name", item.get("fileName"))
        size = item.get("totalBytes", item.get("size"))
        if isinstance(size, str) and size.isdigit():
            size = int(size)
        if name and isinstance(size, int) and size > 0:
            result[name] = size
    return result


def stable(value):
    return hashlib.sha256(("42:" + value).encode()).hexdigest()


def sample_plan(rows, sizes, budget):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["path"]].append(row)
    missing_sizes = sorted(set(grouped) - sizes.keys())
    if missing_sizes:
        return dict(
            status="blocked", reason="Exact byte sizes missing", files_missing_sizes=missing_sizes
        ), []
    selected_files, used = [], 0
    for name in sorted(grouped, key=lambda n: (sizes[n], n)):
        if len(selected_files) >= 3:
            break
        if used + sizes[name] <= budget:
            selected_files.append(name)
            used += sizes[name]
    available = [r for n in selected_files for r in grouped[n]]
    people = sorted({r["participant_id"] for r in available}, key=stable)[:10]
    if len(people) < 3:
        return dict(
            status="blocked",
            reason="Budget/files contain fewer than three participants",
            files=[dict(name=n, bytes=sizes[n]) for n in selected_files],
            download_bytes=used,
        ), []
    ntrain = min(len(people) - 2, max(1, int(len(people) * 0.7)))
    nval = max(1, int(len(people) * 0.15))
    splits = dict(
        train=people[:ntrain],
        validation=people[ntrain : ntrain + nval],
        test=people[ntrain + nval :],
    )
    selected = []
    for split, ids in splits.items():
        for person in ids:
            candidates = sorted(
                [r for r in available if r["participant_id"] == person],
                key=lambda r: stable(r["sequence_id"]),
            )
            selected.extend(dict(r, split=split) for r in candidates[:200])
    # Charge only the union of shards actually needed for selected sequences.
    needed = sorted({r["path"] for r in selected})
    return dict(
        status="proposed_index_only",
        files=[dict(name=n, bytes=sizes[n]) for n in needed],
        download_bytes=sum(sizes[n] for n in needed),
        budget_bytes=budget,
        participants=splits,
        summary=summarize(selected),
        per_split={s: summarize([r for r in selected if r["split"] == s]) for s in splits},
        policy="up to 3 smallest fitting shards; up to 10 hash-ordered participants; 70/15/remainder participant split with minimum 1 validation/test; max 200 hash-ordered sequences/person; seed 42; no text edits",
        limitations="Not optimized for coverage. Missing letters explicitly reported. Frame availability and CTC feasibility not checked without Parquet.",
    ), selected


def environment():
    memory = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        if key in ("MemTotal", "MemAvailable", "SwapTotal"):
            memory[key + "_bytes"] = int(value.split()[0]) * 1024
    gpu = dict(cuda_available=None)
    if importlib.util.find_spec("torch"):
        import torch

        gpu = dict(
            cuda_available=torch.cuda.is_available(),
            devices=torch.cuda.device_count(),
            torch_version=str(torch.__version__),
        )
    return dict(
        disk_free_bytes=shutil.disk_usage(".").free,
        memory=memory,
        gpu=gpu,
        kaggle_cli=shutil.which("kaggle"),
        packages={
            m: bool(importlib.util.find_spec(m))
            for m in ("kaggle", "kagglehub", "pandas", "pyarrow")
        },
        credentials_available=bool(auth_headers()),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true", help="Fetch only list and two index files")
    parser.add_argument("--data-dir", type=Path, default=Path("data/external/asl-fingerspelling"))
    parser.add_argument("--output", type=Path, default=Path("results/sequence_audit"))
    parser.add_argument("--budget-gib", type=float, default=2.0)
    args = parser.parse_args()
    if args.budget_gib <= 0:
        parser.error("Budget must be positive")
    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    dump(args.output / "environment.json", environment())
    if args.fetch:
        dump(args.output / "fetch_attempts.json", download_indexes(args.data_dir))
    required = ["train.csv", "character_to_prediction_index.json", "files.json"]
    missing = [n for n in required if not (args.data_dir / n).exists()]
    if missing:
        dump(
            args.output / "summary.json",
            dict(
                status="blocked_missing_indexes",
                missing=missing,
                sequences=None,
                participants=None,
                J=None,
                Z=None,
            ),
        )
        dump(
            args.output / "sample_plan.json",
            dict(
                status="blocked_missing_indexes",
                files=None,
                download_bytes=None,
                participants=None,
                budget_bytes=int(args.budget_gib * 1024**3),
                proposed_policy="up to 3 smallest shards / 10 participants / 200 sequences per person; split by person; cannot instantiate without indexes",
            ),
        )
        print("BLOCKED: missing", ", ".join(missing))
        return
    with (args.data_dir / "train.csv").open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    fields = {"path", "file_id", "participant_id", "sequence_id", "phrase"}
    if not rows or not fields <= rows[0].keys():
        raise ValueError("Missing required CSV columns or empty index")
    if len({r["sequence_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate sequence IDs")
    mapping = json.loads((args.data_dir / required[1]).read_text())
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("Duplicate character prediction indices")
    stats = summarize(rows)
    stats["status"] = "index_audited"
    stats["characters_absent_from_mapping"] = sorted(
        set(stats["exact_character_counts"]) - mapping.keys()
    )
    stats["character_mapping"] = mapping
    stats["inputs_sha256"] = {
        n: hashlib.sha256((args.data_dir / n).read_bytes()).hexdigest() for n in required
    }
    relations = defaultdict(Counter)
    for row in rows:
        relations[row["path"]][row["participant_id"]] += 1
    dump(args.output / "participant_files.json", {p: dict(c) for p, c in sorted(relations.items())})
    sizes = file_sizes(json.loads((args.data_dir / "files.json").read_text()))
    dump(args.output / "file_sizes.json", sizes)
    stats["parquet_file_count"] = len(relations)
    stats["files_with_multiple_participants"] = sum(len(c) > 1 for c in relations.values())
    dump(args.output / "summary.json", stats)
    plan, selected = sample_plan(rows, sizes, int(args.budget_gib * 1024**3))
    dump(args.output / "sample_plan.json", plan)
    with (args.output / "sample_sequences.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[*rows[0], "split"])
        writer.writeheader()
        writer.writerows(selected)
    print(stats["sequences"], "sequences;", stats["participants"], "participants;", plan["status"])


if __name__ == "__main__":
    main()
