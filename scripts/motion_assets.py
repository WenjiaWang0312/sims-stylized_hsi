import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import tarfile

import yaml


def relative_file(root, filename):
    path = (root / filename).resolve()
    try:
        relative = path.relative_to(root)
    except ValueError:
        raise ValueError("Asset escapes the data root: {}".format(filename))
    return path, relative.as_posix()


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def inventory(root, manifests, with_hashes):
    files = {}
    datasets = []
    for filename in manifests:
        path, relative = relative_file(root, filename)
        data = yaml.safe_load(path.read_text())
        motions = data["motions"]
        styles = Counter()
        caption_count = 0
        references = [path]
        for skill, entries in motions.items():
            if not isinstance(entries, list):
                raise ValueError("Expected a list of motions for {}".format(skill))
            for entry in entries:
                styles[entry.get("emotion/style", "unspecified")] += 1
                captions = entry.get("captions", [])
                if not isinstance(captions, list) or not captions:
                    raise ValueError("Missing caption list in {}".format(relative))
                caption_count += len(captions)
                for key in ("file", "clip_file"):
                    if not entry.get(key):
                        raise ValueError("Missing {} in {}".format(key, relative))
                    references.append(path.parent / entry[key])
                for key in ("object_file", "obj_file"):
                    if entry.get(key):
                        references.append(path.parent / entry[key])
        for reference in references:
            asset, asset_name = relative_file(root, reference)
            if asset_name in files:
                continue
            record = {"path": asset_name, "exists": asset.is_file()}
            if record["exists"]:
                record["bytes"] = asset.stat().st_size
                if with_hashes:
                    record["sha256"] = digest(asset)
            files[asset_name] = record
        datasets.append({
            "manifest": relative, "skills": {skill: len(entries) for skill, entries in motions.items()},
            "captions": caption_count, "styles": dict(styles), "redistribution_status": "unreviewed",
        })
    records = sorted(files.values(), key=lambda item: item["path"])
    return {
        "format_version": 1, "datasets": datasets, "files": records,
        "total_bytes": sum(record.get("bytes", 0) for record in records),
        "missing": [record["path"] for record in records if not record["exists"]],
    }


def pack(root, report, review, output):
    if output.exists():
        raise FileExistsError("Refusing to overwrite {}".format(output))
    if not report.get("datasets") or not report.get("files") or report.get("missing"):
        raise ValueError("Inventory must contain datasets, files, and no missing assets.")
    approvals = review.get("datasets", {})
    for dataset in report["datasets"]:
        approval = approvals.get(dataset["manifest"], {})
        if approval.get("approved") is not True or not approval.get("license") or not approval.get("reviewed_by"):
            raise ValueError("Explicit redistribution review required for {}".format(dataset["manifest"]))
    current = inventory(root, [dataset["manifest"] for dataset in report["datasets"]], True)
    expected = {record["path"]: record.get("sha256") for record in report["files"]}
    actual = {record["path"]: record.get("sha256") for record in current["files"]}
    if not expected or any(not checksum for checksum in expected.values()) or current["missing"] or expected != actual:
        raise ValueError("Inventory differs from source files or lacks hashes; regenerate using --sha256.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(str(output), "x:gz") as archive:
        for filename in sorted(expected):
            asset, relative = relative_file(root, filename)
            if asset == output.resolve():
                raise ValueError("Archive cannot contain itself.")
            archive.add(str(asset), arcname=relative, recursive=False)
        for name, value in [("release_inventory.json", current), ("redistribution_review.json", review)]:
            content = (json.dumps(value, indent=2) + "\n").encode()
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    return {"archive": output.name, "bytes": output.stat().st_size, "sha256": digest(output)}


def main():
    parser = argparse.ArgumentParser(description="Inventory trusted local motion assets, or package reviewed assets without uploading.")
    subcommands = parser.add_subparsers(dest="command", required=True)
    scan = subcommands.add_parser("inventory")
    scan.add_argument("--root", type=Path, required=True)
    scan.add_argument("--manifest", nargs="+", required=True)
    scan.add_argument("--sha256", action="store_true")
    scan.add_argument("--output", type=Path, required=True)
    archive = subcommands.add_parser("pack")
    archive.add_argument("--root", type=Path, required=True)
    archive.add_argument("--inventory", type=Path, required=True)
    archive.add_argument("--review", type=Path, required=True)
    archive.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == "inventory":
        report = inventory(root, args.manifest, args.sha256)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps({key: report[key] for key in ("datasets", "total_bytes", "missing")}, indent=2, ensure_ascii=False))
    else:
        report = json.loads(args.inventory.read_text())
        review = json.loads(args.review.read_text())
        print(json.dumps(pack(root, report, review, args.output), indent=2))


if __name__ == "__main__":
    main()
