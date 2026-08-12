#!/usr/bin/env python3
"""Download and verify third-party model assets declared in weights-manifest.json."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable


DEFAULT_MANIFEST = Path(__file__).resolve().parents[1] / "weights-manifest.json"


class ManifestError(ValueError):
    pass


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") != 1:
        raise ManifestError("unsupported or missing schema_version")
    models = manifest.get("models")
    if not isinstance(models, list) or not models:
        raise ManifestError("manifest.models must be a non-empty list")
    seen: set[str] = set()
    for model in models:
        model_id = model.get("id")
        if not isinstance(model_id, str) or not model_id:
            raise ManifestError("every model needs a non-empty id")
        if model_id in seen:
            raise ManifestError(f"duplicate model id: {model_id}")
        seen.add(model_id)
        if model.get("provider") not in {"huggingface", "modelscope"}:
            raise ManifestError(f"unsupported provider for {model_id}")
        if not model.get("repo_id"):
            raise ManifestError(f"missing repo_id for {model_id}")
        files = model.get("files", [])
        if not isinstance(files, list):
            raise ManifestError(f"files must be a list for {model_id}")
        if not files and not model.get("directory_target"):
            raise ManifestError(f"{model_id} needs files or directory_target")
        required_files = model.get("required_files", [])
        if not isinstance(required_files, list) or not all(
            isinstance(item, str) and item for item in required_files
        ):
            raise ManifestError(f"required_files must contain relative paths for {model_id}")
    return manifest


def safe_target(root: Path, relative: str) -> Path:
    if not relative or Path(relative).is_absolute():
        raise ManifestError(f"target must be a non-empty relative path: {relative!r}")
    resolved_root = root.resolve()
    resolved = (resolved_root / relative).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ManifestError(f"target escapes root: {relative}") from exc
    return resolved


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_models(
    manifest: dict[str, Any], requested: Iterable[str], include_optional: bool
) -> list[dict[str, Any]]:
    models = manifest["models"]
    requested_set = set(requested)
    known = {model["id"] for model in models}
    unknown = requested_set - known
    if unknown:
        raise ManifestError("unknown model id(s): " + ", ".join(sorted(unknown)))
    if requested_set:
        return [model for model in models if model["id"] in requested_set]
    return [model for model in models if model.get("required") or include_optional]


def expected_paths(root: Path, model: dict[str, Any]) -> list[tuple[Path, dict[str, Any] | None]]:
    files = model.get("files", [])
    if files:
        return [(safe_target(root, entry["target"]), entry) for entry in files]
    return [(safe_target(root, model["directory_target"]), None)]


def verify_model(root: Path, model: dict[str, Any], deep: bool) -> list[str]:
    problems: list[str] = []
    for path, entry in expected_paths(root, model):
        if not path.exists():
            problems.append(f"{model['id']}: missing {path}")
            continue
        if entry is None:
            if not path.is_dir() or not any(path.iterdir()):
                problems.append(f"{model['id']}: empty model directory {path}")
                continue
            for relative in model.get("required_files", []):
                required = safe_target(path, relative)
                if not required.is_file():
                    problems.append(f"{model['id']}: missing required file {required}")
            continue
        if not path.is_file():
            problems.append(f"{model['id']}: expected file {path}")
            continue
        expected_size = entry.get("size_bytes")
        if expected_size is not None and path.stat().st_size != expected_size:
            problems.append(
                f"{model['id']}: size mismatch for {path} "
                f"(expected {expected_size}, got {path.stat().st_size})"
            )
            continue
        expected_sha = entry.get("sha256")
        if deep and expected_sha and sha256_file(path).lower() != expected_sha.lower():
            problems.append(f"{model['id']}: sha256 mismatch for {path}")
    return problems


def _download_huggingface(model: dict[str, Any], destination: Path) -> None:
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError as exc:
        raise RuntimeError("install huggingface_hub to download Hugging Face models") from exc

    revision = model.get("revision")
    files = model.get("files", [])
    if files:
        for entry in files:
            target = safe_target(destination, entry["target"])
            target.parent.mkdir(parents=True, exist_ok=True)
            downloaded = hf_hub_download(
                repo_id=model["repo_id"], filename=entry["source"], revision=revision
            )
            with tempfile.NamedTemporaryFile(
                dir=target.parent, prefix=f".{target.name}.", suffix=".part", delete=False
            ) as temporary:
                temporary_path = Path(temporary.name)
            try:
                shutil.copyfile(downloaded, temporary_path)
                temporary_path.replace(target)
            finally:
                temporary_path.unlink(missing_ok=True)
        return

    target = safe_target(destination, model["directory_target"])
    target.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=model["repo_id"], revision=revision, local_dir=str(target)
    )


def _download_modelscope(model: dict[str, Any], destination: Path) -> None:
    try:
        from modelscope.hub.snapshot_download import snapshot_download
    except ImportError as exc:
        raise RuntimeError("install modelscope to download ModelScope models") from exc

    target = safe_target(destination, model["directory_target"])
    target.parent.mkdir(parents=True, exist_ok=True)
    kwargs: dict[str, Any] = {"model_id": model["repo_id"], "local_dir": str(target)}
    if model.get("revision"):
        kwargs["revision"] = model["revision"]
    snapshot_download(**kwargs)


def download_model(root: Path, model: dict[str, Any], force: bool) -> None:
    existing = expected_paths(root, model)
    if not force and all(path.exists() for path, _ in existing):
        problems = verify_model(root, model, deep=True)
        if problems:
            raise RuntimeError(
                "; ".join(problems) + "; remove the invalid target or rerun with --force"
            )
        print(f"SKIP {model['id']}: target already exists and verifies")
        return
    if model["provider"] == "huggingface":
        _download_huggingface(model, root)
    else:
        _download_modelscope(model, root)
    problems = verify_model(root, model, deep=True)
    if problems:
        raise RuntimeError("; ".join(problems))
    print(f"READY {model['id']}")


def command_verify(root: Path, models: list[dict[str, Any]], deep: bool) -> int:
    problems = [problem for model in models for problem in verify_model(root, model, deep)]
    if problems:
        for problem in problems:
            print(f"ERROR {problem}", file=sys.stderr)
        return 1
    for model in models:
        print(f"OK {model['id']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_MANIFEST.parent,
        help="installation root; defaults to the repository root",
    )
    parser.add_argument("--model", action="append", default=[], help="model id; repeatable")
    parser.add_argument("--include-optional", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)
    download = subparsers.add_parser("download")
    download.add_argument("--force", action="store_true")
    verify = subparsers.add_parser("verify")
    verify.add_argument("--deep", action="store_true", help="compute known SHA-256 hashes")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
        root = args.root.resolve()
        root.mkdir(parents=True, exist_ok=True)
        models = selected_models(manifest, args.model, args.include_optional)
        if args.command == "verify":
            return command_verify(root, models, args.deep)
        for model in models:
            download_model(root, model, args.force)
        return 0
    except (ManifestError, RuntimeError, OSError) as exc:
        print(f"ERROR {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
