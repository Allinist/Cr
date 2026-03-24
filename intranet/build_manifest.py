#!/usr/bin/env python3
"""
Build a stable page/slice manifest for intranet code projection.

The script is designed to run on old Python 3.7 environments with only the
standard library available.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


TEXT_EXTENSIONS = {
    ".java": "java",
    ".xml": "xml",
    ".sql": "sql",
    ".properties": "properties",
    ".yml": "yaml",
    ".yaml": "yaml",
    ".sh": "shell",
    ".md": "markdown",
    ".txt": "text",
}

BINARY_INDEX_EXTENSIONS = {
    ".class",
    ".jar",
}

DEFAULT_EXCLUDE_DIRS = {
    ".git",
    ".idea",
    ".vscode",
    "node_modules",
    "target",
    "build",
    "dist",
    "out",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build projection manifest.")
    parser.add_argument("--repo-root", default=".", help="Repository root path.")
    parser.add_argument("--output", required=True, help="Output manifest JSON path.")
    parser.add_argument("--page-lines", type=int, default=40, help="Lines per page.")
    parser.add_argument("--page-cols", type=int, default=110, help="Visible columns per slice.")
    parser.add_argument(
        "--include-glob",
        action="append",
        default=[],
        help="Extra include glob, for example src/main/resources/**/*.tpl",
    )
    parser.add_argument(
        "--exclude-dir",
        action="append",
        default=[],
        help="Extra directory name to exclude.",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="Limit to specific relative file paths.",
    )
    parser.add_argument(
        "--repo-name",
        default=None,
        help="Optional explicit repo name. Defaults to repo-root basename.",
    )
    parser.add_argument("--branch", default="unknown", help="Branch name for metadata.")
    parser.add_argument("--commit", default="unknown", help="Commit hash for metadata.")
    parser.add_argument(
        "--mode",
        choices=["full", "diff"],
        default="full",
        help="Manifest build mode.",
    )
    return parser.parse_args()


def normalize_path(path: str) -> str:
    return path.replace("\\", "/")


def read_text_file(path: str) -> Tuple[List[str], str]:
    for encoding in ("utf-8", "utf-8-sig", "gb18030", "latin-1"):
        try:
            with open(path, "r", encoding=encoding, newline="") as handle:
                content = handle.read()
            return content.splitlines(), encoding
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("unknown", b"", 0, 1, "unable to decode file")


def file_sha1(path: str) -> str:
    digest = hashlib.sha1()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def chunk_sha1(parts: Sequence[str]) -> str:
    digest = hashlib.sha1()
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def classify_path(rel_path: str, include_globs: Sequence[str]) -> Tuple[str, Optional[str]]:
    rel_lower = rel_path.lower()
    _, ext = os.path.splitext(rel_lower)
    if ext in TEXT_EXTENSIONS:
        return "text", TEXT_EXTENSIONS[ext]
    if ext in BINARY_INDEX_EXTENSIONS:
        return "binary_index", ext.lstrip(".")
    for pattern in include_globs:
        if fnmatch.fnmatch(rel_path, pattern):
            return "text", "text"
    return "skip", None


def iter_candidate_files(
    repo_root: str,
    exclude_dirs: Sequence[str],
    only_paths: Sequence[str],
    include_globs: Sequence[str],
) -> Iterable[Tuple[str, str, Optional[str]]]:
    only_set = {normalize_path(item) for item in only_paths}
    exclude_set = set(DEFAULT_EXCLUDE_DIRS)
    exclude_set.update(exclude_dirs)

    for current_root, dir_names, file_names in os.walk(repo_root):
        dir_names[:] = [item for item in dir_names if item not in exclude_set]
        for file_name in sorted(file_names):
            abs_path = os.path.join(current_root, file_name)
            rel_path = normalize_path(os.path.relpath(abs_path, repo_root))
            if only_set and rel_path not in only_set:
                continue
            item_type, language = classify_path(rel_path, include_globs)
            if item_type == "skip":
                continue
            yield abs_path, item_type, language


def compute_line_slice_count(lines: Sequence[str], page_cols: int) -> int:
    max_len = 1
    for line in lines:
        if len(line) > max_len:
            max_len = len(line)
    return max(1, (max_len + page_cols - 1) // page_cols)


def slice_lines(lines: Sequence[str], col_start: int, col_end: int) -> List[str]:
    sliced = []
    for line in lines:
        start_index = max(col_start - 1, 0)
        end_index = max(col_end, 0)
        sliced.append(line[start_index:end_index])
    return sliced


def build_text_entry(
    abs_path: str,
    repo_root: str,
    language: str,
    page_lines: int,
    page_cols: int,
    branch: str,
    commit: str,
    repo_name: str,
    mode: str,
) -> Dict[str, object]:
    rel_path = normalize_path(os.path.relpath(abs_path, repo_root))
    lines, encoding = read_text_file(abs_path)
    total_lines = len(lines)
    max_line_len = max([len(line) for line in lines] or [0])
    total_pages = max(1, (max(total_lines, 1) + page_lines - 1) // page_lines)
    file_digest = file_sha1(abs_path)

    pages = []
    total_slices = 0
    for page_no in range(1, total_pages + 1):
        start_line = (page_no - 1) * page_lines + 1
        end_line = min(page_no * page_lines, total_lines)
        page_body = lines[start_line - 1:end_line]
        slice_total = compute_line_slice_count(page_body, page_cols)
        total_slices += slice_total
        for slice_no in range(1, slice_total + 1):
            col_start = (slice_no - 1) * page_cols + 1
            col_end = slice_no * page_cols
            body_lines = slice_lines(page_body, col_start, col_end)
            chunk_id = chunk_sha1(
                [
                    rel_path,
                    str(start_line),
                    str(end_line),
                    str(slice_no),
                    str(col_start),
                    str(col_end),
                    "\n".join(body_lines),
                ]
            )
            pages.append(
                {
                    "repo": repo_name,
                    "branch": branch,
                    "commit": commit,
                    "mode": mode,
                    "file": rel_path,
                    "language": language,
                    "page": page_no,
                    "page_total": total_pages,
                    "slice_no": slice_no,
                    "slice_total": slice_total,
                    "start_line": start_line,
                    "end_line": end_line,
                    "col_start": col_start,
                    "col_end": col_end,
                    "chunk_id": chunk_id,
                    "body": body_lines,
                }
            )

    return {
        "type": "text",
        "file": rel_path,
        "language": language,
        "encoding": encoding,
        "total_lines": total_lines,
        "max_line_len": max_line_len,
        "page_total": total_pages,
        "slice_total": total_slices,
        "file_sha1": file_digest,
        "pages": pages,
    }


def build_binary_index_entry(abs_path: str, repo_root: str, language: str) -> Dict[str, object]:
    rel_path = normalize_path(os.path.relpath(abs_path, repo_root))
    stat = os.stat(abs_path)
    return {
        "type": "binary_index",
        "file": rel_path,
        "language": language,
        "size": stat.st_size,
        "mtime": int(stat.st_mtime),
        "file_sha1": file_sha1(abs_path),
    }


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)


def build_manifest(args: argparse.Namespace) -> Dict[str, object]:
    repo_root = os.path.abspath(args.repo_root)
    repo_name = args.repo_name or os.path.basename(repo_root)

    text_files = []
    binary_files = []
    for abs_path, item_type, language in iter_candidate_files(
        repo_root=repo_root,
        exclude_dirs=args.exclude_dir,
        only_paths=args.only,
        include_globs=args.include_glob,
    ):
        if item_type == "text":
            text_files.append(
                build_text_entry(
                    abs_path=abs_path,
                    repo_root=repo_root,
                    language=language or "text",
                    page_lines=args.page_lines,
                    page_cols=args.page_cols,
                    branch=args.branch,
                    commit=args.commit,
                    repo_name=repo_name,
                    mode=args.mode,
                )
            )
        elif item_type == "binary_index":
            binary_files.append(build_binary_index_entry(abs_path, repo_root, language or "binary"))

    total_pages = sum(len(item["pages"]) for item in text_files)
    return {
        "version": 1,
        "repo_root": normalize_path(repo_root),
        "repo_name": repo_name,
        "branch": args.branch,
        "commit": args.commit,
        "mode": args.mode,
        "page_lines": args.page_lines,
        "page_cols": args.page_cols,
        "text_files": text_files,
        "binary_files": binary_files,
        "stats": {
            "text_file_count": len(text_files),
            "binary_index_count": len(binary_files),
            "page_count": total_pages,
        },
    }


def main() -> int:
    args = parse_args()
    manifest = build_manifest(args)
    ensure_parent_dir(args.output)
    with open(args.output, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print("manifest written to %s" % args.output)
    print("text_files=%s pages=%s" % (manifest["stats"]["text_file_count"], manifest["stats"]["page_count"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
