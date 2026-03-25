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

from project_registry import DEFAULT_CONFIG_PATH, resolve_paths_from_project


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
    parser.add_argument("--project", default=None, help="Registered project name from config/projects.json.")
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="Project registry JSON path.",
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Stable repository root used to build relative paths.",
    )
    parser.add_argument(
        "--scan-root",
        default=None,
        help="Actual directory to scan. Defaults to repo-root.",
    )
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
    scan_root: str,
    exclude_dirs: Sequence[str],
    only_paths: Sequence[str],
    include_globs: Sequence[str],
) -> Iterable[Tuple[str, str, Optional[str]]]:
    only_set = {normalize_path(item) for item in only_paths}
    exclude_set = set(DEFAULT_EXCLUDE_DIRS)
    exclude_set.update(exclude_dirs)

    for current_root, dir_names, file_names in os.walk(scan_root):
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


def wrap_source_line(line_no: int, text: str, page_cols: int) -> List[Dict[str, object]]:
    if page_cols <= 0:
        raise ValueError("page-cols must be greater than 0")

    if text == "":
        return [{"source_line_no": line_no, "display_line_no": line_no, "text": ""}]

    rows = []
    segments = max(1, (len(text) + page_cols - 1) // page_cols)
    for index in range(segments):
        start = index * page_cols
        end = start + page_cols
        rows.append(
            {
                "source_line_no": line_no,
                "display_line_no": line_no if index == 0 else None,
                "text": text[start:end],
            }
        )
    return rows


def build_visual_rows(lines: Sequence[str], page_cols: int) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for offset, text in enumerate(lines):
        rows.extend(wrap_source_line(offset + 1, text, page_cols))
    return rows


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
    file_name = os.path.basename(abs_path)
    lines, encoding = read_text_file(abs_path)
    total_lines = len(lines)
    max_line_len = max([len(line) for line in lines] or [0])
    visual_rows = build_visual_rows(lines, page_cols)
    total_visual_rows = len(visual_rows)
    total_pages = max(1, (max(total_visual_rows, 1) + page_lines - 1) // page_lines)
    file_digest = file_sha1(abs_path)

    pages = []
    for page_no in range(1, total_pages + 1):
        start_index = (page_no - 1) * page_lines
        end_index = min(page_no * page_lines, total_visual_rows)
        page_rows = visual_rows[start_index:end_index]
        source_line_numbers = [int(row["source_line_no"]) for row in page_rows]
        start_line = min(source_line_numbers) if source_line_numbers else 1
        end_line = max(source_line_numbers) if source_line_numbers else start_line
        chunk_id = chunk_sha1(
            [
                rel_path,
                str(start_line),
                str(end_line),
                "\n".join(
                    "%s|%s" % (
                        "" if row["display_line_no"] is None else row["display_line_no"],
                        row["text"],
                    )
                    for row in page_rows
                ),
            ]
        )
        pages.append(
            {
                "repo": repo_name,
                "branch": branch,
                "commit": commit,
                "mode": mode,
                "file": rel_path,
                "file_name": file_name,
                "language": language,
                "page": page_no,
                "page_total": total_pages,
                "page_lines": page_lines,
                "start_line": start_line,
                "end_line": end_line,
                "visual_row_count": len(page_rows),
                "chunk_id": chunk_id,
                "visual_rows": page_rows,
            }
        )

    return {
        "type": "text",
        "file": rel_path,
        "file_name": file_name,
        "language": language,
        "encoding": encoding,
        "total_lines": total_lines,
        "total_visual_rows": total_visual_rows,
        "max_line_len": max_line_len,
        "page_total": total_pages,
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
    resolved = resolve_paths_from_project(
        project_name=args.project,
        config_path=args.config,
        explicit_root=args.repo_root,
        explicit_scan_root=args.scan_root,
        explicit_repo_name=args.repo_name,
        explicit_exclude_dirs=args.exclude_dir,
    )
    repo_root = os.path.abspath(resolved["project_root"])
    scan_root = os.path.abspath(resolved["scan_root"])
    if os.path.commonpath([repo_root, scan_root]) != repo_root:
        raise ValueError("scan-root must be inside repo-root")
    repo_name = str(resolved["project_name"] or os.path.basename(repo_root))
    scan_root_rel = normalize_path(os.path.relpath(scan_root, repo_root))
    if scan_root_rel == ".":
        scan_root_rel = "."

    text_files = []
    binary_files = []
    for abs_path, item_type, language in iter_candidate_files(
        repo_root=repo_root,
        scan_root=scan_root,
        exclude_dirs=resolved["exclude_dirs"],
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
        "project_name": repo_name,
        "config_path": normalize_path(os.path.abspath(args.config)),
        "repo_root": normalize_path(repo_root),
        "scan_root": normalize_path(scan_root),
        "scan_root_relative": scan_root_rel,
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
