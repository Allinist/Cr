#!/usr/bin/env python3
"""
Scan all directories and file types under a project folder.

Designed for restricted intranet environments:
- Python standard library only
- Works on Python 3.7+
- Outputs JSON for downstream automation
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List


DEFAULT_EXCLUDE_DIRS = {
    ".git",
    ".idea",
    ".vscode",
    "node_modules",
    "target",
    "build",
    "dist",
    "out",
    "__pycache__",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan project folders and file types.")
    parser.add_argument("--root", default=".", help="Project root directory.")
    parser.add_argument("--output", required=True, help="Output JSON path.")
    parser.add_argument(
        "--exclude-dir",
        action="append",
        default=[],
        help="Additional directory names to exclude.",
    )
    parser.add_argument(
        "--include-hidden",
        action="store_true",
        help="Include hidden files and directories.",
    )
    return parser.parse_args()


def normalize_path(path: str) -> str:
    return path.replace("\\", "/")


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)


def is_hidden_name(name: str) -> bool:
    return name.startswith(".")


def detect_file_type(file_name: str) -> str:
    _, ext = os.path.splitext(file_name)
    if ext:
        return ext.lower()
    return "[no_ext]"


def build_directory_entry(rel_dir: str) -> Dict[str, object]:
    return {
        "path": rel_dir,
        "file_count": 0,
        "subdir_count": 0,
        "file_types": {},
    }


def scan_project(root: str, exclude_dirs: List[str], include_hidden: bool) -> Dict[str, object]:
    root = os.path.abspath(root)
    exclude_set = set(DEFAULT_EXCLUDE_DIRS)
    exclude_set.update(exclude_dirs)

    file_type_stats: Dict[str, int] = defaultdict(int)
    directories: Dict[str, Dict[str, object]] = {}
    files: List[Dict[str, object]] = []

    for current_root, dir_names, file_names in os.walk(root):
        filtered_dirs = []
        for dir_name in sorted(dir_names):
            if dir_name in exclude_set:
                continue
            if not include_hidden and is_hidden_name(dir_name):
                continue
            filtered_dirs.append(dir_name)
        dir_names[:] = filtered_dirs

        rel_dir = normalize_path(os.path.relpath(current_root, root))
        if rel_dir == ".":
            rel_dir = "."
        dir_entry = directories.setdefault(rel_dir, build_directory_entry(rel_dir))
        dir_entry["subdir_count"] = len(dir_names)

        for file_name in sorted(file_names):
            if not include_hidden and is_hidden_name(file_name):
                continue
            abs_path = os.path.join(current_root, file_name)
            rel_path = normalize_path(os.path.relpath(abs_path, root))
            file_type = detect_file_type(file_name)
            size = os.path.getsize(abs_path)

            file_record = {
                "path": rel_path,
                "dir": rel_dir,
                "name": file_name,
                "type": file_type,
                "size": size,
            }
            files.append(file_record)

            file_type_stats[file_type] += 1
            dir_entry["file_count"] += 1
            dir_types = dir_entry["file_types"]
            dir_types[file_type] = dir_types.get(file_type, 0) + 1

    sorted_types = [
        {"type": file_type, "count": count}
        for file_type, count in sorted(file_type_stats.items(), key=lambda item: (-item[1], item[0]))
    ]
    sorted_dirs = [
        directories[key]
        for key in sorted(directories.keys())
    ]
    files.sort(key=lambda item: item["path"])

    return {
        "root": normalize_path(root),
        "directory_count": len(sorted_dirs),
        "file_count": len(files),
        "file_types": sorted_types,
        "directories": sorted_dirs,
        "files": files,
    }


def main() -> int:
    args = parse_args()
    payload = scan_project(args.root, args.exclude_dir, args.include_hidden)
    ensure_parent_dir(args.output)
    with open(args.output, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print("scan written to %s" % args.output)
    print("directories=%s files=%s" % (payload["directory_count"], payload["file_count"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
