#!/usr/bin/env python3
"""
Helpers for loading multi-project registry configuration.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple


DEFAULT_CONFIG_PATH = os.path.join("config", "projects.json")


def normalize_path(path: str) -> str:
    return path.replace("\\", "/")


def load_registry(config_path: str) -> Dict[str, object]:
    with open(config_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    projects = payload.get("projects")
    if not isinstance(projects, dict) or not projects:
        raise ValueError("config must contain a non-empty 'projects' object")
    return payload


def resolve_project(
    config_path: str,
    project_name: str,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    registry = load_registry(config_path)
    projects = registry["projects"]
    if project_name not in projects:
        raise ValueError("unknown project: %s" % project_name)
    project = projects[project_name]
    if not isinstance(project, dict):
        raise ValueError("project entry must be an object: %s" % project_name)
    return registry, project


def merge_exclude_dirs(base: List[str], extra: List[str]) -> List[str]:
    ordered: List[str] = []
    for item in base + extra:
        if item not in ordered:
            ordered.append(item)
    return ordered


def coalesce(value, fallback):
    if value is None:
        return fallback
    return value


def resolve_paths_from_project(
    project_name: Optional[str],
    config_path: str,
    explicit_root: Optional[str],
    explicit_scan_root: Optional[str],
    explicit_repo_name: Optional[str] = None,
    explicit_exclude_dirs: Optional[List[str]] = None,
) -> Dict[str, object]:
    if not project_name:
        project_root = os.path.abspath(explicit_root or ".")
        scan_root = os.path.abspath(explicit_scan_root or project_root)
        return {
            "project_name": explicit_repo_name or os.path.basename(project_root),
            "project_root": project_root,
            "scan_root": scan_root,
            "exclude_dirs": explicit_exclude_dirs or [],
        }

    _, project = resolve_project(config_path, project_name)
    project_root = os.path.abspath(str(project.get("project_root")))
    scan_root = os.path.abspath(explicit_scan_root or str(project.get("scan_root") or project_root))
    repo_name = explicit_repo_name or str(project.get("repo_name") or project_name)
    exclude_dirs = merge_exclude_dirs(
        list(project.get("exclude_dirs", [])),
        explicit_exclude_dirs or [],
    )
    return {
        "project_name": repo_name,
        "project_root": project_root,
        "scan_root": scan_root,
        "exclude_dirs": exclude_dirs,
    }
