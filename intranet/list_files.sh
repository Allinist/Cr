#!/usr/bin/env bash
set -eu

REPO_ROOT="${1:-.}"

find "$REPO_ROOT" \
  -type d \( -name .git -o -name target -o -name build -o -name dist -o -name node_modules \) -prune -o \
  -type f \( \
    -name "*.java" -o \
    -name "*.xml" -o \
    -name "*.sql" -o \
    -name "*.properties" -o \
    -name "*.yml" -o \
    -name "*.yaml" -o \
    -name "*.sh" -o \
    -name "*.md" -o \
    -name "*.txt" -o \
    -name "*.class" \
  \) -print | sort
