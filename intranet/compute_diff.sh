#!/usr/bin/env bash
set -eu

OLD_COMMIT="${1:?old commit required}"
NEW_COMMIT="${2:?new commit required}"

git diff --name-status "$OLD_COMMIT" "$NEW_COMMIT"
