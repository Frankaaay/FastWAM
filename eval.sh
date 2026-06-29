#!/bin/bash
set -u
cd "$(dirname "$0")" || exit 1
exec bash scripts/eval.sh "$@"
