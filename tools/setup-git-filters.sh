#!/usr/bin/env bash
# Register the notebook output-stripping filter for this clone.
#
# Git filter definitions live in .git/config, which is NOT committed, so this
# must be run once per clone. .gitattributes (which IS committed) says WHICH
# files use the filter; this says WHAT the filter is.
#
#   ./tools/setup-git-filters.sh
#
# Effect: notebooks under code/ are committed without outputs. Your working
# copy is untouched -- you keep seeing your results locally.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! command -v python3 >/dev/null 2>&1; then
  echo "error: python3 not found; the nbstrip filter needs it." >&2
  exit 1
fi

git config filter.nbstrip.clean  "python3 '$ROOT/tools/nbstrip.py'"
git config filter.nbstrip.smudge cat
git config filter.nbstrip.required true

echo "Registered filter.nbstrip in .git/config:"
echo "  clean  = python3 tools/nbstrip.py   (strips outputs on 'git add')"
echo "  smudge = cat                        (working copy left as-is)"
echo
echo "Applies to paths marked 'filter=nbstrip' in .gitattributes:"
git check-attr filter -- code/transfer_CRRA_wage.ipynb
echo
echo "Note: archive/ notebooks stay in Git LFS and are not touched."
