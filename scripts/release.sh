#!/usr/bin/env bash
set -euo pipefail

VERSION="${1:?Usage: ./scripts/release.sh <version>  (e.g. 0.1.13)}"

# Validate format
if ! echo "$VERSION" | grep -qE '^[0-9]+\.[0-9]+\.[0-9]+$'; then
    echo "Error: version must be semver (e.g. 0.1.13)" >&2
    exit 1
fi

# Ensure we're on main and up to date
BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [ "$BRANCH" != "main" ]; then
    echo "Error: must be on main branch (currently on $BRANCH)" >&2
    exit 1
fi
git pull --ff-only origin main

# Bump version in pyproject.toml
sed -i.bak "s/^version = \".*\"/version = \"$VERSION\"/" pyproject.toml
rm -f pyproject.toml.bak

# Commit, tag, push
git add pyproject.toml
git commit -m "release: v${VERSION}"
git tag "v${VERSION}"
git push origin main "v${VERSION}"

echo "Released v${VERSION} — GitHub Actions will build and attach wheels."
echo "https://github.com/PrimeIntellect-ai/router/releases/tag/v${VERSION}"
