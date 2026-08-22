#!/usr/bin/env bash
set -euo pipefail

workspace_version="$(sed -n 's/^version = "\([^"]*\)"$/\1/p' pixi.toml)"
metadata_dir="$(mktemp -d "${TMPDIR:-/tmp}/nerai-package-metadata.XXXXXX")"
cleanup() {
  if [[ -n "${metadata_dir:-}" && -d "$metadata_dir" ]]; then
    rm -rf -- "$metadata_dir"
  fi
}
trap cleanup EXIT

artifacts=()
while IFS= read -r artifact; do
  artifacts+=("$artifact")
done < <(find output -type f -name "mojo-nerai-${workspace_version}-*.conda" -print)
if [[ "${#artifacts[@]}" -ne 1 ]]; then
  printf 'expected one mojo-nerai %s package, found %d\n' \
    "$workspace_version" "${#artifacts[@]}" >&2
  exit 1
fi

pixi run rattler-build package extract \
  --dest "$metadata_dir" "${artifacts[0]}" >/dev/null
python3 - "$metadata_dir/info/index.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as index_file:
    index = json.load(index_file)

compiler_dependencies = [
    dependency
    for dependency in index.get("depends", [])
    if dependency.split(maxsplit=1)[0] == "mojo-compiler"
]
if compiler_dependencies != ["mojo-compiler ==1.0.0"]:
    raise SystemExit(
        "package must depend on exactly mojo-compiler ==1.0.0; got "
        + repr(compiler_dependencies)
    )
PY
