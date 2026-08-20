#!/usr/bin/env bash
set -euo pipefail

for test_file in tests/test_*.mojo; do
  mojo run -I src "$test_file"
done

mkdir -p .pixi/test-bin
mojo build -I src examples/basic.mojo -o .pixi/test-bin/basic

compile_fail_dir=$(mktemp -d "${TMPDIR:-/tmp}/nerai-compile-fail.XXXXXX")
cleanup() {
  if [[ -n "${compile_fail_dir:-}" && -d "$compile_fail_dir" ]]; then
    rm -rf -- "$compile_fail_dir"
  fi
}
trap cleanup EXIT

for fixture in tests/compile_fail/*.mojo; do
  fixture_name=$(basename "${fixture%.mojo}")
  diagnostic="$compile_fail_dir/$fixture_name.stderr"
  binary="$compile_fail_dir/$fixture_name"
  if mojo build -I src "$fixture" -o "$binary" >"$diagnostic" 2>&1; then
    printf 'Expected compilation failure: %s\n' "$fixture" >&2
    exit 1
  fi

  while IFS= read -r expected; do
    if [[ -n "$expected" ]] && ! grep -Fq -- "$expected" "$diagnostic"; then
      printf 'Missing diagnostic %q for %s\n' "$expected" "$fixture" >&2
      sed -n '1,120p' "$diagnostic" >&2
      exit 1
    fi
  done <"${fixture%.mojo}.stderr"
done

printf 'Compile-fail API checks passed.\n'
