#!/usr/bin/env bash
set -euo pipefail

benchmark_binary=.pixi/bench-least-squares
mkdir -p .pixi

if command -v shasum >/dev/null 2>&1; then
  hash_command=(shasum -a 256)
elif command -v sha256sum >/dev/null 2>&1; then
  hash_command=(sha256sum)
else
  printf '%s\n' 'benchmark provenance requires shasum or sha256sum' >&2
  exit 1
fi

git_head=$(git rev-parse HEAD 2>/dev/null || printf 'unavailable')
if git diff --quiet --ignore-submodules HEAD -- &&
  [[ -z $(git ls-files --others --exclude-standard) ]]; then
  git_state=clean
else
  git_state=dirty
fi

source_manifest_sha256=$(
  {
    find src/nerai -type f -name '*.mojo'
    printf '%s\n' benchmarks/bench_least_squares.mojo
  } | LC_ALL=C sort | while IFS= read -r source_file; do
    "${hash_command[@]}" "$source_file"
  done | "${hash_command[@]}" | awk '{print $1}'
)

if command -v sysctl >/dev/null 2>&1; then
  cpu_model=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || uname -m)
else
  cpu_model=$(uname -m)
fi

printf 'BENCH_PROVENANCE git_head=%s git_state=%s\n' "$git_head" "$git_state"
printf 'BENCH_PROVENANCE source_manifest_sha256=%s\n' \
  "$source_manifest_sha256"
printf 'BENCH_PROVENANCE timestamp_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf 'BENCH_PROVENANCE os=%s arch=%s cpu=%s\n' \
  "$(uname -sr)" "$(uname -m)" "$cpu_model"
if command -v pmset >/dev/null 2>&1; then
  printf 'BENCH_PROVENANCE power=%s\n' "$(pmset -g batt | sed -n '1p')"
  printf 'BENCH_PROVENANCE thermal=%s\n' \
    "$(pmset -g therm 2>/dev/null | tr '\n' ';' || printf 'unavailable')"
fi
printf 'BENCH_PROVENANCE mojo=%s\n' "$(mojo --version | head -n 1)"
printf '%s\n' \
  'BENCH_PROVENANCE build=mojo build --optimization-level 3 -I src benchmarks/bench_least_squares.mojo -o .pixi/bench-least-squares'

mojo build --optimization-level 3 -I src benchmarks/bench_least_squares.mojo \
  -o "$benchmark_binary"
"$benchmark_binary"
