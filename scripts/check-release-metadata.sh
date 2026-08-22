#!/usr/bin/env bash
set -euo pipefail

workspace_version="$(sed -n 's/^version = "\([^"]*\)"$/\1/p' pixi.toml)"
recipe_version="$(sed -n 's/^  version: "\([^"]*\)"$/\1/p' conda.recipe/recipe.yaml)"

[[ "$workspace_version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]
[[ "$recipe_version" == "$workspace_version" ]]

awk '
  BEGIN {
    valid = 1
  }
  $0 == "requirements:" {
    in_requirements = 1
    next
  }
  in_requirements && /^[^ ]/ {
    in_requirements = 0
  }
  in_requirements && /^  (build|host|run):$/ {
    section = $1
    sub(/:$/, "", section)
    next
  }
  in_requirements && /^  [a-z][a-z_-]*:$/ {
    section = ""
    next
  }
  in_requirements && /^    - mojo-compiler/ {
    total++
    if ($0 != "    - mojo-compiler ==1.0.0") {
      valid = 0
    }
    seen[section]++
  }
  END {
    if (!valid || total != 3 || seen["build"] != 1 || \
        seen["host"] != 1 || seen["run"] != 1) {
      print "recipe must pin mojo-compiler ==1.0.0 exactly once in build, host, and run" > "/dev/stderr"
      exit 1
    }
  }
' conda.recipe/recipe.yaml
