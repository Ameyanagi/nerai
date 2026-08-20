# Compatibility

## Toolchain

Development currently pins Mojo `1.0.0`. Precompiled `.mojoc` files are tied to
the exact compiler version that produced them, so both the Pixi environment and
Conda recipe pin the compiler. Compiler upgrades are explicit compatibility
events and require the full locked test suite.

## Platforms

| Platform | Status |
| --- | --- |
| macOS ARM64 | CI target |
| Linux x86-64 | CI target |
| Linux ARM64 | CI target |
| Windows/WSL | Not yet supported or tested |
| GPU | Not supported unless explicitly listed in the roadmap |

The repository is experimental and has no source-compatibility promise before
its first release. Each release names the exact compiler used to build it.

## Mojo 1.0 public-field limitation

Mojo 1.0 callers can mutate underscore-prefixed struct fields. Nerai's closed
semantic values remain defined under that mutation; numeric report snapshots
require revalidation after caller mutation. The exact guarantees and audited
public structs are documented in
[Mojo 1.0 mutation and invariants](design.md#mojo-10-mutation-and-invariants).
