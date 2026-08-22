# Releasing

1. Update the changelog, compatibility notes, and both package-version fields.
2. Reset the Conda build number to zero for a new version; increment it only
   when rebuilding the same source version.
3. Commit the release metadata, then run `pixi run --locked check` and
   `pixi run --locked package` on a clean tree.
4. Merge the release commit and wait for CI to pass on that exact `main` SHA.
5. Create and push an annotated or signed `vX.Y.Z` tag on that SHA.
6. Verify the tag workflow and run the `mojo-channel` build in preflight mode.
7. Publish the three native packages through `mojo-channel`; its workflow
   rejects mutable refs and package-file overwrites.
8. Replace the local `source.path` in the modular-community recipe submission
   with the repository URL and full 40-character tag commit SHA.
9. Build and install-smoke the modular-community submission separately.
10. Publish benchmark results only with the checked-in methodology.

The tag workflow creates a source archive after the supported check and package
matrix passes. Publishing to modular-community remains a separate reviewed
operation.
