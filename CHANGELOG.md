# Changelog

Package versions (`deixis.__version__`) and wire schema versions (`version` in every message)
are separate. Schema 1.x only adds optional fields and new enum values.

## 0.1.0 (unreleased)

First release, split out of `deixis-doodle` (the pen plotter) in October 2026.

- Wire schema 1.0: `Where` (adds `version`, `z_mm`, `thing`, `label`; sources `default`
  and `adjust`), `Size`, `Ask`, `Answer`, `Profile`. Committed under `schema/`.
- Robot profiles (`deixis.profile`), with three examples: plotter, vacuum, arm.
