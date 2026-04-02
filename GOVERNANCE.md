# Governance

## Project model

torch-fem currently follows a maintainer-led governance model.

This is an early-stage research software project. APIs and internal behavior may
change as methods are improved and validated.

- Lead maintainer: Nils Meyer
- Contributors: community members via pull requests and issue discussions

## Decision process

Technical decisions are made in public through GitHub issues and pull requests.

- Backward-incompatible API or behavior changes are allowed during this research
  phase. Opening an issue before major changes is encouraged, but not always
  required.
- Small bug fixes and documentation updates may be merged directly by a
  maintainer after review.

## Contribution and review

- Contributions are welcome through pull requests.
- Pull requests from external contributors should receive maintainer review before
  merge.
- Maintainer-authored changes may be merged directly when needed.
- CI checks must pass before merge.

## Release process

- Versioning uses semantic versioning style (`MAJOR.MINOR.PATCH`).
- User-facing changes are documented in `CHANGELOG.md`.
- Releases are tagged in git and published to PyPI via GitHub Actions.

## Long-term maintenance

The project is actively developed for research use and accepts community
contributions. Priorities are driven by research relevance, correctness,
reproducibility, and maintainable API evolution.
