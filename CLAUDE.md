# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Keep it simple
- Build only what was asked: no extra features, no abstractions for single-use code, no unnecessary configurability, no error handling for impossible cases.
- Write the shortest version that works. If it's 200 lines and could be 50, rewrite it.
- Don't touch what isn't broken.
- Mention unrelated dead code briefly, don't delete it.
- Match existing style, even if you'd do it differently.
- Keep comments and docstrings concise. Make sure they make sense without the context of the chat.
- Describe the code as it is, never how it got there.
- Default to one line. Write a second only if the first cannot carry the meaning.

## Commands
- **Environment:** `conda activate torchfem`
- **Lint:** `ruff format . && ruff check --fix .`
- **Types:** `basedpyright`
- **Test:** `pytest`

## This repo
- Public API change: add a `CHANGELOG.md` entry under "Unreleased", maximum one or two sentences.
- New example notebook: add a card in `docs/examples.md` and a test in `tests/test_notebooks.py`.
- Everything runs in float64.