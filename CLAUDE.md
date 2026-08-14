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

## Installing AmgX (optional GPU backend)
Needs the CUDA toolkit whose version matches the installed torch and cupy, a host compiler (MSVC 2022 Build Tools on Windows, gcc on Linux) and `pip install cmake ninja`.
``` sh
git clone --depth 1 https://github.com/NVIDIA/AMGX   # main, no submodules
cmake -S AMGX -B AMGX/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_NO_MPI=TRUE -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build AMGX/build   # ~5 minutes for one architecture
export AMGX_DLL=$PWD/AMGX/build/libamgxsh.so   # setx AMGX_DLL ...\amgxsh.dll on Windows
```
- Build for this GPU's compute capability only, from `torch.cuda.get_device_capability()`; the default list of all architectures takes an hour and needs a toolkit new enough for each.
- Windows: configure from an "x64 Native Tools Command Prompt for VS 2022", otherwise Ninja does not find `cl.exe`.
- `CMAKE_NO_MPI=TRUE` skips the MPI search, which the single-GPU binding never uses.
- The library resolves cuBLAS, cuSPARSE and cuSOLVER at load time from the toolkit's `bin` on `PATH`.
- Check with `python -c "import torch; import torchfem.amgx"`, which fails loudly if the library is missing.