# Contributing to torch-fem

Thank you for considering contributing to torch-fem! We appreciate your time and effort. Here’s a quick guide to help you get started.

Please review `CODE_OF_CONDUCT.md` before participating.

For usage questions and support requests, see `SUPPORT.md`.

## Reporting Issues
- If you find a bug or have a feature request, please open an issue in the [Issues](https://github.com/meyer-nils/torch-fem/issues) section.
- Provide as much detail as possible (screenshot, minimal working example, etc.) to help us understand and reproduce the issue.

## Making Changes
1. **Fork the repository**: Create your own fork of the repo by clicking "Fork" at the top right of the project page.
2. **Set up a development environment**: Install the package in editable mode with the development tools and notebook dependencies:
   ```sh
   pip install -e ".[all]"
   ```
3. **Commit changes in your personal fork**: Ensure your commit messages are clear and descriptive.
4. **Run the checks**: Reproduce the CI checks locally before pushing (see below).
5. **Test your changes**: Run the existing tests and add new ones if you’ve introduced new functionality.
6. **Submit a Pull Request (PR)**: Go to the main repository and click "New Pull Request." Provide a description of your changes and reference any related issue numbers.

## Code Style
- Follow the [black](https://github.com/psf/black) coding style for Python.
- Keep code clean and modular.
- Use type annotations.

## Running the checks
The CI enforces formatting, linting, type checking, and tests. To reproduce them locally:
```sh
black --check src/ tests/       # formatting (drop --check to auto-format)
isort --check-only src/ tests/  # import ordering (drop --check-only to fix)
flake8 src/ tests/              # linting
basedpyright                    # type checking (same engine as Pylance)
pytest -m "not notebook"        # fast unit tests
pytest -m "notebook"            # slow tests that execute the example notebooks
```
