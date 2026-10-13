# Contributing

Install the development and documentation groups, then install the git hooks:

```bash
uv sync --group docs
uv run pre-commit install
```

The hooks run ruff (lint with fixes, then the formatter), trailing-whitespace and end-of-file checks, YAML and TOML checks, a 1000 KB added-file limit, merge-conflict and line-ending checks, and `uv run mkdocs build --strict`.

Run the same checks by hand:

```bash
uv run pre-commit run --all-files
uv run pytest
uv run mkdocs build --strict
```

Continuous integration runs ruff and pytest on Python 3.11 and Python 3.13, and runs the pre-commit hooks on the whole tree. Documentation is built on every pull request. Pushes to `master` also publish the site with GitHub Pages.

The site is configured for `https://kosmitive.github.io/sticky-hdp-slds-hmm/`. The repository owner sets that source once under Settings, Pages, Source: GitHub Actions.
