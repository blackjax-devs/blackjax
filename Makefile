PKG_VERSION = $(shell uv run python -m setuptools_scm)

test:
    uv sync --group dev --extra progress
	JAX_PLATFORM_NAME=cpu uv run pytest -n auto -vv --benchmark-disable --cov=blackjax --cov-report=xml --cov-report=term tests

# We launch the package release by tagging the master branch with the package's
# new version number. The version number is read from git via setuptools_scm.
release:
	git tag -a $(PKG_VERSION) -m $(PKG_VERSION)
	git push --tag


build-docs:
	uv sync --group docs
	PYDEVD_DISABLE_FILE_VALIDATION=1 uv run sphinx-build -b html docs docs/_build/html
