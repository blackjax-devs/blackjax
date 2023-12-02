PKG_VERSION = $(shell python setup.py --version)

test:
	JAX_PLATFORM_NAME=cpu pytest -n 4 --cov=blackjax --cov-report term --cov-report html:coverage tests

# We launch the package release by tagging the master branch with the package's
# new version number. The version number is read from `blackjax/__init__.py`
release:
	git tag -a $(PKG_VERSION) -m $(PKG_VERSION)
	git push --tag


build-docs:
	pip install -r requirements-doc.txt
	PYDEVD_DISABLE_FILE_VALIDATION=1 sphinx-build -b html docs docs/_build/html
