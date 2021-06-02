PKG_VERSION = $(shell python setup.py --version)

lint:
	isort --profile black blackjax tests
	black blackjax tests

lint_check:
	isort --profile black --check blackjax tests
	flake8 blackjax tests --count --ignore=E501,E203,E731,W503 --show-source --statistics
	black --check blackjax tests
	mypy blackjax tests

test:
	pytest -n 4 --cov=blackjax --cov-report term --cov-report html:coverage tests

# We launch the package release by tagging the master branch with the package's
# new version number. The version number is read from `blackjax/__init__.py`
publish:
	git tag -a $(LIB_VERSION) -m $(LIB_VERSION)
	git push --tag

