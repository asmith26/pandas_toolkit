help:                           ## Show help docs
	@sed -ne '/@sed/!s/## //p' $(MAKEFILE_LIST)

bump2version-patch: test-all    ## Bump package patch version
	bump2version patch

bump2version-minor: test-all    ## Bump package minor version
	bump2version minor

bump2version-major: test-all    ## Bump package major version
	bump2version major

clean-test:                     ## Remove all coverage, lint, test artifacts
	rm -f .coverage
	rm -rf .mypy_cache
	rm -rf .tox
	rm -rf *.egg-info

coverage:                       ## Create python test coverage report and open in firefox
	coverage run -m nose2 -v
	coverage report
	coverage html
	firefox htmlcov/index.html

deploy-doc: update-api-doc      ## Deploy doc to github pages
	mkdocs gh-deploy

fix-lint:                       ## Automatically fix style violations
	black --line-length=120 pandas_toolkit setup.py
	isort --lines 120 --recursive --use-parentheses pandas_toolkit setup.py

lint:                           ## Run lint checks
	tox -e lint

test-all: clean-test            ## Run all checks with tox
	tox

update-api-doc:                 ## Generate API doc
	python docs/api_md_generate.py
