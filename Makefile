bump2version-patch: test-all
	bump2version patch

bump2version-minor: test-all
	bump2version minor

bump2version-major: test-all
	bump2version major

clean-test:
	rm -f .coverage
	rm -rf .mypy_cache
	rm -rf .tox
	rm -rf *.egg-info

coverage:
	coverage run -m nose2 -v
	coverage report
	coverage html
	firefox htmlcov/index.html

deploy-doc: update-api-doc
	mkdocs gh-deploy

fix-lint:
	black --line-length=120 pandas_toolkit setup.py
	isort --lines 120 --recursive --use-parentheses pandas_toolkit setup.py

lint:
	tox -e lint

test-all: clean-test
	tox

update-api-doc:
	python docs/api_md_generate.py
