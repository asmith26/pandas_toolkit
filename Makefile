deploy-doc: update-api-doc
	mkdocs gh-deploy

fix-lint:
	black --line-length=120 pandas_toolkit setup.py
	isort --lines 120 --recursive --use-parentheses pandas_toolkit setup.py

lint:
	tox lint

test-all:
	tox

update-api-doc:
	python docs/api_md_generate.py
