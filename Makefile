fix-lint:
	black pandas_toolkit setup.py
	isort --lines 120 --recursive --use-parentheses --trailing-comma pandas_toolkit setup.py

lint:
	tox lint

test-all:
	tox