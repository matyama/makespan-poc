.PHONY: fmt fmt-check lint tests type-check

setup:
	poetry env use python3.8
	poetry install
	pre-commit install

fmt:
	@echo ">>> Formatting sources using black"
	black .

isort:
	@echo ">>> Formatting imports using isort"
	isort .

lint:
	@echo ">>> Linting Python sources with flakehell"
	flakehell lint || ( echo ">>> Linting failed"; exit 1; )

type-check:
	@echo ">>> Checking consistency of type annotations with mypy"
	MYPYPATH=src mypy . || ( echo ">>> Type check failed"; exit 1; )

check: lint type-check

tests:
	@echo ">>> Running tests with coverage"
	PYTHONPATH=src pytest --cov=src --verbose --doctest-modules .

release-check: check tests
