.PHONY: check fmt lint tests

setup:
	poetry env use python3.8
	poetry install
	pre-commit install

fmt:
	isort .
	black .

lint:
	flakehell lint
	MYPYPATH=src mypy .

tests:
	PYTHONPATH=src pytest

check: lint tests
