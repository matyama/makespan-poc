.PHONY: check fmt lint tests

setup:
	sudo apt install -y libglpk-dev glpk-utils glpk-doc
	poetry env use python3.8
	poetry install
	poetry run pre-commit install

fmt:
	isort .
	black .

lint:
	flakehell lint
	MYPYPATH=src mypy .

tests:
	PYTHONPATH=src pytest

check: lint tests
