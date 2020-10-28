# Makespan PoC
Prototype impementations for [makespan](https://github.com/matyama/makespan) project.

Note that just some algorithms implemented here will be ported to the main Rust project. This repository generally serves to analyze the minimum makespan problem and compare different approaches (mainly in terms of solution quality, performance is of secondary concern).

## Environment setup
1. Download and install [Poetry](https://python-poetry.org/docs/#installation). E.g. `pipx install poetry`.
2. Inside project directory use Python 3.8 environment: `poetry env use python3.6`.
3. [Activate](https://python-poetry.org/docs/cli/#shell) project environment by running `poetry shell`.
4. [Install](https://python-poetry.org/docs/cli/#install) project dependencies by running `poetry install` or [update](https://python-poetry.org/docs/cli/#update) current ones with `poetry update`.
5. Install [pre-commit](https://pre-commit.com/) by running `pre-commit install`.

There is also a convenience make target:
```bash
make setup
```
which executes steps 2, 4 and 5.

## Additional libraries

### JupyterLab extensions
```bash
jupyter labextension install jupyterlab-plotly
jupyter labextension install @jupyter-widgets/jupyterlab-manager plotlywidget
```

### GLPK
To be able to use GLPK in MILP formulation, install following system libraries.
```bash
sudo apt install -y libglpk-dev glpk-utils glpk-doc
```

## Development
It is advised to use following make targets during development.

```bash
make fmt
```
1. Reformats imports using [isort](https://pycqa.github.io/isort/).
2. Runs [black](https://black.readthedocs.io/en/stable/) formatting tool on source code.

Again, find configuration for both tools in `pyproject.toml`.

```bash
make lint
```
This command runs:
1. Bunch of [flakehell](https://flakehell.readthedocs.io/) checks against code in `src/`. One can find config and tweak plugins in `pyproject.toml`.
2. Runs [mypy](https://mypy.readthedocs.io/en/latest/) static type check on `src/`. Configuration is in default `mypy.ini`.

```bash
make tests
```
Runs tests with doctests enabled and displays code coverage. Configuration can be tweaked in relevant sections of `pyproject.toml`.

All lints and tests are also available via single command:
```bash
make check
```

Finally, initial setup installs pre-commit hooks (see `.pre-commit-config.yaml`). One can run these hooks as follows:
```bash
pre-commit run --all-files
```
