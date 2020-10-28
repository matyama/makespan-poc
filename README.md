# Makespan PoC
Prototype impementations for [makespan](https://github.com/matyama/makespan) project. 

Note that just some algorithms implemented here will be ported to the main Rust project. This repository generally serves to analyze the minimum makespan problem and compare different approaches (mainly in terms of solution quality, performance is of secondary concern).

## Environment setup
1. Download and install [Poetry](https://python-poetry.org/docs/#installation). E.g. `pipx install poetry`.
2. Inside project directory use Python 3.8 environment: `poetry env use python3.6`.
3. [Activate](https://python-poetry.org/docs/cli/#shell) project environment by running `poetry shell`.
4. [Install](https://python-poetry.org/docs/cli/#install) project dependencies by running `poetry install` or [update](https://python-poetry.org/docs/cli/#update) current ones with `poetry update`.

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
