# ProbML Sandbox
Sandbox Enviroment for the ProbML Toolset

## Description

This is a sandbox environment for the ProbML toolset. It is a place to experiment with the ProbML toolset and to develop new features. The folder structure is as follows:

- `data/`: Contains the datasets and data generation scripts.
- `experiments/`: Contains subfolders for the experiments. Within the experiment subfolders the setup of the experiment is free and can include everything from `.py` over conifguration files, `Dockerfile`s to streamlit apps.
- `ipynb/`: Contains the Jupyter notebooks for the evaluation of experiments. The notebooks should either have a common prefix or be placed in a subfolder.
- `module_sandbox/`: Contains the sandbox for the ProbML package. This is the place to develop new features. When proven to be useful, the features can be moved to the [ProbML toolbox](https://github.com/lisa-wm/probabilisticml).
- `paper_{paperid}`: Contains code for the paper with the id `{paperid}`. The code likely aggregates results from multiple experiments and is used to generate the figures and tables for the paper.
- `results/`: Contains the results of the experiments. The results are stored in subfolders corresponding to the experiments. The results are stored in a format that is easy to read and process. This folder is gitignored.
- `tests/`: Contains tests for the modules in the `module_sandbox` for the ProbML package.

Important top level files:
- `pyproject.toml`,`poetry.lock`: Details on the dependency management below.
- `pre-commit-config.yaml`: Contains the configuration for the pre-commit hooks.
- Docker Compose files: The Docker Compose files are used to run the experiments in a containerized environment. The Docker Compose files are named after the experiment they are used for in the format `docker-compose-{experiment}.yml`.
- `.env`: Contains the environment variables for the Docker Compose files. (Template: `.env.template`)

## Setup

Top level requirements are managed with `poetry`.

1. You need `poetry` installed. The best practice is to install poetry only once globally and make it accessable to any virtual environment. It can be done using [`pipx`](https://pipx.pypa.io/stable/installation/) (it is very easy to install 2 commands)

```shell
pipx install poetry
```

2. Then you just execute `poetry install` to install dependancies locked in `poetry.lock`
3. To extend dependancies for example add jax you execute: `poetry add jax` or specify version `poetry add jax==0.4.25` this automatically updates .lock and .toml file
4. `poetry remove jax` removes jax dep and so on.. [more info](https://python-poetry.org/docs/managing-dependencies/)
5. It automatically handles cross-platform problems and you can even define [platform-specific dependancies](https://python-poetry.org/docs/dependency-specification/#combining-git--url--path-dependencies-with-source-repositories)

Activate the pre-commit hooks: `pre-commit install`

> Make sure to have Docker and Docker Compose installed for some experiments. Therein currently `pip-tools` is used to manage the requirements. To update the local `requirements.txt` file install `pip install pip-tools` and then run `pip-compile` from the experiment folder after adding the new package to the experiment specific `requirements.in` file. If you set up a new experiment with Docker and `poetry` let us know so we can update the instructions.


# Configuring Experiments
For configuring experiments we advise you to use our configuration framework:
[readme](module_sandbox/config/README.md)


# Running Experiments on linux compute server

Our compute servers contains a pre-built docker image built on top of the Nvidia NGC JAX image. It additionally has our dependencies installed. The image is named `probabilisticml`.

- Since we expect the code to change frequently we propose the following workflow:

    1. Create your own directory in the compute server. For example: `mkdir -p /home/username/`
    2. Clone repository in your new directory. `git clone ...`
    3. Mount your directory to the docker container and run experiments. You can use `run.sh` as a good starting point:

```shell
docker run \
    --gpus device=1 \
    --mount type=bind,src="$(pwd)",target=/app/probabilisticml \
    --workdir /app/probabilisticml \
    --detach \
    --rm \
    probabilisticml \
    bash -c "poetry install && python3 -m experiments.bnn_llms.train_bde &> run.log"


# --gpus flag is used to specify visible devices: `all`: all GPUs, `device=1`: GPU 1, `device=0,1`: GPU 0 and 1, ...
# --mount flag is used to mount directory in the container: `type=bind,src="$(pwd)",target=/app/probabilisticml`: mount current directory to /app/probabilisticml in the container
# --workdir sets the working directory in the container: `/app/probabilisticml`
# --detach : run container in background
# --rm : remove container after it exits
# --env : set environment variables
# `probabilisticml` : image name
# bash -c "python3 -c 'import os; print(os.listdir())' > test.log": command to run in the container
# `&> run.log` : redirect stdout and stderr to run.log
```
After configuring the `run.sh` script, you can run the script using `sh run.sh`. The script will run the experiment in the background and log the output to `run.log`.

- To check cuda device usage: `nvidia-smi`
- To check the status of the running container: `docker ps`
- To check the logs see the log file: `tail run.log`
- To stop the container: `docker stop <container_id>`
