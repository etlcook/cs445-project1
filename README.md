# Project 1 for CS-445

## To setup the python environment and run the code

#### Install python Poetry
```bash
python3 -m pip install --user pipx
python3 -m pipx ensurepath
pipx install poetry
poetry --version # To check installation
```

#### Install dependencies for this repo
```bash
# After cd'ing into the repo:
poetry install
```

#### Help Poetry environment kernel to be useable from jupyter notebook
`poetry run python -m ipykernel install --user --name my-poetry-env --display-name "My Poetry Env"`

#### Run jupyter lab through poetry to use the installed dependencies:
`poetry run jupyter lab`
Then just select the "my-poetry-env" python kernel made earlier, and everything will run fine.