# Named Entity Recognition

## Setup
Using [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/):

```bash
# to create the environement:
conda env create -f environment.yml
```

```bash
# in case you install new packages you have to update the environment:
# NOTE: delete the prefix at the end, might cause issues on different environments
conda env export > environment.yml
```

```bash
# !!! assuming the workdir is the repository path.

# to run a file:
PYTHONPATH=. python <path_to_file>
# e.g.
PYTHONPATH=. python train/crosloeng.py
```
