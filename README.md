# Named Entity Recognition

## Setup
Create a new environment using [venv](https://docs.python.org/3/library/venv.html).
Make sure it is a python 3.8 environment:
```bash
# to create the environement:
python3 -m venv NER-env
```
Activate the environment:
```bash
source NER-env/bin/activate
```

Install all the dependendencies
```bash
# in case you install new packages you have to update the environment:
# NOTE: delete the prefix at the end, might cause issues on different environments
python install -r requirements.txt
```

Or just do it via your IDE, e.g. PyCharm, the point is to install the dependencies from `requirements.txt` :)

Running the files:
```bash
# !!! assuming the workdir is the repository path.

# to run a file:
PYTHONPATH=. python <path_to_file>
# e.g.
PYTHONPATH=. python train/crosloeng.py
```

## Running on the DGX A100 cluster

Before you begin, make sure you have the code from the repository on the cluster.
Once there, you can run the [SloNERT](./src/train/crosloeng.py) by runnig:

```bash
./bin/run.sh
```
Which will setup the environment for you, and run the training, and testing of the model.
The tasks are dependent and will execute consecutevely via [SLURM](https://slurm.schedmd.com/) using the [`sbatch` command](https://slurm.schedmd.com/sbatch.html).

Should you wish to have an interactive environment, for debugging and online developing, run:

```bash
./bin/run-container.sh
```
Which will start the created container with an interactive `bash` console.

Should you wish to create a new container, run the following command:

```bash
CONTAINER_IMAGE_PATH=/path/where/to/store/your/image
srun \
    --container-image pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime \  # the source docker image, we use the pytorch one, with all pytorch and CUDA requirements pre-installed
    --container-save "$CONTAINER_IMAGE_PATH" \  # where to store the created image
    --container-mounts "$PWD":/workspace,/some/dir:/mount/point \  # choose which directories to mount in the container
    --container-entrypoint /workspace/bin/exec-setup.sh # choose which script to be executed when the container is created.
    # alternatively, you can replace the last line with:
    --pty bash -l # if you wish to have an interactive bash shell
```

For more examples, please inspect the scripts within the [`bin`](./bin/) directory named with the `run-*.sh` pattern.
