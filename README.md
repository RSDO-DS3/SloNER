# SloNER - Named Entity Recognition for Slovene language

V tem repozitoriju se nahaja rezultat aktivnosti A3.2 - R3.2.1 Orodje za prepoznavanje imenskih entitet, ki je nastalo v okviru projekta [Razvoj slovenščine v digitalnem okolju](https://slovenscina.eu).

---

## 1. Training

### 1.1. Environment setup (local)

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

### 1.2. Environment setup (DGX A100)

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

## 2. Inference

### 2.2. Named Entity Recognition API

Named Entity Recognition can be run as a standalone service which exposes a RESTful API. It accepts an arbitrary text and returns annotated text, word by word, with recognised named entities.

#### 2.2.1. Docker Container

##### Build

To build a model serving Docker container, from the project's root folder, execute

```bash
$ bin/docker-build-api.sh
```

Upon a sucessfull build, the resulting container image is named `rsdo-ds3-ner-api:v1`. By default gpu images are used. For CPU-only images use commands with name ending `-cpu`.

##### Run

The resulting Docker container image from the above build action **does not include** a trained Named Entity Recognition model; therefore, it has to be mounted as a Docker volume. The container expects a model to be mounted into a path, defined by `NER_MODEL_PATH` environment variable.

A build model is available on [CJVT NAS](https://nas.cjvt.si/index.php/f/2246207) (access needed).

Save a trained model into some directory, e.g., `/data/models/bert-based`. It should be stored into a directory, following the same file structure as the models on the HuggingFace repository.

Edit `bin/docker-run-api.sh` accordingly, to point to the path of your trained model.

To run a model serving Docker container, from the project's root folder, execute

```bash
$ bin/docker-run-api.sh
```

The container is named `rsdo-ds3-ner-api`. Inspect with Docker logs to find out when the respective Flask server starts listening and accepting HTTP requests.

Then fire up your web browser and navigate to `http://localhost:5000/apidocs`. A Swagger's UI should show up where one can explore all the exposed endpoints.

## 3. Results
### 3.1 Experimentation setup

The SloNER model was trained on the [SUK 1.0 corpus](https://www.clarin.si/repository/xmlui/handle/11356/1747). We present the results for 5 different versions of the model, that differ in the pretrained language model that was used. The beset performance was achieved using the [sloberta-2.0](https://www.clarin.si/repository/xmlui/handle/11356/1397) pretrained model.

| **model_name**                 | **precision_score** | **recall_score** | **f1_score** |
|--------------------------------|---------------------|------------------|--------------|
| cro-slo-eng-bert               |                0,91 |             0,93 |         0,92 |
| bert-base-multilingual-cased   |                0,90 |             0,93 |         0,91 |
| bert-base-multilingual-uncased |                0,60 |             0,63 |         0,61 |
| sloberta-1.0                   |                0,91 |             0,94 |         0,93 |
| sloberta-2.0                   |                0,91 |             0,94 |         0,93 |

---

> Operacijo Razvoj slovenščine v digitalnem okolju sofinancirata Republika Slovenija in Evropska unija iz Evropskega sklada za regionalni razvoj. Operacija se izvaja v okviru Operativnega programa za izvajanje evropske kohezijske politike v obdobju 2014-2020.

![](Logo_EKP_sklad_za_regionalni_razvoj_SLO_slogan.jpg)
