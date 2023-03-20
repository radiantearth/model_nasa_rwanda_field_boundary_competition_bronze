# Borderline: a segmentation model for fields

In the [NASA Harvest Field Boundary Detection Challenge](https://zindi.africa/competitions/nasa-harvest-field-boundary-detection-challenge/leaderboard)
this was the third place solution by the team `borderline`.


## ML Model Documentation

Please review the model architecture, license, applicable spatial and temporal extents
and other details in the [model documentation](/docs/index.md).

## System Requirements

* Git client
* [Docker](https://www.docker.com/) with
    [Compose](https://docs.docker.com/compose/) v1.28 or newer.

## Hardware Requirements

|Inferencing|Training|
|-----------|--------|
|32 GB RAM | 32 GB RAM|

## Get Started With Inferencing

First clone this Git repository.

Please note: this repository uses
[Git Large File Support (LFS)](https://git-lfs.github.com/) to include the
model checkpoint file. Either install `git lfs` support for your git client,
use the official Mac or Windows GitHub client to clone this repository.

:zap: Shell commands have been tested with Linux and MacOS but will
differ on Windows, or depending on your environment.

```bash
git clone https://github.com/radiantearth/model_nasa_rwanda_field_boundary_competition_bronze.git
cd mmodel_nasa_rwanda_field_boundary_competition_bronze/
```

After cloning the model repository, you can use the Docker Compose runtime
files as described below.

## Pull or Build the Docker Image

Pull pre-built image from Docker Hub (recommended):

```bash
docker pull docker.io/radiantearth/mmodel_nasa_rwanda_field_boundary_competition_bronze:1
```

Or build image from source:

```bash
cd docker-services/
docker build -t radiantearth/model_nasa_rwanda_field_boundary_competition_bronze:1 .
```

## Run Model to Generate New Inferences

1. Prepare your input and output data folders:

    * The `data/input` folder in this repository contains some placeholder files to guide you.
    The input data should follow the following convention. It should be placed in a directory named
    `xxx_<tile_id>_<year>_<month>`,
    where `xxx` is arbitrary and `<tile_id>` represents the id of the tile stored in that directory.

    Here is a sample for reference.

    ```text
    data/input/nasa_rwanda_field_boundary_competition_source_test_00_2021_03
    data/input/nasa_rwanda_field_boundary_competition_source_test_00_2021_04
    data/input/nasa_rwanda_field_boundary_competition_source_test_00_2021_08
    data/input/nasa_rwanda_field_boundary_competition_source_test_00_2021_10
    data/input/nasa_rwanda_field_boundary_competition_source_test_00_2021_11
    data/input/nasa_rwanda_field_boundary_competition_source_test_00_2021_12
    data/input/nasa_rwanda_field_boundary_competition_source_test_01_2021_03
    data/input/nasa_rwanda_field_boundary_competition_source_test_01_2021_04
    data/input/nasa_rwanda_field_boundary_competition_source_test_01_2021_08
    data/input/nasa_rwanda_field_boundary_competition_source_test_01_2021_10
    data/input/nasa_rwanda_field_boundary_competition_source_test_01_2021_11
    data/input/nasa_rwanda_field_boundary_competition_source_test_01_2021_12
    ```

    These directories will contain tiff files for three tiles (id `00`,
    and `01`). 

    * The `output/` folder is where the model will write inferencing results.

2. Set `INPUT_DATA` and `OUTPUT_DATA` environment variables corresponding with
    your input and output folders. These commands will vary depending on operating
    system and command-line shell:

    ```bash
    # change paths to your actual input and output folders
    export INPUT_DATA="/home/my_user/model_nasa_rwanda_field_boundary_competition_bronze/data/input/"
    export OUTPUT_DATA="/home/my_user/model_nasa_rwanda_field_boundary_competition_bronze/data/output/"
    export MODELS_DIR="/home/my_user/model_nasa_rwanda_field_boundary_competition_bronze/models"
    export WORKSPACE_DIR="/home/my_user/model_nasa_rwanda_field_boundary_competition_bronze/workspace"
    ```

3. Run the appropriate Docker Compose command for your system:

    ```bash
    cd docker-services/
    docker compose up model_nasa_rwanda_field_boundary_competition_bronze_v1
    ```

4. Wait for the `docker compose` to finish running, then inspect the
`OUTPUT_DATA` folder for results.

## Understanding Output Data

Please review the model output format and other technical details in the [model
documentation](/docs/index.md).
