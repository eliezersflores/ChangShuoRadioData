# YOLO Training Pipeline for ChangShuoRadioData

This fork was developed by **Prof. Eliezer Soares Flores** ([Google Scholar Profile](https://scholar.google.com.br/citations?user=3jNjADcAAAAJ&hl=pt-BR&oi=ao)) with the goal of generating a smaller version of the **ChangShuoRadioData** dataset, converting it into suitable annotation format, and training and testing **YOLO** models on the resulting data.

If you have any questions, please feel free to contact me at `eliezersflores@gmail.com`.

This work, as detailed below, was based on the stable version of the **ChangShuoRadioData** repository, available at:

- `https://github.com/Singingkettle/ChangShuoRadioData/tree/a6d09a4b264894b76f852ce33bfd82adc7b270b5`

## 1. Customization

After cloning the stable version of the repository, the following configuration file was modified:

`config/_base_/simulate/ChangShuo/CSRD2025.json`

Specifically, the parameter `NumFrames` was changed from `100000000` to `10000`.

This change was necessary to make the simulation feasible for local experimentation.

## 2. Execution order

The pipeline used in this project follows the sequence below.

All scripts should be executed from the `tools` subdirectory.

### Step 1 — Generate raw CSRD data

Run:

`simulation.m`

For faster execution, use:

`multi_simulation.sh`

This step generates the raw simulated data based on the CSRD configuration.

### Step 2 — Convert CSRD output to COCO format

Run:

`convert_csrd_to_coco.m`

**Important:** this script currently includes a limit of at most `10,000` frames.

### Step 3 — Convert COCO annotations to YOLO format

Run:

`convert_coco_yolo.m`

This script converts the generated COCO-style dataset into the directory and annotation structure expected by YOLO. It also uses the auxiliary functions `build_class_map.m` and `save_yolo_yaml.m`, both developed by **Prof. Eliezer Soares Flores**.

### Step 4 — Train the YOLO model

Run:

`train_yolo.py`

This step trains the **YOLO** model using the dataset generated in the previous steps. The dataset structure and class definitions are read from the corresponding `config_yolo_*.yaml` file, which is generated during Step 3 by `convert_coco_yolo.m`. In addition, the Conda environment used in these experiments is documented in `environment.yml`.

**Note:** the file `environment.yml` is provided to help reproduce the **Conda environment** used for training.
