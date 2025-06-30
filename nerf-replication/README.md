# Nerf-Replication

Codebase for replicating the paper NeRF.

## Setup

```sh
# Create conda environment
conda create -n nerf-rep python=3.10
conda activate nerf-rep

# Install PyTorch according to your environment
# Be sure you have CUDA installed, we use CUDA 12.1 in our experiments
# NOTE: you need to make sure the CUDA version used to compile the torch is the same as the version you installed
# Use `nvcc -V` to check the CUDA version
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install -r requirements.txt
```

## Data preparation

Download NeRF synthetic dataset and add a link to the data directory. After preparation, you should have the following directory structure:
We have provied the minimal `lego` dataset for you in the [Github release page](https://github.com/pengsida/project_page_assets/releases/download/nerf-replication/lego.zip).

```
data/nerf_synthetic
|-- chair
|   |-- test
|   |-- train
|   |-- val
|-- drums
|   |-- test
......
```

## NeRF Replication

### Configurations

The configuration file are stored in `configs/nerf`, which contains the parameters for replicating NeRF.
You can adjust the parameters according to your preference.

Parameter description:
```yaml
task: "nerf_replication"
gpus: [1] # set gpu device number
exp_name: "nerf"
scene: "lego"
trained_model_dir: "data/trained_model"

# set path for each modules
train_dataset_module: src.datasets.nerf.blender
test_dataset_module: src.datasets.nerf.blender
network_module: src.models.nerf.network
renderer_module: src.models.nerf.renderer.volume_renderer
loss_module: src.train.trainers.nerf
evaluator_module: src.evaluators.nerf

...
```

The default parameters can be found in `src/config/config.py`.

### Dataset

Dataset file path: `src/datasets/nerf/blender.py`

You need to implement the dataset class, which is responsible for loading the data and providing it to the model.
The core functions include: `__init__`, `__getitem__`, and `__len__`.
- `__init__`: This function is responsible for loading the specified format file from disk, calculating and storing it in a specific form.
- `__getitem__`: This function is responsible for providing the input required for training and the ground truth output to the model at runtime. For example, for NeRF, it provides 1024 rays and 1024 RGB values.
- `__len__`: This function returns the number of training or testing samples. The index value obtained from `__getitem__` is usually in the range [0, len-1].

After implementing the dataset class, you can use the following command to debug it:

```sh
python run.py --type dataset --cfg_file configs/nerf/lego.yaml
```

### Model

> [!CAUTION]
> This module has been implemented, don't modify the model file.

Model file path: `src/models/nerf/network.py`

The model module is responsible for defining the model structure and the forward propagation process.
The core functions include: `__init__`, `forward`.
- `__init__`: This function is responsible for defining the model structure, including the number of layers, the number of neurons in each layer, and the activation function.
- `forward`: This function is responsible for the forward propagation process, which includes the input of the dataset and the output of the model.
For NeRF, the input is a batch of rays, and the output is a batch of RGB values.

After implementing the network class, you can use the following command to debug it:
```sh
python run.py --type network --cfg_file configs/nerf/lego.yaml
```

### Renderer

Renderer file path: `src/models/nerf/renderer/volume_renderer.py`

The renderer module is responsible for rendering the output of the model.
The core functions include: `__init__`, `render`.
- `__init__`: This function is responsible for defining the rendering parameters, including the number of samples, the step size, and the background color.
- `render`: This function is responsible for rendering the output of the model, which includes the RGB values and the depth values.
The detailed rendering process is described in the paper.

### Trainer and Evaluator

Trainer file path: `src/train/trainers/nerf.py`
Evaluator file path: `src/evaluators/nerf.py`

The trainer module is responsible for training the model, including the loss function and the optimizer.
The evaluator module is responsible for evaluating the model, including the PSNR and SSIM metrics.
This two modules are simple and easy to implement, so here we only provide the file path.

```sh
python train.py --cfg_file configs/nerf/lego.yaml
```

```sh
python run.py --type evaluate --cfg_file configs/nerf/lego.yaml
```

### Inference

> [!CAUTION]
> Don't modify the model file.

Considering that you are limited by the GPU memory, we provide a pretrained model for you to test the inference process.
You can download the pretrained model from the [Github release page](https://github.com/pengsida/project_page_assets/releases/download/nerf-replication/latest.pth) and put in in the `data/trained_model` directory.

The pretrained model path should be like this: `data/trained_model/task_name/scene_name/exp_name/latest.pth`.

Then, run the following command to test the inference process:
```sh
python run.py --type evaluate --cfg_file configs/nerf/lego.yaml
```
