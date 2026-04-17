# Final Assignment: Beyond Peak Performance  

## Semantic Segmentation for Autonomous Driving on Cityscapes
This repository contains the full implementation for the final project of the Neural Networks for Computer Vision course. The project investigates semantic segmentation within urban environments, focusing on achieving a balance between high-accuracy models, deployable efficiency, and safety through Out-of-Distribution (OOD) detection.  



## Repository Structure  

The repository is organized into distinct folders, each corresponding to a specific benchmark or model developed during the project:

- `baseline/`: Contains the U-Net architecture used as the initial performance baseline. This model operates on $256 \times 256$ resolution images.

- `peak/`: Contains the DeepLabV3+ model with a ResNet-50 backbone. This folder contains the code for achieving the highest segmentation accuracy using $1024 \times 512$ resolution and Atrous Spatial Pyramid Pooling (ASPP).

- `efficiency/`: Contains the MobileNetV3-Large implementation. It includes the Knowledge Distillation scripts used to train this lightweight "student" model from the "teacher" peak model.

- `ood/`: Includes the scripts for binary anomaly detection. It contains the Flow Matching (FM) generative model and the predictive entropy baseline used to identify images outside the Cityscapes distribution.

Each folder includes all necessary code for training, testing, and evaluation (including `train.py` and `predict.py` scripts) specific to that model.


## Dataset Preparation

The models are trained using the Cityscapes fine annotations split (2,975 training, 500 validation, and 1,525 test images). Follow the provided `download_docker_and_data.sh` script and `README-Installation.md` to fetch the Cityscapes data.

The OOD benchmark evaluates the system's ability to identify samples from the Common Objects in Context (COCO) dataset as anomalies.
1. Download the 2017 Val images from the [official COCO website](https://cocodataset.org/#download).
2. Unzip the images into your data directory:
```bash
   wget http://images.cocodataset.org/zips/val2017.zip
   unzip val2017.zip -d ./data/coco
```


## Installation and Requirements 
To ensure reproducibility and manage dependencies, every model folder includes a custom `Dockerfile`. This allows you to generate a consistent environment with all required libraries without manual installation.

### Installation Steps
1. **Clone the repository**
```bash
   git clone https://github.com/FrancescTUe/NNCV.git
   cd NNCV
```
2. **Build the Docker Image by navigating to the desired model folder**
```bash
   cd baseline_model
   docker build -t nncv-baseline .
```

### Training and testing the models
Each folder contains a `train.py` and `predict.py` scripts. Use the following steps to execute them within a container:
1. **Build**
```docker build -t nncv-[folder] .``` 
2. **Train**
```docker run --gpus all -v /path/to/data:/data nncv-[folder] python train.py```
4. **Test**
```docker run --gpus all -v /path/to/data:/data nncv-[folder] --checkpoint model.pt```
