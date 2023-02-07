# EM_Reg
The PyTorch implementation for paper:

Electron Microscopy Image Registration using Correlation Volume (ISBI 2023)

![figure](https://user-images.githubusercontent.com/87810937/217230931-7f92c13b-42c6-4f8d-bc01-ab0c2dd3c287.jpg)
(a) Illustration of our proposed registration method. Given two consecutive EM images, \textit{i.e.}, the fixed image and the moving image (to be registered), we obtain feature pyramids for both images through feature extraction and crossed spatial attention. (b) Illustration of feature extraction and crossed spatial attention. (c) Illustration of DFRE. At each level of the pyramids, we construct a correlation volume between features and then update the displacement field using DFRE.

# Requirements
Our model is implemented in the following environments：
- Python 3.8
- PyTorch 1.11
- CUDA 11.3

1. To install all the required packages, you can use the following command:

`pip install -r requirements.txt`

2. You can also use docker to easily set up a virtual environment with the right package versions. Our docker image can be pulled as following:

`docker pull registry.cn-hangzhou.aliyuncs.com/liuxzzz/em_registration:v4`

# Data
We acquire the experimental data by synthesizing the deformation field and generating the warped images manually. Three datasets are used in our experiments: CREMI, FAFB, and FIB25. All the three datasets are open source and you can easily download them if needed:
- CREMI: [https://cremi.org/data](https://cremi.org/data)
- FAFB: [https://temca2data.org/data.html](https://temca2data.org/data.html)
- FIB25 [https://www.janelia.org/tools-and-data-release](https://www.janelia.org/tools-and-data-release)

To add deformation to the EM images and simulate unregistered image pairs, you can run:

`python dvf_generation/create_deformation.py`

Adjust the parameter `flitter` to synthesize deformation of different sizes.

# Train
Run the script `train.py` to start training. For example:

`python train.py  --optim cosine --data cremi --label --batchsize 2`

Adjust the necessary parameters according to your training data as well as the hardware equipment.

# Acknowledgment
This code is partially borrowed from [RAFT](https://github.com/princeton-vl/RAFT). 
