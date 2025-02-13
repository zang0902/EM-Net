# EM-Net

The official code for "EM-Net: Efficient Channel and Frequency Learning with Mamba for 3D Medical Image Segmentation" ![GitHub Repo stars](https://img.shields.io/github/stars/zang0902/EM-Net) [![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fzang0902%2FEM-Net&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com) 

Our paper is available at [arXiv](https://arxiv.org/abs/2409.17675) or [MICCAI 2024](https://papers.miccai.org/miccai-2024/paper/1923_paper.pdf).

This is an efficient 3D medical image segmentation method that balances high spatial modeling performance with moderate computational costs. Well-designed channel and spatial Mamba blocks (CSRM blocks) and efficient frequency and spatial learning Mamba blocks (CSRM-F blocks) contribute to this framework (i.e., EM-Net).

## Installation

1. First, clone this repository and create the environment.

```bash
git clone https://github.com/zang0902/EM-Net.git
cd EM-Net
conda create -n EM-Net python=3.10.13
conda activate EM-Net
```

Note: Before proceeding to the next step, please ensure that CUDA-11.8 is installed and its path has been added to the environment variable.

```bash
# Check CUDA version
nvcc -V  # Should return CUDA version 11.8
```

2. Install dependencies and the `mamba` package.

```bash
# Install PyTorch with CUDA 11.8
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt

# Install causal-conv1d
cd causal-conv1d
python setup.py install

# Install mamba
cd ../mamba
pip install -e .
```

## Usage

Main directories and files are listed below:

```
.
├── configs  # Training-related configurations
│   ├── datasets  # Dataset configurations
│   ├── models  # Model configurations
│   └── trainers  # Training hyperparameters
├── models  
│   └── em_net_model.py  # EM-Net model
├── networks  # Comparison models
│   ├── ...
│   └── unetr.py
├── runs  # Logs and checkpoints save root
├── scripts  # Training/testing/parameter calculation scripts
│   ├── ...
│   ├── calc_params_flops.sh
│   ├── test_synapse.sh
│   └── train_synapse.sh
├── train.py  
├── test.py  
└── main.py  
```

It's easy to train and test using the provided scripts and configurations.

```bash
# Example command to run the script
sh scripts/train_synapse.sh
# or
sh scripts/test_synapse.sh
```

## Overview

![Framework](./assets/framework.png "The EM-Net framework")
![Motivation](./assets/motivation.png "The EM-Net layers motivation")
Our proposed framework, including two different blocks—the CSRM block and CSRM-F block—is designed to extract the spatial relationships in channel view and combine channel and frequency modeling capabilities. The motivation is that Mamba excels in capturing long sequence information, and frequency information is more suitable for capturing spatial relationships.

![Method](./assets/layer.png "The EM-Net layer architecture")

- CSRM Layer:
  Calibrates and highlights relevant regional features.
- EFL Layer:
  Learns global and local features by leveraging the frequency domain.

![Results](./assets/result.png "The EM-Net visualization results")
EM-Net not only performs well in the segmentation of large organs but also shows more significant improvements in the segmentation of small organs.
![Results](./assets/comparison.png "The EM-Net quantitative results")

![Results](./assets/ablation.png "The EM-Net ablation study results")
Due to the high efficiency of the EM-Net framework, it is able to achieve a high level of accuracy with a relatively small number of parameters, which is particularly advantageous for resource-constrained applications.

## Acknowledgment

We sincerely acknowledge the following projects that provided valuable insights and resources:

- https://github.com/Project-MONAI/research-contributions/tree/main/UNETR/BTCV
- https://github.com/ge-xing/SegMamba
- https://github.com/Dao-AILab/causal-conv1d
- https://github.com/state-spaces/mamba
- https://github.com/Project-MONAI/MONAI
- https://github.com/Amshaker/unetr_plus_plus/
- https://github.com/MrYxJ/calculate-flops.pytorch

## Citation

If you find this work helpful for your research, we would appreciate it if you cite our paper:

```bibtex
@inproceedings{chang2024net,
  title={EM-Net: Efficient Channel and Frequency Learning with Mamba for 3D Medical Image Segmentation},
  author={Chang, Ao and Zeng, Jiajun and Huang, Ruobing and Ni, Dong},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={266--275},
  year={2024},
  organization={Springer}
}
```

## License

This project is licensed under the [MIT License](LICENSE).

## Contact Us

For any questions or suggestions, please contact us at:

- 2200241009 [at] email [dot] szu [dot] edu [dot] cn
- im [dot] jiajun [dot] zeng [at] gmail [dot] com

Or submit an issue on GitHub.

If you find this project useful, please give it a star!
