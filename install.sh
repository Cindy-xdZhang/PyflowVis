#!/usr/bin/env bash
module purge
module load cuda/11.8


conda deactivate
conda env remove --name deepvortex
conda create -n deepvortex  -y python=3.12.3 numpy 
conda activate deepvortex


# please always double check installation for pytorch and torch-scatter from the official documentation
conda install -y pytorch-cuda=11.8  torchvision cudatoolkit=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
