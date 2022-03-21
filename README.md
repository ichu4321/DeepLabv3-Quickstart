# DeepLabv3-Quickstart
This is a simplified shortcut for training and using the DeepLabv3 network

(from tensorflow's repo: https://github.com/tensorflow/models/tree/master/research/deeplab)

This repo aims to be as straightforward as possible with clear instructions for environment setup and use

# Environment Setup
## Untested OS'es
I don't have a Mac so I can't verify any installation steps. 

However, the Linux install instructions should be close to what you'd need to do for Mac.

The Linux instructions were tested with Ubuntu 18.04. They should work with other distros, but they may require some minor tweaking.

## Windows
  * install Anaconda3 (https://www.anaconda.com/products/individual)
  * open up the Anaconda3 command prompt terminal and enter these commands:
    - "conda create --name deep"
    - "conda activate deep"
    - "conda install python=3.6"
    - "pip install -r requirements_python36_windows10.txt"

## Linux
  * install Python 3.6
  * open up a terminal window and "cd" to the repo folder
  * enter these commands:
    - "python3 venv ~/deep"
    - "source ~/deep/bin/activate"
    - "pip install --upgrade pip"
    - "pip install -r requirements_python36_ubuntu18.txt"

# Use
  * copy images into the TrainingImages folder
  * copy labels in to the TrainingLabels folder
  * run "bash TRAIN.sh" (Linux) or "TRAIN.bat" (Windows)
    - creates "model.pb" after it finishes training
  * run "python DEEPLAB_DEMO.py" to test it out live (uses the first available camera)


