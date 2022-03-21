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

## All Systems
  * Download the model checkpoint from Tensorflow's model zoo
    - http://download.tensorflow.org/models/deeplabv3_pascal_train_aug_2018_01_04.tar.gz
  * extract the files
    - Windows users will need 7zip (https://www.7-zip.org) to extract
  * there should be these files in the extracted folder
    - frozen_inference_graph.pb
    - model.ckpt.data-00000-of-00001
    - model.ckpt.index
  * rename and move these files to the BaseCheckpoint folder
    - "model.ckpt.data-00000-of-00001" renamed "base.ckpt.data-00000-of-00001"
    - "model.ckpt.index" renamed "base.ckpt.index"
  * rename and move this file to the base directory of the repository (same location as the TRAIN scripts)
    - "frozen_inference_graph.pb" renamed "model.pb"

# Use
## Demo with Camera
  * run "python DEEPLAB_DEMO.py" to test out model.pb live (uses the first available camera)
    - WARNING: this may be very slow
    - if the console window is printing allocation errors, try closing other programs to free up RAM

## Training
  * copy images into the TrainingImages folder
  * copy labels in to the TrainingLabels folder
  * run "bash TRAIN.sh" (Linux) or "TRAIN.bat" (Windows)
    * NOTE: you should change the NUM_EPOCHS and NUM_CLASSES variables in the scripts
      * NUM_EPOCHS tells the script how many rounds of training you want (checkpoints are saved every 10 epochs)
      * NUM_CLASSES is the number of different classes you have (including a background class)
    * creates/overwrites "model.pb" after it finishes training

# TODO
  * add instructions for training/running with GPU (CUDA and CuDNN installation)
  * add instructions for converting to an ONNX model
  * add detailed instruction for creating Pascal-styled labels