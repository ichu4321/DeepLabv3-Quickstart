#!/bin/bash

# exit immediately if a command exits with a non-zero return status
set -e

# set pythonpath
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

# define vars

# program paths
CURRENT_DIR=$(pwd)
SCRIPT_DIR="deeplab"

# data paths
IMAGE_DIR="${CURRENT_DIR}/TrainingImages";
LABEL_DIR="${CURRENT_DIR}/TrainingLabels";
SPLITS_DIR="${CURRENT_DIR}/Splits";
OUTPUT_DIR="${CURRENT_DIR}/TFRecord";

# training paths
NUM_EPOCHS=1;
INIT_CHECKPOINT="${CURRENT_DIR}/BaseCheckpoint/base.ckpt";
CHECKPOINT_DIR="${CURRENT_DIR}/TrainingCheckpoints"

# NOTE: base.ckpt is the checkpoint downloaded from here:
# http://download.tensorflow.org/models/deeplabv3_pascal_train_aug_2018_01_04.tar.gz

# export paths
NUM_CLASSES=21
EXPORT_PATH="model.pb"

# create the train/val split lists
echo "Creating train/val splits..."
python3 "${SCRIPT_DIR}/train_val_splitter.py" ${IMAGE_DIR}

# convert the dataset to a TFRecord
echo "Converting dataset to TFRecords"
python3 "${SCRIPT_DIR}/convert_image_data.py" \
	--image_folder="${IMAGE_DIR}" \
	--label_folder="${LABEL_DIR}" \
	--splits_folder="${SPLITS_DIR}" \
	--image_format="jpg" \
	--label_format="png" \
	--output_dir="${OUTPUT_DIR}"

# delete all old training checkpoints
# (if you want to continue training)
# (copy the last checkpoint to ${INIT_CHECKPOINT}")
# (this needs both the .index and the .data files)
echo "Deleting old training data...";
if [ ! -z "$(ls -A ${CHECKPOINT_DIR})" ]; then
	rm ${CHECKPOINT_DIR}/*
fi

# train on data
echo "Training..."
python "${SCRIPT_DIR}/train.py" \
	--logtostderr \
	--train_split="trainval" \
	--model_variant="xception_65" \
	--atrous_rates=6 \
	--atrous_rates=12 \
	--atrous_rates=18 \
	--output_stride=16 \
	--decoder_output_stride=4 \
	--train_crop_size="513,513" \
	--train_batch_size=4 \
	--training_number_of_steps="${NUM_EPOCHS}" \
	--fine_tune_batch_norm=true \
	--tf_initial_checkpoint="${INIT_CHECKPOINT}" \
	--train_logdir="${CHECKPOINT_DIR}" \
	--dataset_dir="${OUTPUT_DIR}"

# export the model
echo "Exporting model to ${EXPORT_PATH}..."
python "${SCRIPT_DIR}/export_model.py" \
	--logtostderr \
	--checkpoint_dir="${CHECKPOINT_DIR}/" \
	--export_path="${EXPORT_PATH}" \
	--model_variant="xception_65" \
	--atrous_rates=6 \
	--atrous_rates=12 \
	--atrous_rates=18 \
	--output_stride=16 \
	--decoder_output_stride=4 \
	--num_classes="${NUM_CLASSES}" \
	--crop_size=513 \
	--crop_size=513 \
	--inference_scales=1.0


