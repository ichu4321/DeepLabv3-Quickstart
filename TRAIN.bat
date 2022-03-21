@ECHO OFF

rem set pythonpath
echo "%PYTHONPATH%"
rem set PYTHONPATH=%PYTHONPATH%:%cd%:%cd%\slim
set PYTHONPATH=%cd%;%cd%\slim

rem define vars

rem program paths
set CURRENT_DIR=%cd%
set SCRIPT_DIR=deeplab

rem data paths
set IMAGE_DIR=%CURRENT_DIR%\TrainingImages
set LABEL_DIR=%CURRENT_DIR%\TrainingLabels
set SPLITS_DIR=%CURRENT_DIR%\Splits
set OUTPUT_DIR=%CURRENT_DIR%\TFRecord

rem training paths
set NUM_EPOCHS=1
set INIT_CHECKPOINT=%CURRENT_DIR%\BaseCheckpoint\base.ckpt
set CHECKPOINT_DIR=%CURRENT_DIR%\TrainingCheckpoints

rem NOTE: base.ckpt is the checkpoint downloaded from here:
rem http://download.tensorflow.org/models/deeplabv3_pascal_train_aug_2018_01_04.tar.gz

rem export paths
set NUM_CLASSES=21
set EXPORT_PATH=model.pb

rem create the train/val split lists
echo "Creating train/val splits..."
python %SCRIPT_DIR%/train_val_splitter.py %IMAGE_DIR%

rem convert the dataset to a TFRecord
echo "Converting dataset to TFRecords"
python %SCRIPT_DIR%/convert_image_data.py ^
--image_folder=%IMAGE_DIR% ^
--label_folder=%LABEL_DIR% ^
--splits_folder=%SPLITS_DIR% ^
--image_format=jpg ^
--label_format=png ^
--output_dir=%OUTPUT_DIR%

rem delete all old training checkpoints
rem if you want to continue training
rem copy the last checkpoint to ${INIT_CHECKPOINT}"
rem this needs both the .index and the .data files
echo "Deleting old training data..."
dir /A /B "%CHECKPOINT_DIR%" | findstr /R ".">NULL && del /q "%CHECKPOINT_DIR%\*.*"

rem train on data
echo "Training..."
python %SCRIPT_DIR%/train.py ^
--logtostderr ^
--train_split=trainval ^
--model_variant=xception_65 ^
--atrous_rates=6 ^
--atrous_rates=12 ^
--atrous_rates=18 ^
--output_stride=16 ^
--decoder_output_stride=4 ^
--train_crop_size=513,513 ^
--train_batch_size=4 ^
--training_number_of_steps=%NUM_EPOCHS% ^
--fine_tune_batch_norm=true ^
--tf_initial_checkpoint=%INIT_CHECKPOINT% ^
--train_logdir=%CHECKPOINT_DIR% ^
--dataset_dir=%OUTPUT_DIR%

rem export the model
echo "Exporting model to %EXPORT_PATH%..."
python %SCRIPT_DIR%/export_model.py ^
--logtostderr ^
--checkpoint_dir=%CHECKPOINT_DIR%\ ^
--export_path=%EXPORT_PATH% ^
--model_variant=xception_65 ^
--atrous_rates=6 ^
--atrous_rates=12 ^
--atrous_rates=18 ^
--output_stride=16 ^
--decoder_output_stride=4 ^
--num_classes=%NUM_CLASSES% ^
--crop_size=513 ^
--crop_size=513 ^
--inference_scales=1.0

