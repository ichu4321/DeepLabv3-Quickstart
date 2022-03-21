# Lint as: python2, python3
# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Exports trained model to TensorFlow frozen graph."""

import os

import tensorflow as tf
from tensorflow.contrib import quantize as contrib_quantize
from tensorflow.python.tools import freeze_graph

from deeplab import common
from deeplab import input_preprocess
from deeplab import model

# aliases
slim = tf.contrib.slim;
flags = tf.app.flags;

# define flags
FLAGS = flags.FLAGS;
flags.DEFINE_string("checkpoint_dir", None, "Folder with .ckpt files");
flags.DEFINE_string("export_path", None, "Path to output Tensorflow frozen graph");
flags.DEFINE_integer("num_classes", 21, "Number of classes");
flags.DEFINE_multi_integer("crop_size", [513,513], "Crop size [height, width]");

# For `xception_65`, use atrous_rates = [12, 24, 36] if output_stride = 8, or
# rates = [6, 12, 18] if output_stride = 16. For `mobilenet_v2`, use None. Note
# one could use different atrous_rates/output_stride during training/evaluation.
flags.DEFINE_multi_integer('atrous_rates', None, 'Atrous rates for atrous spatial pyramid pooling');

flags.DEFINE_integer("output_stride", 8,
	"The ratio of input to output spatial resolution");

# Change to [0.5, 0.75, 1.0, 1.25, 1.5, 1.75] for multi-scale inference.
flags.DEFINE_multi_float('inference_scales', [1.0],
                         'The scales to resize images for inference');

flags.DEFINE_bool("add_flipped_images", False,
	"Add flipped images during inference or not");

flags.DEFINE_integer("quantize_delay_step", -1,
	"Steps to start quantized training. If < 0, will not quantize mode");

flags.DEFINE_bool("save_inference_graph", False,
	"Save inference graph in text proto");

# name of the input layer for the exported model
_INPUT_NAME = "ImageTensor";

# Output name of the exported predictions
_OUTPUT_NAME = "SemanticPredictions";
_RAW_OUTPUT_NAME = "RawSemanticPredictions";

# Output name of the exported probabilities
_OUTPUT_PROB_NAME = "SemanticProbabilities";
_RAW_OUTPUT_PROB_NAME = "RawSemanticProbabilities";

# creates an input tensor for deeplab
def _create_input_tensors():
	# take in a 4D tensor
	input_image = tf.placeholder(tf.uint8, [1, None, None, 3], name=_INPUT_NAME);
	original_image_size = tf.shape(input_image)[1:3];

	# squeeze axis=0 to make it 3D
	image = tf.squeeze(input_image, axis=0);

	# do preprocessing step
	resized_image, image, _ = input_preprocess.preprocess_image_and_label(
		image,
		label=None,
		crop_height=FLAGS.crop_size[0],
		crop_width=FLAGS.crop_size[1],
		min_resize_value=FLAGS.min_resize_value,
		max_resize_value=FLAGS.max_resize_value,
		resize_factor=FLAGS.resize_factor,
		is_training=False,
		model_variant=FLAGS.model_variant);
	resized_image_size = tf.shape(resized_image)[:2];

	# expand axis=0 to get back to 4D
	image = tf.expand_dims(image, 0);

	# return
	return image, original_image_size, resized_image_size;

# main is called through "tf.app.run()"
def main(unused_argv):
	# set tensorflow logging
	tf.logging.set_verbosity(tf.logging.INFO);
	tf.logging.info("Prepare to export model to: %s", FLAGS.export_path);

	# find the newest (biggest number) checkpoint file
	checkpoint_dir = FLAGS.checkpoint_dir;
	files = os.listdir(checkpoint_dir);
	biggest_ckpt = 0;

	# loop and compare the files
	model_str = "model.ckpt-";
	for file in files:
		# look for the model string	
		chop = file.find(model_str);
		if chop != -1:
			# chop out the number
			chop += len(model_str);
			file = file[chop:];
			num = int(file[:file.find(".")]);
			biggest_ckpt = max(num, biggest_ckpt);

	# construct into a filename
	checkpoint_file = checkpoint_dir + model_str + str(biggest_ckpt);

	# set up the model graph
	with tf.Graph().as_default():
		image, image_size, resized_image_size = _create_input_tensors();

		# gather model options
		model_options = common.ModelOptions(
			outputs_to_num_classes={common.OUTPUT_TYPE: FLAGS.num_classes},
			crop_size=FLAGS.crop_size,
			atrous_rates=FLAGS.atrous_rates,
			output_stride=FLAGS.output_stride);

		# only do single-scale inference
		tf.logging.info("Exported model does single-scale inference");
		predictions = model.predict_labels(
			image,
			model_options=model_options,
			image_pyramid=FLAGS.image_pyramid);

		# parse predictions out into raw predictions/probabilities
		raw_preds = tf.identity(tf.cast(predictions[common.OUTPUT_TYPE], tf.float32),
			_RAW_OUTPUT_NAME);

		raw_probs = tf.identity(
			predictions[common.OUTPUT_TYPE + model.PROB_SUFFIX],
			_RAW_OUTPUT_PROB_NAME);

		# crop out the resized regions
		rw = resized_image_size[0];
		rh = resized_image_size[1];
		semantic_preds = raw_preds[:, :rw, :rh];
		semantic_probs = raw_probs[:, :rw, :rh];

		# resize back to the original shape
		semantic_preds = tf.expand_dims(semantic_preds, 3);
		semantic_preds = tf.image.resize_images(
			semantic_preds, 
			image_size,
			method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
			align_corners=True);
		semantic_preds = tf.cast(tf.squeeze(semantic_preds, 3), tf.int32);
		semantic_preds = tf.identity(semantic_preds, name=_OUTPUT_NAME);

		semantic_probs = tf.image.resize_bilinear(
			semantic_probs,
			image_size,
			align_corners=True,
			name=_OUTPUT_PROB_NAME);

		# no model quantization

		# save model
		saver = tf.train.Saver(tf.all_variables());
		dirname = os.path.dirname(FLAGS.export_path);
		tf.gfile.MakeDirs(dirname);
		graph_def = tf.get_default_graph().as_graph_def(add_shapes=True);
		freeze_graph.freeze_graph_with_def_protos(
			graph_def,
			saver.as_saver_def(),
			checkpoint_file,
			_OUTPUT_NAME + "," + _OUTPUT_PROB_NAME,
			restore_op_name=None,
			filename_tensor_name=None,
			output_graph=FLAGS.export_path,
			clear_devices=True,
			initializer_nodes=None);

		# save the inference graph (not sure what this is used for)
		# add if necessary


if __name__ == "__main__":
	flags.mark_flag_as_required("checkpoint_dir");
	flags.mark_flag_as_required("export_path");
	tf.app.run();