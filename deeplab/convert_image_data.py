from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import os.path
import sys
import build_data
from six.moves import range
import tensorflow as tf

# alias the FLAGS
FLAGS = tf.app.flags.FLAGS

# set up flags
tf.app.flags.DEFINE_string(
	"image_folder",
	None,
	"Folder with the training images"
	);

tf.app.flags.DEFINE_string(
	"label_folder",
	None,
	"Folder with the training labels"
	);

tf.app.flags.DEFINE_string(
	"splits_folder",
	None,
	"Folder with the train/val split lists"
	);

tf.app.flags.DEFINE_string(
	"output_dir",
	"./tfrecord",
	"Path to the TFRecord output files"
	);

# how many pieces to split the data up into
_NUM_SHARDS = 4;

# convert the dataset into a TFRecord shard
def _convert_dataset(dataset_split):
	# get the dataset name [train,val,trainval]
	dataset = os.path.basename(dataset_split)[:-4];
	print("Processing ", dataset);

	# get the filenames from the split file
	filenames = [a.strip() for a in open(dataset_split, 'r')];

	# calculate shard splits
	num_images = len(filenames);
	num_per_shard = int(math.ceil(num_images / _NUM_SHARDS));

	# create image readers
	image_reader = build_data.ImageReader(FLAGS.image_format, channels=3);
	label_reader = build_data.ImageReader(FLAGS.label_format, channels=1);

	# create shards
	for shard_id in range(_NUM_SHARDS):
		# set up filename
		output_filename = os.path.join(
			FLAGS.output_dir,
			'%s-%05d-of-%05d.tfrecord' % (dataset, shard_id, _NUM_SHARDS));

		# write to tf record
		with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
			
			# index split for shard
			start_index = shard_id * num_per_shard;
			end_index = min((shard_id + 1) * num_per_shard, num_images);
			print("ID Range: " + str([start_index, end_index]));

			# do splits
			for i in range(start_index, end_index):
				# progress check
				sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
					i + 1, len(filenames), shard_id));
				sys.stdout.flush();

				# read the image
				image_filename = os.path.join(
					FLAGS.image_folder,
					filenames[i] + "." + FLAGS.image_format
					);
				image_data = tf.gfile.GFile(image_filename, 'rb').read();
				height, width = image_reader.read_image_dims(image_data);

				# read the label
				label_filename = os.path.join(
					FLAGS.label_folder,
					filenames[i] + "." + FLAGS.label_format
					);
				label_data = tf.gfile.GFile(label_filename, 'rb').read();
				label_height, label_width = label_reader.read_image_dims(label_data);

				# check for mismatch
				if height != label_height or width != label_width:
					raise RuntimeError('Shape mismatched between image and label');

				# convert to TFRecord
				record = build_data.image_seg_to_tfexample(
					image_data,
					filenames[i],
					height,
					width,
					label_data
					);
				tfrecord_writer.write(record.SerializeToString());

		# flush stdout
		sys.stdout.write("\n");
		sys.stdout.flush();

# gets called from tf.app.run()
def main(unused_argv):
	dataset_splits = tf.gfile.Glob(os.path.join(FLAGS.splits_folder, "*.txt"));
	for split in dataset_splits:
		_convert_dataset(split);


if __name__ == "__main__":
	tf.app.run();