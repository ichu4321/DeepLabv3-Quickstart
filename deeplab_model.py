import cv2
import numpy as np
import tensorflow as tf

class DeepLabModel(object):

	# CLASS CONSTANTS
	INPUT_TENSOR_NAME = "ImageTensor:0";
	OUTPUT_TENSOR_NAME = "SemanticPredictions:0";
	INPUT_SIZE = 513;

	def __init__(self, filepath):
		# create colormap
		self.colormap = self.create_colormap();

		# load pre-trained model from graph (.pb) file
		self.graph = tf.Graph();

		# load the graph file
		graph_file = open(filepath, 'rb');
		graph_def = tf.GraphDef.FromString(graph_file.read());
		graph_file.close();

		# set definition
		with self.graph.as_default():
			tf.import_graph_def(graph_def, name='');

		# start session
		self.sess = tf.Session(graph=self.graph);

	# give predictions on a single image
	def predict(self, image):
		# calculate resize
		height, width = image.shape[:2];
		resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height);
		target_size = (int(resize_ratio * width), int(resize_ratio * height));

		# resize image
		resized = cv2.resize(image, (target_size));

		# convert to RGB
		rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB);

		# run it through network
		seg_map = self.sess.run(
			self.OUTPUT_TENSOR_NAME,
			feed_dict = {self.INPUT_TENSOR_NAME: [np.asarray(rgb)]});

		# convert segment map into an opencv image
		segment_img = self.colormap[seg_map[0]].astype(np.uint8);
		return resized, segment_img;

	# returns a colormap (maps grayscale to random-ish colors)
	def create_colormap(self):
		colormap = np.zeros((256,3), dtype = np.uint8);
		indices = np.arange(256, dtype = np.uint8);

		# shift colors
		for shift in reversed(range(8)):
			for channel in range(3):
				colormap[:, channel] |= ((indices >> channel) & 1) << shift;
			indices >>= 3;
		return colormap;