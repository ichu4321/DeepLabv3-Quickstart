import cv2
import numpy as np
from deeplab_model import DeepLabModel

# overlays label onto image
def overlay(image, label):
	# dull the colors
	mult = 0.50;
	label = cv2.multiply(label, (mult,mult,mult,mult)); # needs 4-channel for some reason

	# overlay
	image = cv2.add(image, label);
	return image;

# run loop on live cam
def main():
	# set up camera
	cap = cv2.VideoCapture(0);

	# load model
	model = DeepLabModel(filepath = "model.pb");

	# loop until 'q'
	done = False;
	while not done:
		# get frame
		ret, frame = cap.read();
		if not ret:
			continue;

		# get prediction
		resized, seg_map = model.predict(frame);

		# overlay segment map and show
		image = overlay(resized, seg_map);
		cv2.imshow("Image", image);
		cv2.imshow("Label", seg_map);
		done = cv2.waitKey(1) == ord('q');


if __name__ == "__main__":
	main();
