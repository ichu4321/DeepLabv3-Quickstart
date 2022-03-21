import os
import sys
import random

def split(image_folder):
	# get list of files in directory
	files = os.listdir(image_folder);

	# chop off file types
	files = [file[:file.find(".")] for file in files];

	# create an index list and shuffle
	indices = [a for a in range(len(files))];
	random.shuffle(indices);

	# create an 80-20 train/val split
	train_size = int(0.8 * len(files));

	# split indices
	train_indices = indices[:train_size];
	val_indices = indices[train_size:];

	# split files
	train_files = [];
	for index in train_indices:
		train_files.append(files[index]);
	val_files = [];
	for index in val_indices:
		val_files.append(files[index]);

	# write splits to file
	split_dir = "Splits/";
	train_split = open(split_dir + "train.txt", 'w');
	for file in train_files:
		train_split.write(file + "\n");
	train_split.close();

	val_split = open(split_dir + "val.txt", 'w');
	for file in val_files:
		val_split.write(file + "\n");
	val_split.close();

	trainval = open(split_dir + "trainval.txt", 'w');
	for file in files:
		trainval.write(file + "\n");



if __name__ == "__main__":
	image_folder = sys.argv[1];
	print("Creating Splits For " + str(image_folder));
	split(image_folder);