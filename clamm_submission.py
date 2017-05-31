
import os
import sys
import numpy as np
import scipy.spatial.distance
import caffe
import cv2

# default networks to use for prediction
NET_CONFIG_FILE = "scripts.config"

# acceptable image suffixes
IMAGE_SUFFIXES = ('.jpg', '.jpeg', '.tif', '.tiff', '.png', '.bmp', '.ppm', '.pgm')

# number of subwindows processed by a network in a batch
# Higher numbers speed up processing (only marginally if BATCH_SIZE > 16)
# The larger the batch size, the more memory is consumed (both CPU and GPU)
BATCH_SIZE=4

JPG_SHAVE_PERC = 0.15


def setup_networks():
	networks = list()
	for ln, line in enumerate(open(NET_CONFIG_FILE).readlines()):
		try:
			if line.startswith('#'):
				continue
			tokens = line.split()
			scale = int(tokens[0])
			deploy_file = tokens[1]
			weights_file = tokens[2]
		except:
			print "Error occured in parsing NET_CONFIG_FILE %r on line %d" % (NET_CONFIG_FILE, ln)
			print "Offending line: %r" % line
			print "Exiting..."
			exit(1)
		network = caffe.Net(deploy_file, weights_file, caffe.TEST)
		networks.append( (scale, network) )

	return networks


def fprop(network, ims, batchsize=BATCH_SIZE):
	# batch up all transforms at once
	idx = 0
	responses = list()
	while idx < len(ims):
		sub_ims = ims[idx:idx+batchsize]
		network.blobs["data"].reshape(len(sub_ims), 1, ims[0].shape[1], ims[0].shape[0]) 
		for x, im in enumerate(sub_ims):
			transposed = im[np.newaxis, np.newaxis, :, :]
			network.blobs["data"].data[x,:,:,:] = transposed
		idx += batchsize

		# propagate on batch
		network.forward()
		output = np.squeeze(np.copy(network.blobs["prob"].data), axis=(2,3))
		responses.append(output)
	return np.concatenate(responses, axis=0)


def predict(network, ims):
	all_outputs = fprop(network, ims)

	# throw out non-script predictions for each tile
	for idx in xrange(all_outputs.shape[0]):
		# index 0 does not correspond to any GT class
		all_outputs[idx][0] = 0

		# renormalize
		all_outputs[idx] = all_outputs[idx] / np.sum(all_outputs[idx])
	
		
	mean_outputs = np.average(all_outputs, axis=0)
	return mean_outputs


def resize(im, scale_factor):
	new_height = int(scale_factor * im.shape[0])
	new_width = int(scale_factor * im.shape[1])
	return cv2.resize(im, (new_width, new_height))


def get_subwindows(im):
	height, width, = 227, 227
	y_stride, x_stride, = 100, 100
	if (height > im.shape[0]) or (width > im.shape[1]):
		print "Invalid crop: crop dims larger than image (%r with %r)" % (im.shape, tokens)
		exit(1)
	ims = list()
	y = 0
	while (y + height) <= im.shape[0]:
		x = 0
		if (y + height + y_stride) > im.shape[0]:
			y = im.shape[0] - height
		while (x + width) <= im.shape[1]:
			if (x + width + x_stride) > im.shape[1]:
				x = im.shape[1] - width
			ims.append(im[y:y+height,x:x+width])
			x += x_stride
		y += y_stride

	return ims

	
def evaluate(networks, image_dir):
	predictions = list()
	image_files = list()
	image_num = 0
	for image_file in os.listdir(image_dir):
		image_path = os.path.join(image_dir, image_file)
		if not image_path.lower().endswith(IMAGE_SUFFIXES):
			print "Skipping non-image file", image_file
			continue
		# keep track of filenames for output file bookkeeping
		image_files.append(image_file)

		# load, shift, and scale pixel values
		image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

		if image_file.lower().endswith('jpg'):
			shave_y = int(image.shape[0] * JPG_SHAVE_PERC)
			shave_x = int(image.shape[1] * JPG_SHAVE_PERC)
			image = image[shave_y:-1 * shave_y, shave_x:-1 * shave_x]
			
		image = 0.0039 * (image - 128.)

		network_predictions = list()
		for scale, network in networks:
			# resize image if necessary
			if scale == 100:
				resized_image = np.copy(image)
			else:
				resized_image = resize(np.copy(image), scale / 100.)

			# chop up image into 227x227 subwindows with a stride of 100x100
			subwindows = get_subwindows(resized_image)

			# get the average prediction over all the subwindows
			network_prediction = predict(network, subwindows)
			network_predictions.append(network_prediction)

		# average predictions over all the networks
		network_predictions = np.asarray(network_predictions)
		average_prediction = np.average(network_predictions, axis=0)
		predictions.append(average_prediction)

		image_num += 1
		print "Processed %d images" % image_num

	return image_files, np.asarray(predictions)


def write_dist_matrix(image_files, dist_matrix, dist_matrix_file):
	fd = open(dist_matrix_file, 'w')

	# write header
	header = ",".join(["N/A"] + image_files)
	fd.write("%s\n" % header)

	for idx in xrange(len(image_files)):
		image_file = image_files[idx]
		values = map(str, list(dist_matrix[idx]))
		row = ",".join([image_file] + values)
		fd.write("%s\n" % row)

	fd.close()
	

def write_all_class_predictions(image_files, predictions, predictions_file):
	fd = open(predictions_file, 'w')

	for idx in xrange(len(image_files)):
		image_file = image_files[idx]
		values = map(str, map(lambda f: round(100 * f) / 100.,  list(predictions[idx, 1:])))
		row = ",".join([image_file] + values)
		fd.write("%s\n" % row)

	fd.close()


def write_results(image_files, predictions, out_dir):
	# pairwise distance matrix
	dist_matrix = scipy.spatial.distance.pdist(predictions, metric='euclidean')
	dist_matrix = scipy.spatial.distance.squareform(dist_matrix)
	dist_matrix /= np.sum(dist_matrix.flatten())

	# distance matrix output
	dist_matrix_file = os.path.join(out_dir, "distance_matrix.csv")
	write_dist_matrix(image_files, dist_matrix, dist_matrix_file)

	# distribution over output classes
	all_class_prediction_file = os.path.join(out_dir, "belonging_matrix.csv")
	write_all_class_predictions(image_files, predictions, all_class_prediction_file)


def main(image_dir, out_dir):
	networks = setup_networks()
	image_files, predictions = evaluate(networks, image_dir)
	write_results(image_files, predictions, out_dir)


if __name__ == "__main__":
	if len(sys.argv) < 2:
		print "USAGE: python clamm_submission.py image_directory [output_directory] [scripts|dates] [gpu#]"
		print "\timage_directory is the input directory containing the test images"
		print "\toutput_directory is where the output files will be written (defaults to '.')"
		print "\t[scripts|dates] indicates what is predicted.  (defaults to 'scripts')"
		print "\tgpu is an integer device ID to run networks on the specified GPU.  If omitted, CPU mode is used"
		exit(1)
	# only required argument
	image_dir = sys.argv[1]

	# attempt to parse an output directory
	try:
		out_dir = sys.argv[2]
		try:
			os.makedirs(out_dir)
		except:
			pass
	except:
		out_dir = "./"

	try:
		if sys.argv[3] == 'dates':
			NET_CONFIG_FILE = 'dates.config'
			print "Predicting Date Types"
		else:
			print "Predicting Script Types"
	except:
		print "Predicting Script Types"

	# use gpu if specified
	try:
		gpu = int(sys.argv[4])
		if gpu >= 0:
			caffe.set_mode_gpu()
			caffe.set_device(gpu)
	except:
		caffe.set_mode_cpu()

	main(image_dir, out_dir)
	
