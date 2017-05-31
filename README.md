# clamm_2017
Submission for CLaMM competition as part of ICDAR 2017

Requires using a version of Caffe where the BatchNormLayer implementation
is the robust version from the DIGITs version of Caffe.  One such Caffe
version (and the recommended) is my personal fork: https://github.com/waldol1/caffe
As Caffe tends to change over time, I tested using the lastest commit of mine
as of May 25, 2017.

Running clamm_submission.py requires compiling the python bindings for Caffe,
as well as the python bindings for OpenCV, and requires the numpy and scipy
python modules.

Running `python clamm_submission.py` with no arguments produces the following usage statement:

```
USAGE: python clamm_submission.py image_directory [output_directory] [scripts|dates] [gpu#]
	image_directory is the input directory containing the test images
	output_directory is where the output files will be written (defaults to '.')
	[scripts|dates] indicates what is predicted.  (defaults to 'scripts')
	gpu is an integer device ID to run networks on the specified GPU.  If omitted, CPU mode is used
```
