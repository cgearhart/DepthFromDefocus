# Interpolated Depth From Defocus

This project implements a form of passive depth from defocus to create a novel image approximating the depth map of a scene from multiple exposures of the same scene with slight variations in focal point by interpolating the depth of each pixel using graph cut optimization. Depth maps have a variety of practical uses in computer vision and robotics, so the allure of recovering depth without stereo vision systems is appealing. The initial premise was based on previous results implementing focus stacking, then the scope was expanded to incorporate techniques described by other researchers.

## Setup & Requirements

In addition to the package dependencies in requirements.txt, running this code also requires two packages, [OpenCV](http://opencv.org/downloads.html) and [gco_python](https://github.com/amueller/gco_python), that must be installed separately following the instructions on their respective project pages. The packages in requirements.txt can be installed with [pip](https://pip.pypa.io/en/stable/installing/) by running the following command:

`pip install -r requirements.txt`

## Usage

The program is invoked from the command line with two positional arguments for the source and destination directories and two optional arguments for the file extension and message logging verbosity level.

```
usage: main.py [-h] [-e EXT] [-v {0,1,2,3,4,5}] source [dest]

Depth from focus pipeline.

positional arguments:
  source                Directory containing input images
  dest                  Directory to write output images

optional arguments:
  -h, --help            show this help message and exit
  -e EXT, --ext EXT     Image filename extension pattern [see fnmatch]
  -v {0,1,2,3,4,5}, --verbose {0,1,2,3,4,5}
                        Logging level [0 - CRITICAL, 3 - INFO, 5 - ALL]
```