# Chris Gearhart

import argparse
import logging
import os
import sys

from sys import path
from glob import glob

import cv2
import numpy as np


LEVELS = [logging.CRITICAL, logging.ERROR, logging.WARNING,
          logging.INFO, logging.DEBUG, logging.NOTSET]
LEVEL_NAMES = ["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "ALL"]

FORMAT = '%(levelname)s: %(message)s'
logging.basicConfig(format=FORMAT)
log = logging.getLogger()


def save_images(stack, prefix, norm=False):

    if norm:
        stack = [x / np.max(x) * 255 for x in stack]

    log.debug("Saving files: {}".format(prefix))

    for idx, img in enumerate(stack):
        cv2.imwrite(prefix + '{}.png'.format(idx), img)


def read_images(filepattern):
    """
    """
    log.info("Reading images from {}".format(filepattern))
    log.debug("files: " + ', '.join(glob(filepattern)))

    try:
        images = map(cv2.imread, glob(filepattern))
    except Exception as e:
        log.error('Fatal error: image files failed to load.')
        log.error(e)
        exit()

    return images


def align(image_stack):
    """
    Align the image stack with rigid affine transforms between frames
    """

    _stack = [image_stack[0]]
    (r, c) = image_stack[0].shape[:2]
    corners = np.array([[0., c], [0., r], [1., 1.]], dtype=np.float)
    nw_corner, se_corner = corners[:, 0], corners[:, 1]
    transform = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
                         dtype=np.float)

    for anchor, img in zip(image_stack[:-1], image_stack[1:]):

        new_t = cv2.estimateRigidTransform(img, anchor, fullAffine=False)
        transform[:2, :2] = new_t[:2, :2].dot(transform[:2, :2])
        transform[:2, 2] = new_t[:, 2] + transform[:2, 2]
        new_im = cv2.warpAffine(img, transform[:2, :], (c, r))
        _stack.append(new_im)

        bounds = transform.dot(corners)
        nw_corner = np.max([nw_corner, bounds[:, 0]], axis=0)
        se_corner = np.min([se_corner, bounds[:, 1]], axis=0)

        log.debug("\n" + str(transform))

    lx, ly = nw_corner[:2]
    rx, ry = se_corner[:2]

    _stack = [img[ly:ry, lx:rx] for img in _stack]

    # DEBUG SAVE THE IMAGES
    save_images(_stack, "aligned")

    return _stack


def piecewiseBilteral(I, L, sigmaS, sigmaR):
    """
    I - intensity image (original bw image)
    L - label image (label number * max_laplacian)
    sigmaS - spacial standard deviation
    sigmaR - intensity standard deviation
    """

    J = np.zeros(I.shape, dtype=np.float)
    minI, maxI = np.min(I), np.max(I)
    NB_SEGMENTS = (maxI - minI) / sigmaR
    SQRT_2PI = np.sqrt(2 * np.pi)

    log.debug("Bilateral interpolation...")
    log.debug("min/max {}/{}".format(minI, maxI))
    log.debug("Segments: {}".format(NB_SEGMENTS))

    def g(x, s):
        return np.exp(-(x * x) / (2 * s * s)) / (s * SQRT_2PI)

    for j in np.arange(NB_SEGMENTS):

        ij = minI + (j * sigmaR)
        Gj = g(I - ij, sigmaR)
        Kj = cv2.GaussianBlur(Gj, (0, 0), sigmaS)
        Hj = cv2.GaussianBlur(Gj * L, (0, 0), sigmaS)

        # temporarily disable divide-by-zero warnings and
        # invalid value warnings -- values from those locations
        # in K are ignored
        with np.errstate(divide='ignore', invalid='ignore'):
            Jj = np.where((Kj > 0), Hj / Kj, 0)

        # linear interpolation based on equation (1) in:
        # http://www.ee.cuhk.edu.hk/~tblu/monsite/pdfs/blu0401.pdf
        dist = np.abs((I - ij) / sigmaR)
        hat_weights = np.where(dist <= 1, 1 - dist, 0)
        J += (Jj * hat_weights)

    return J


def interpolate_depth(depth_pixels, images, params):
    """
    """
    # Given an image with some known pixel depths, interpolate the
    # most likely distance of unknown pixels using bilinear interpolation
    # from the source images.
    # The depth pixels should be concentrated along edges in the source
    # images, so interpolating from those depths using a bilateral filter
    # on the focal stack will spread the distance estimate based on the
    # visual similarity of the surrounding scene.
    pass


def main(images):
    """
    """
    # pipeline:
    # - convert to BW
    # - align then crop focal stack - (at least pairwise with anchor,
    #     better in parallel)
    # - calculate gradients
    # - for each pixel, determine the gradient strength as a function
    #     of focal stack index and use maximal selection to choose the
    #     pixels that are candidates for focal boundaries
    # - build an image for each focal plane with the maximal pixels
    #     activate and apply a large gaussian gradient to build a weight
    #     matrix for each image in the focal stack
    # - interpolate between layers using the combined weight of the
    #     focal stacks

    # gradient depends on intensity, not color planes
    bw_images = [cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in images]

    # align and crop the images to account for minute magnification
    # start with homography; try optical flow possibly
    bw_stack = align(bw_images)

    # calculate the gradient magnitude for all points
    grad_stack = map(lambda x: np.abs(cv2.Laplacian(x, cv2.CV_64F)), bw_stack)
    max_vals = np.max(np.array(grad_stack), axis=0)
    max_grads = [np.where(g == max_vals, g, 0) for g in grad_stack]

    save_images(max_grads, "max", norm=True)

    sigmaR = 9.
    sigmaS = 31.
    label_stack = [piecewiseBilteral(b, w, sigmaS, sigmaR) for b, w in
                   zip(bw_stack, max_grads)]
    label_stack = [np.zeros(label_stack[0].shape)] + label_stack

    save_images(label_stack[1:], "labels", norm=True)

    d = np.argmax(label_stack, axis=0)

    print np.min(d), np.max(d)
    cv2.imwrite('dist.png', d / np.float(np.max(d)) * 255)

    d = d * np.sum(label_stack, axis=0) / 4.
    cv2.imwrite('dist2.png', d / np.float(np.max(d)) * 255)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Depth from focus pipeline.")
    parser.add_argument('source',
                        default=os.getcwd(),
                        help="Directory containing input images")
    parser.add_argument('dest',
                        nargs='?',
                        default="output",
                        help="Directory to write output images")
    parser.add_argument("-b", "--bilateral",
                        nargs=2,
                        default='',
                        help="")
    parser.add_argument('-e', '--ext',
                        default='*[jgptJGPT][pinPIN][gfGF]*',
                        help='Image filename extension pattern [see fnmatch]')
    parser.add_argument("-w", "--width",
                        default=2,
                        type=int,
                        help="")
    parser.add_argument("-v", "--verbose",
                        type=int,
                        default=0,
                        choices=[0, 1, 2, 3, 4, 5],
                        help='Logging level [0 - CRITICAL, 5 - ALL]')
    args = parser.parse_args()
    log.setLevel(LEVELS[args.verbose])
    log.info("Set logging level: {}".format(LEVEL_NAMES[args.verbose]))

    filepattern = os.path.join(args.source, args.ext)
    images = read_images(filepattern)

    main(images)
    # main(images, args.dest, args.bilateral, args.width)
