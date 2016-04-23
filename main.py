# CS6475 Final Project - Spring 2016
# Chris Gearhart

import argparse
import logging
import os

from glob import glob
from collections import OrderedDict

import cv2
import numpy as np

from gco_python.pygco import cut_simple
from skimage.morphology import disk
from scipy import optimize


LEVELS = [logging.CRITICAL, logging.ERROR, logging.WARNING,
          logging.INFO, logging.DEBUG, logging.NOTSET]
LEVEL_NAMES = ["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "ALL"]

FORMAT = '%(levelname)s: %(message)s'
logging.basicConfig(format=FORMAT)
log = logging.getLogger()


def save_images(stack, prefix, ext=".png", norm=False, dest=os.getcwd()):

    if not os.path.exists(dest):
        os.makedirs(dest)

    if norm:
        stack = map(norm_img, stack)

    log.debug("Saving files: {}".format(dest))

    for idx, img in enumerate(stack):
        cv2.imwrite(os.path.join(dest, prefix + '{}.png'.format(idx)), img)


def norm_img(img):
    _img = img.astype(np.float32)
    i_min, i_max = np.min(_img), np.max(_img)
    return ((img - i_min) / (i_max - i_min) * 255.0).astype(np.uint8)


def norm_limits(img, lo, hi):
    _img = img.astype(np.float32)
    i_min, i_max = np.min(_img), np.max(_img)
    return ((_img - i_min) / (i_max - i_min) * (hi - lo)) + lo


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

    log.info("Aligning images with rigid transform.")

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

        log.debug("Transformation matrix:\n" + str(transform))

    lx, ly = nw_corner[:2]
    rx, ry = se_corner[:2]

    _stack = [img[ly:ry, lx:rx] for img in _stack]

    return _stack, _stack[0].shape


def all_in_focus(images):
    """
    """
    log.info("Generating all-in-focus image.")

    # unary scaling factor to convert from float32 -> int32
    # pair scaling factor performs conversion and discounts mismatches
    unary_scale = 2**22
    pair_scale = 2**8

    log.debug("Unary scaling factor: {}".format(unary_scale))
    log.debug("Pairwise scaling factor: {}".format(pair_scale))

    n = len(images)

    unary = []
    for idx, img in enumerate(images):
        _img = img.astype(np.float32) / 255.
        grad = np.exp(-(cv2.Sobel(_img, cv2.CV_32F, 1, 1)**2))
        unary.append(cv2.GaussianBlur(grad, (13, 13), 0) * unary_scale)

    unary = np.stack([x.astype(np.int32) for x in unary], axis=-1)
    unary = (unary - np.min(unary))

    ii, jj = np.meshgrid(range(n), range(n))
    pairwise = np.abs(4*ii - 4*jj).astype(np.int32) * pair_scale

    graph_img = cut_simple(unary, pairwise, n_iter=20)

    aif_img = np.sum([np.where(i == graph_img, images[i], 0)
                     for i in range(n)], axis=0, dtype=np.float64)

    return graph_img, aif_img


def generate_blur_stack(_img, num_steps=26, size=0.25):
    """
    """
    log.info("Generating blur stack in range 0.75-{}."
             .format(0.75 + num_steps * size))

    disks = OrderedDict()
    for i in range(num_steps):
        d = disk(0.75 + i * size, dtype=np.float64)
        if d.size % 2 == 0:
            continue
        disks[(np.sum(d), d.size)] = d / max(np.sum(d), 1)

    blur_stack = [cv2.filter2D(_img, cv2.CV_64F, d,
                  borderType=cv2.BORDER_REFLECT) for d in disks.values()]

    return blur_stack


def estimate_focal_depths(img_stack, blur_stack, s, fi, A=5.0, F=18.0):
    """
    """
    log.info("Solving for focal depth parameters.")

    # parameters from paper
    alpha = .3
    B, C = [], []
    for frame in img_stack:
        D = [cv2.GaussianBlur(np.abs(frame - b), (15, 15), 0)
             for b in blur_stack]
        B.append(np.argmin(D, axis=0) / 2.0)
        C.append(np.power(np.mean(D, axis=0) - np.min(D, axis=0), alpha))

    max_c = np.max(C)
    C = [c / max_c for c in C]

    BC = np.array([b * c for b, c in zip(B, C)])

    def f_residuals(x, y, s):
        A = x[0]
        F = x[1]
        fi = x[2:]
        y_hat = np.concatenate([A * (abs(f - s) / s) * (F / (f - F))
                                for f in fi])
        err = y.ravel() - y_hat
        return err

    x = np.concatenate([[A], [F], fi])
    sol = optimize.leastsq(f_residuals, x.astype(np.float64),
                           args=(BC, s.ravel()))

    log.debug("Parameters:\nA: {}\nF: {}\nFocal Depths: {}"
              .format(sol[0][0], sol[0][1], sol[0][2:]))

    return sol[0], BC, C


def graph_interpolate(fi, BC, steps=10, A=5, F=18):

    log.info("Interpolating depth from graph.")

    s_vals = np.linspace(np.min(fi), np.max(fi), num=steps)
    # unary scaling factor to convert from float64 -> int32
    # pair scaling factor performs conversion and discounts mismatches
    unary_scale = 2**23
    pair_scale = 2**11

    log.debug("Unary scaling factor: {}".format(unary_scale))
    log.debug("Pairwise scaling factor: {}".format(pair_scale))

    unary = []
    for idx, s in enumerate(s_vals):
        err = np.min([(bc - (A * np.abs(f - s) / s) * (F / (f - F)))**2
                      for (f, bc) in zip(fi, BC)], axis=0)
        unary.append(err)

    max_u, min_u = np.max(unary), np.min(unary)
    unary = np.stack([(((u - min_u) / (max_u - min_u)) * unary_scale)
                      for u in unary], axis=-1)

    ii, jj = np.meshgrid(range(steps), range(steps))
    pairwise = np.abs(4*ii - 4*jj) * pair_scale

    graph_img = cut_simple(unary.astype(np.int32),
                           pairwise.astype(np.int32),
                           n_iter=50)
    return graph_img


def main(images, dest):
    """
    """
    # align and crop the images to account for minute magnification
    bw_images, imshape = align([cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
                                for x in images])
    n = len(bw_images)
    imshape = bw_images[0].shape

    save_images(bw_images, "aligned", dest=os.path.join(dest, "aligned"))

    graph_img, aif_img = all_in_focus(bw_images)
    save_images([aif_img], "aifimg", dest=os.path.join(dest, "aifimg"))

    # num_steps specified in paper
    blur_stack = generate_blur_stack(aif_img, num_steps=15, size=.25)
    save_images(blur_stack, "blur", norm=True, dest=os.path.join(dest, "blur"))

    # initialize parameters for the solver
    A, F = 5, 18
    fi = (np.arange(n)*40 + 500).astype(np.float64)
    s = np.zeros(imshape, dtype=np.float64)
    for idx, f in enumerate(fi):
        s[graph_img == idx] = f

    sol, BC, C = estimate_focal_depths(bw_images, blur_stack, s, fi, A=A, F=F)

    A, F, fi = sol[0], sol[1], sol[2:]
    s2 = norm_limits(graph_interpolate(fi, BC, steps=25, A=A, F=F),
                     np.min(fi), np.max(fi))

    # laplacian blending shenanigans
    c = np.max(C, axis=0)

    g, h = s, s2
    stack = []
    for _ in range(3):
        _g, _h, _c = map(cv2.pyrDown, [g, h, c])
        y, x = g.shape
        lg = g - cv2.pyrUp(_g)[:y, :x]
        lh = h - cv2.pyrUp(_h)[:y, :x]
        blend = (1 - c[:y, :x]) * lg + c[:y, :x] * lh
        stack.append(blend)
        g, h, c = _g, _h, _c

    y, x = g.shape
    stack.append((1 - c[:y, :x]) * g + c[:y, :x] * h)

    res = np.zeros(stack[-1].shape)
    for img in stack[::-1]:
        y, x = img.shape
        res = cv2.pyrUp(res[:y, :x] + img)

    s3 = cv2.pyrDown(res)

    save_images([s, s2, s3], "depthmap", norm=True,
                dest=os.path.join(dest, "graph"))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Depth from focus pipeline.")
    parser.add_argument('source',
                        default=os.getcwd(),
                        help="Directory containing input images")
    parser.add_argument('dest',
                        nargs='?',
                        default="output",
                        help="Directory to write output images")
    parser.add_argument('-e', '--ext',
                        default='*[jgptJGPT][pinPIN][gfGF]*',
                        help='Image filename extension pattern [see fnmatch]')
    parser.add_argument("-v", "--verbose",
                        type=int,
                        default=3,
                        choices=[0, 1, 2, 3, 4, 5],
                        help='Logging level [0 - CRITICAL, 5 - ALL]')
    args = parser.parse_args()
    log.setLevel(LEVELS[args.verbose])
    log.info("Set logging level: {}".format(LEVEL_NAMES[args.verbose]))

    filepattern = os.path.join(args.source, args.ext)
    images = read_images(filepattern)

    project = os.path.split(args.source)[-1]
    abs_path = os.path.join(os.path.abspath(args.dest), project)
    if not os.path.exists(abs_path):
        os.makedirs(abs_path)

    main(images, abs_path)
