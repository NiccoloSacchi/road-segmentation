
from itertools import chain, product, permutations
from functools import partial
from scipy.ndimage.interpolation import rotate
import logging


import numpy as np

logger = logging.getLogger(__name__)

def cube_group_action(transpose, mirror, arr):
    """
    `transpose` is a permutation of the axes (0, 1, 2...)

    `mirror` is a sequence of values in {0, 1} indicating mirroring in those
    axes after transposition
    """

    # transpose = list(transpose) + list(range(len(transpose), arr.ndim))
    extra_dims = arr.ndim - len(transpose)
    transpose = list(range(extra_dims)) + [i + extra_dims for i in transpose]
    arr = arr.transpose(*transpose)
    mirror_slices = (slice(None), ) * extra_dims + tuple(slice(None, None, -1 if i else None) for i in mirror)
    return arr[mirror_slices]

def all_transformations(ndim):
    transpositions = permutations(range(ndim))
    mirrors = product(*([0, 1],) * ndim)
    args = list(product(transpositions, mirrors))
    return [partial(cube_group_action, *arg) for arg in args]

def d4_transformations(ndim):
    extra_dims = ndim - 2
    transpositions = [tuple(xrange(extra_dims)) + i for i in permutations([extra_dims, extra_dims + 1])]
    mirrors = [((0,) * extra_dims) + i for i in product([0, 1], [0, 1])]
    args = list(product(transpositions, mirrors))
    return [partial(cube_group_action, *arg) for arg in args]


def crop_central(x,h_rand,l_rand):

    size = 384
    l    = x.shape[-2]
    h    = x.shape[-1]

    l_s = int(np.floor((l-size)/2*l_rand))
    l_e = int(l_s+size)
    h_s = int(np.floor((h-size)/2*h_rand))
    h_e = int(h_s+size)

    return x[...,l_s:l_e,h_s:h_e]

def rotation(x, deg):
    return rotate(x,angle = deg,axes=(-1,-2),reshape=False,order=1,cval=2);
