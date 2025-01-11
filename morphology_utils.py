"""This file contains utilities for morphology operators

It contains 3 functions:
    * algebraic_opening: performs an algebraic opening of an image
    * conjugate_tophats: performs a sum of conjugate top-hats of an image
    * erosion_reconstruction: performs an algebraic erosion of an image
        followed by a reconstruction


This file has been developed by Anthony Aoun and Olivier Laurent
"""

import numpy as np

from skimage.morphology import black_tophat, erosion, opening, reconstruction


def algebraic_opening(img, structures):
    """Function which performs an algebraic opening of an image with the chosen structuring elements

    Parameters
    ----------
    img : np.ndarray<float>
        The input image
    structures : np.ndarray<float>
        The structuring elements

    Returns
    -------
    np.ndarray
        The output image
    """
    opd_imgs = []
    for struct in structures:
        opd_imgs.append(opening(img, struct))
    return np.maximum.reduce(opd_imgs)


def conjugate_tophats(img, structures):
    """Funtion which performs a sum of conjugate top-hats of an image with the chose structuring elements

    Parameters
    ----------
    img : np.ndarray<float>
        The input image
    structures : np.ndarray<float>
        The structuring elements

    Returns
    -------
    img_th : np.ndarray<float>
        The output image
    """
    img_th = np.zeros(img.shape)
    for struct in structures:
        img_th += black_tophat(img, struct)
    return img_th


def erosion_reconstruction(img, structures):
    """Funtion which performs an algebraic erosion of an image with the chosen structuring elements 
        followed by a reconstruction

    Parameters
    ----------
    img : np.ndarray<float>
        The input image
    structures : np.ndarray<float>
        The structuring elements


    Returns
    -------
    img_erd : np.ndarray<float>
        The image after the erosion
    img_rec : np.ndarray<float>
        The image after the reconstruction
    """
    erds_img = []
    for struct in structures:
        erds_img.append(erosion(img, struct))

    img_erd = np.maximum.reduce(erds_img)
    img_rec = reconstruction(img_erd, img)

    return img_erd, img_rec
