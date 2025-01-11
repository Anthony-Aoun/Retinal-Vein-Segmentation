""" Utilities for MI206 Project

This file can be imported as a module and contains the following functions:
    * load_image: load an image with its id
    * evaluate: evaluate the quality of the segmentation

It also contains the following class:
    * lineArrayGenerator: Objects which creates linear structures to cross the plane


This file has been developed by Anthony Aoun and Olivier Laurent.
"""

import numpy as np
from skimage.morphology import thin

from PIL import Image, ImageEnhance


def load_image(img_id):
    """
    Loading image function. 
    The image is defined by a number between 1 and 10 (both included).

    Parameters
    ----------
    img_id: int between 1 and 10
        The id of the image to be loaded.

    Returns
    -------
    np.ndarray:
        The img to be segmented.
    np.ndarray:
        The mask of the image - the useful and useless pixels.
    np.ndarray:
        The ground truth image.
    """
    assert img_id > 0, "L'identifiant de l'image doit être supérieur ou égal à 1."
    assert img_id < 11, "L'identifiant de l'image doit être inférieur ou égal à 10."

    if img_id in [1, 2, 3]:
        true_img_id = img_id
    elif img_id == 4:
        true_img_id = 8
    elif img_id == 5:
        true_img_id = 21
    elif img_id == 6:
        true_img_id = 26
    elif img_id == 7:
        true_img_id = 28
    elif img_id == 8:
        true_img_id = 32
    elif img_id == 9:
        true_img_id = 37
    elif img_id == 10:
        true_img_id = 48

    if true_img_id in [1, 2, 21]:
        end = "OSC"
    elif true_img_id in [3, 8, 48]:
        end = "OSN"
    elif true_img_id in [26, 32]:
        end = "ODC"
    else:
        end = "ODN"
    if true_img_id < 10:
        img_path = "./images_IOSTAR/star0"+str(true_img_id)+"_" + end + ".jpg"
        img_path_GT = "./images_IOSTAR/GT_0"+str(true_img_id)+".png"
    else:
        img_path = "./images_IOSTAR/star"+str(true_img_id)+"_" + end + ".jpg"
        img_path_GT = "./images_IOSTAR/GT_"+str(true_img_id)+".png"

    img_GT = np.asarray(Image.open(img_path_GT)).astype(np.bool_)

    img = np.asarray(Image.open(img_path)).astype(np.uint8)
    nrows, ncols = img.shape
    row, col = np.ogrid[:nrows, :ncols]

    # On ne considere que les pixels dans le disque inscrit
    img_mask = (np.ones(img.shape)).astype(np.bool_)
    invalid_pixels = ((row - nrows / 2)**2 +
                      (col - ncols / 2)**2 > (nrows / 2)**2)
    img_mask[invalid_pixels] = 0
    return img, img_mask, img_GT


def evaluate(img_out, img_GT, do_map: bool = False):
    """
    Evaluate is a function which measures how good the image has been segmented.
    The criterium is imposed by the exercise.

    Parameters
    ----------
    img_out: np.ndarray<float||int> (square)
        The input image to evaluate.
    img_GT: np.ndarray<float||int> (square - same size as img_out)
        The ground-truth image to compare to img_out.
    do_map: bool, default is False
        if do_map is true, return the false positive and false negative images.

    Returns
    -------
    float 
        The accuracy: the part of real positives in the found positives.
    float
        The recall: the part of found positives in the total number of positives.
    float
        The F1 score. Defined by 2/(1/Acc+1/Rec).
    np.ndarray<int>
        The skeleton of the proposed image.
    np.ndarray<int>
        The skeleton of the ground truth image.
    np.ndarray<int>
        The map of false positives. Only if do_map is True.
    np.ndarray<int>
        The map of false negatives. Only if do_map is True.
    """
    GT_skel = thin(img_GT, max_iter=15)
    img_out_skel = thin(img_out, max_iter=15)

    TP = np.sum(img_GT & img_out)  # Vrais positifs
    if not do_map:
        FP = np.sum(img_out_skel & ~img_GT)  # Faux positifs (relaxes)
        FN = np.sum(GT_skel & ~img_out)  # Faux negatifs (relaxes)
    else:
        FP_map = img_out_skel & ~img_GT
        FN_map = GT_skel & ~img_out
        FP = np.sum(FP_map)
        FN = np.sum(FN_map)

    ACCU = TP / (TP + FP)  # Precision
    RECALL = TP / (TP + FN)  # Rappel

    if TP != 0:
        # F1 score - same weight for both measures
        F1 = 2 / (1/RECALL + 1/ACCU)
    else:
        F1 = 0
        print("Erreur lors du script. Résultat incohérent.")

    if not do_map:
        return ACCU, RECALL, F1, img_out_skel, GT_skel
    else:
        return ACCU, RECALL, F1, img_out_skel, GT_skel, FP_map, FN_map


class lineArrayGenerator():
    """
    Class to build linear structures to cross the plane.
    Instantiate the class with one of the constructors and create linear structures with the angles which you want.

    Attributes
    ----------
    _line_length: int
        The length of the linear structures. Is 2n+1 with n in N.
    _struct_number: int
        The number of element to cross the plane with.
    _structs: list 
        The linear structures.
    _angles: list
        The angles corresponding to the structures.

    Methods
    -------
    _line(angle: int = 0)
        Create a structure for the builder with the specified angle.
    _create_structs()
        Create the structures of the builder.
    _get_structs()
        Get the structures of the builder.
    _get_angles()
        Get the angle of the structures of the builder.


    Developed by Anthony Aoun and Olivier Laurent. 
    """

    def __init__(self, line_lengths: list, struct_number: int = 12):
        """Parametered constructor

        Parameters
        ----------
        line_length: int, optional (7)
            The array of the lengths of the linear structures. Should be a list of integers of form 2n+1 with n in N.
        struct_number: int, optional (12)
            The number of element to cross the plane with.
        """
        if not isinstance(line_lengths, list):
            raise ValueError("The first argument should be a list of integers")
        for line_length in line_lengths:
            if not isinstance(line_length, int):
                raise ValueError(
                    "The first argument should be a list of integers")
            assert line_length % 2 == 1, "Erreur. La longueur de chaque ligne doit être de la forme 2n+1."

        self._line_lengths = line_lengths
        self._struct_number = struct_number
        self._structs = []
        self._angles = []
        self._create_structs()

    def _line(self, length: int = 7, angle: int = 0):
        """Create a structure for the builder with the specified angle.

        Parameters
        ----------
        length: int
            The length of the structuring element. Default is 7.
        angle: float 
            The angle of the structure, 0 being horizontal. Default is 0.

        Returns
        -------
        np.ndarray<int>
            The linear structure created by the function.
        """
        angle = angle % 180

        n = length//2
        if angle > 135:
            angle_rad = np.deg2rad(180-angle)
        elif angle > 90:
            angle_rad = np.deg2rad(angle-90)
        elif angle > 45:
            angle_rad = np.deg2rad(90-angle)
        else:
            angle_rad = np.deg2rad(angle)

        line_array = np.zeros((length, length))
        values = (np.round(np.arange(-n, n+1) * np.tan(angle_rad))).astype(int)
        for idv, val in enumerate(values):
            if -n <= val <= n:
                line_array[-val+n, idv] = 1

        if angle > 135:
            line_array = np.flip(line_array, axis=1)
        elif angle > 90:
            line_array = np.flip(line_array, axis=1)
            line_array = line_array.transpose()
        elif angle > 45:
            line_array = line_array.transpose()
        return line_array

    def _create_structs(self):
        """Create the structures of the builder."""
        self._structs = {}
        steps = 180//self._struct_number
        for length in self._line_lengths:
            self._structs[length] = []
            for i in range(0, 180, steps):
                self._structs[length].append(self._line(length, i))
                self._angles.append(i)

    def get_structs(self):
        """Get the structures of the builder.

        Returns
        -------
        list<np.ndarray>
            The linear structures.
        """
        return self._structs

    def get_angles(self):
        """Get the angle of the structures of the builder.

        Returns
        -------
        list<float>
            The angles corresponding to the linear structures.
        """
        return self._angles
