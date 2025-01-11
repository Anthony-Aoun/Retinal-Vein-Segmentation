"""Main algorithm

It contains the segmentation function and the optimization function as well as variables for user interaction.

Developed by Anthony Aoun and Olivier Laurent
"""

import numpy as np

from PIL import Image, ImageEnhance

from skimage import filters
from skimage.morphology import erosion, opening, reconstruction, black_tophat
from skimage.morphology import disk

from matplotlib import pyplot as plt

import optuna

from utils import load_image, evaluate, lineArrayGenerator
from morphology_utils import algebraic_opening, conjugate_tophats, erosion_reconstruction


def img_segmentation(img, img_mask, low_threshold: float = 80, high_threshold: float = 80, contrast: float = 2.0, len_struct_1: int = 15,
                     len_struct_2: int = 11, len_struct_3: int = 19, sigma: float = 2.0, watch: bool = False):
    """
    Main segmentation function

    Parameters
    ----------
    img_out: np.ndarray<float||int> (square)
        The input image to evaluate
    img_mask: np.ndarray<bool> (square - same size as img_out)
        The ground-truth image to compare to img_out
    low_threshold: float, optional (80)
        The low threshold for the hysteresis thresholding.
    high_threshold: float, optional (80)
        The high threshold for the hysteresis thresholding.
    contrast: float, optional (2.0)
        The enhancement of contrast to apply to the image.
    len_struct_1 : int, optional
        The length of the first structuring elements, by default 17
    len_struct_2 : int, optional
        The length of the second structuring elements, by default 7
    len_struct_3 : int, optional
        The length of the third structuring elements, by default 15
    sigma : float, optional
        The standard deviation of the final gaussian filter, by default 2.0
    watch: bool, optional (False)
        if watch is true, plot the different steps of the processing.

    Returns
    -------
    img_out : np.ndarray
        The segmented image.
    """

    # image brightness enhancer to detect the small details
    im = Image.fromarray(img)
    enhancer = ImageEnhance.Contrast(im)
    im = enhancer.enhance(contrast)
    img = np.asarray(im).astype(int)

    img_op = algebraic_opening(img, structures[len_struct_1])

    if watch:
        plt.subplot(231)
        plt.imshow(img_op)
        plt.title("After Opening")
        plt.axis('off')

    img_th = conjugate_tophats(img_op, structures[len_struct_2])

    if watch:
        plt.subplot(232)
        plt.imshow(img_th)
        plt.title("After Top-Hats")
        plt.axis('off')

    img_erd, img_rec = erosion_reconstruction(img_th, structures[len_struct_3])

    if watch:
        plt.subplot(233)
        plt.imshow(img_erd)
        plt.title("After Erosion")
        plt.axis('off')

        plt.subplot(234)
        plt.imshow(img_rec)
        plt.title("After Reconstruction")
        plt.axis('off')

    fld_imd = filters.gaussian(img_rec, sigma)

    final_img = filters.apply_hysteresis_threshold(
        fld_imd, low_threshold, high_threshold)

    if watch:
        plt.subplot(235)
        plt.imshow(fld_imd)
        plt.title("Filtered Image")
        plt.axis('off')

        plt.subplot(236)
        plt.imshow(final_img.astype(bool))
        plt.title("Final Image")
        plt.axis('off')

    img_out = (img_mask & final_img.astype(bool))

    img_out[0:12, :] = False
    img_out[501:513, :] = False
    img_out[:, 0:12] = False
    img_out[:, 501:513] = False

    return img_out


def mean_results(high_threshold: float = 76.6, lower: float = 22.1, contrast: float = 1.82, len_struct_1: int = 17,
                 len_struct_2: int = 7, len_struct_3: int = 15, sigma: float = 2, verbose: bool = False):
    """Evalutes the mean results for some defined parameters

    Parameters
    ----------
    high_threshold : float, optional
        The high threshold of the hysteresis, by default 76.6
    lower : float, optional
        The difference between the high and the low thresholds, by default 22.1
    contrast : float, optional
        The contrast enhancement, by default 1.82
    len_struct_1 : int, optional
        The length of the first structuring elements, by default 17
    len_struct_2 : int, optional
        The length of the second structuring elements, by default 7
    len_struct_3 : int, optional
        The length of the third structuring elements, by default 15
    sigma : float, optional
        The standard deviation of the final gaussian filter, by default 2.0
    verbose : bool, optional
        If true, prints the values of the , by default False

    Returns
    -------
    F1 : float
        The mean F1 score
    """

    F1 = 0
    ACC = 0
    REC = 0
    for i in range(1, 11):
        img, img_mask, img_GT = load_image(i)
        img_out = img_segmentation(
            img, img_mask, high_threshold-lower, high_threshold, contrast, len_struct_1, len_struct_2, len_struct_3, sigma, False)
        cACC, cREC, cF1, _, _ = evaluate(img_out, img_GT, False)
        F1 += cF1
        ACC += cACC
        REC += cREC
        if verbose:
            print(i, 'Image Accuracy =', cACC,
                  ', Image Recall =', cREC, ', Image F1 =', cF1)
    F1 /= 10
    ACC /= 10
    REC /= 10

    if verbose:
        print('Mean Accuracy =', ACC, ', Mean Recall =', REC, ', Mean F1 =', F1)
    return F1


def optimize(trial):
    """Wrapper for optuna"""
    threshold = trial.suggest_uniform('threshold', 60, 85)  # 100
    lower = trial.suggest_uniform('lower', 10, 35)
    contrast = trial.suggest_uniform('contrast', 0.6, 2.5)
    len_struct_1 = trial.suggest_int('len_struct_1', 11, 21, 2)
    len_struct_2 = trial.suggest_int('len_struct_2', 5, 11, 2)
    len_struct_3 = trial.suggest_int('len_struct_3', 11, 19, 2)
    sigma = trial.suggest_uniform('sigma', 0.8, 3)
    return -mean_results(threshold, lower, contrast, len_struct_1, len_struct_2, len_struct_3, sigma)


gen = lineArrayGenerator(list(range(5, 30, 2)), 18)

structures = gen.get_structs()


do_optimize = False  # try to find the best parameters
mean = False  # compute the mean metrics for the defined set of parameters
exemple = True  # compute the segmentation for the image img_nb

summary = True  # see the segmentation and the ground truth
watch = True  # see the different steps of the segmentation
do_map = True  # see the false positives and negatives maps

img_nb = 1

# gives a mean of 0.9512 - redefine these parameters if needed and exemple/mean is True & opt is False
threshold = 64.0953
lower = 15.3577
contrast = 1.73199
len_struct_1 = 15
len_struct_2 = 7
len_struct_3 = 15
sigma = 1.558186

if do_optimize:
    study = optuna.create_study()
    study.optimize(optimize, n_jobs=7, n_trials=200)
    print(study.best_params)
    threshold = study.best_params['threshold']
    lower = study.best_params['lower']
    constrast = study.best_params['contrast']
    len_struct_1 = study.best_params['len_struct_1']
    len_struct_2 = study.best_params['len_struct_2']
    len_struct_3 = study.best_params['len_struct_3']
    sigma = study.best_params['sigma']
if exemple:
    img, img_mask, img_GT = load_image(img_nb)
    img_out = img_segmentation(
        img, img_mask, threshold-lower, threshold, contrast, len_struct_1, len_struct_2, len_struct_3, sigma, watch)

    if watch:
        plt.show()

    if do_map:
        ACCU, RECALL, F1, img_out_skel, GT_skel, FP_map, FN_map = evaluate(
            img_out, img_GT, do_map)
    else:
        ACCU, RECALL, F1, img_out_skel, GT_skel = evaluate(
            img_out, img_GT, do_map)

    print('Accuracy =', ACCU, ', Recall =', RECALL, ', F1 =', F1)

    if do_map:
        plt.subplot(121)
        plt.imshow(FP_map)
        plt.title("False Positives Map")
        plt.axis('off')

        plt.subplot(122)
        plt.imshow(FN_map)
        plt.title("False Negatives Map")
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    if summary:
        plt.subplot(231)
        plt.imshow(img, cmap='gray')
        plt.title('Original Image')
        plt.axis('off')
        plt.subplot(232)
        plt.imshow(img_out)
        plt.title('Segmentation')
        plt.axis('off')
        plt.subplot(233)
        plt.imshow(img_out_skel)
        plt.title('Segmentation Skeleton')
        plt.axis('off')
        plt.subplot(235)
        plt.imshow(img_GT)
        plt.title('Ground Truth')
        plt.axis('off')
        plt.subplot(236)
        plt.imshow(GT_skel)
        plt.title('Ground Truth Skeleton')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
if mean:
    mean_results(threshold, lower, contrast, len_struct_1,
                 len_struct_2, len_struct_3, sigma, True)
