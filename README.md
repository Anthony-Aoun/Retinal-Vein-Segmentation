# Retinal Vein Segmentation

## Project Description
The main goal of this project is to use image segmentation techniques to automatically identify and pinpoint the blood vessels in the retina of the eye. This tool can assist in medical diagnostics and research by providing an automated way to analyze retinal images. The approach combines morphological operations and optimization techniques to achieve accurate segmentation results.

---

## Prerequisites
This program is written in Python 3, which should be installed on your computer.

### Required Dependencies
Ensure the following Python packages are installed:

- **numpy**
- **matplotlib**
- **pillow**
- **scikit-image**
- **scipy**

You can install these packages using pip:
```bash
pip install numpy matplotlib pillow scikit-image scipy
```

### Input Images
The program requires a folder named `images_IOSTAR` at the root of the project directory. This folder must contain the following images and their corresponding annotations:

- Images numbered: **1, 2, 3, 8, 21, 26, 28, 32, 37, 48**

Ensure the images and annotations are properly formatted and named.

---

## Project Structure
The project folder contains the following files:

### Utility Files
1. **[utils.py](utils.py)**
   - Contains utility functions and classes to facilitate image loading and evaluation.
   - **Functions and Classes:**
     - `load_image`: Loads an image by its ID.
     - `evaluate`: Evaluates the quality of a segmentation by comparing it to ground truth.
     - `lineArrayGenerator`: Generates linear structures that cross the plane.

2. **[morphology_utils.py](morphology_utils.py)**
   - Implements morphological operations to aid in the segmentation process.
   - **Functions:**
     - `algebraic_opening`: Performs an algebraic opening on an image.
     - `conjugate_tophats`: Computes a sum of conjugate top-hats on an image.
     - `erosion_reconstruction`: Performs algebraic erosion followed by reconstruction.

### Main Modules
3. **[segmentation.py](segmentation.py)**
   - The main script containing:
     - The segmentation function.
     - The optimization function.
     - Variables for user interaction.

4. **[optimizers.py](optimizers.py)**
   - Contains older optimization functions that may be of use for further exploration.

---

## How to Use

To use the script, configure the boolean and float variables in **segmentation.py** (starting at line 190). Then execute the script **from within the project folder** to ensure proper handling of relative paths.

### Key Variables and Options
- **`do_optimize`**: If `True`, the script will attempt to find the best parameters for segmentation.
- **`mean`**: If `True`, computes the mean metrics for a predefined set of parameters.
- **`example`**: If `True`, performs segmentation on the specified image (`img_nb`).
- **`summary`**: If `True`, displays the segmentation and ground truth (only when `example` is `True`).
- **`watch`**: If `True`, visualizes the different steps of segmentation (only when `example` is `True`).
- **`do_map`**: If `True`, displays the false positive and negative maps (only when `example` is `True`).

### Constraints
Ensure the selected image number (`img_nb`) satisfies the condition:
```python
0 < img_nb < 11
```

### Running the Script
To run the script:
```bash
python segmentation.py
```
Modify the variables as needed within the script for your specific use case.

---

## Additional Notes
- Make sure to properly format and annotate the input images for accurate results.
- This project is designed for research and experimentation. While it can produce valuable results, further refinement may be needed for clinical applications.
- Contributions, suggestions, and bug reports are welcome!

## License
This project is licensed under the MIT License.

## Author
© 2021 [Anthony Aoun](https://github.com/Anthony-Aoun) and [Olivier Laurent](https://github.com/o-laurent). All rights reserved.

This project is open-source and free to use for educational purpouses only.