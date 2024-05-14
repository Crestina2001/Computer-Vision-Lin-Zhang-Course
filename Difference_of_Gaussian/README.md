# Gaussian Pyramid and Difference of Gaussian (DoG) Image Generation

This project demonstrates how to generate Gaussian pyramid images and Difference of Gaussian (DoG) images using OpenCV in Python. The resulting images are saved to disk for further analysis or visualization.

## Table of Contents
- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Code Explanation](#code-explanation)
- [Results](#results)
- [Contributing](#contributing)

## Overview

In this project, we employ OpenCV and NumPy to create Gaussian pyramids and Difference of Gaussian (DoG) images. Gaussian pyramids are used in image processing for multi-scale signal representation, and DoG images are useful for edge detection and feature extraction.

## Requirements

- Python 3.x
- OpenCV
- NumPy
- Matplotlib (for visualization)

## Installation

To install the required packages, use pip:

```bash
pip install opencv-python-headless numpy matplotlib
```

## Usage

1. Clone the repository:

```bash
git clone https://github.com/Crestina2001/Computer-Vision-Lin-Zhang-Course
cd Difference_of_Gaussian
```

2. Run the Jupyter Notebook:

```bash
jupyter notebook dog.ipynb
```

3. Follow the instructions in the notebook to generate and save the images.

## Code Explanation

### Gaussian Pyramid Generation

The code generates a Gaussian pyramid by progressively applying Gaussian blur to a base image and downsampling it. The blurred images are stored in a list of lists, where each sublist represents an octave.

```python
# Generate Gaussian images
initial_blur = sigma
image = baseImage
gaussianImages = []

for i in range(numOctave):
    octave_i = []
    # Put the first image into the ith octave
    octave_i.append(image)
    print("Initial Blur of Octave " + str(i + 1) + ": " + str(initial_blur))
    blur_i = []
    blur_i.append(initial_blur)
    for kernel in gaussianKernels[1:]:
        image = cv.GaussianBlur(image, (0, 0), sigmaX=kernel)
        initial_blur = np.sqrt(initial_blur**2 + kernel**2)
        blur_i.append(initial_blur)
        octave_i.append(image)
    gaussianImages.append(octave_i)
    octaveNext = octave_i[-3]
    initial_blur = blur_i[-3]
    rows, cols = (int(octaveNext.shape[1] / 2), int(octaveNext.shape[0] / 2))
    initial_blur = initial_blur / 2
    image = cv.resize(octaveNext, (rows, cols), interpolation=cv.INTER_NEAREST)

# Save the Gaussian images
for i, octave in enumerate(gaussianImages):
    for j, img in enumerate(octave):
        path = f"Pyramid/octave_{i + 1}_image_{j + 1}.jpg"
        cv.imwrite(path, img)
```

### Difference of Gaussian (DoG) Calculation

The DoG images are computed by subtracting consecutive Gaussian blurred images within each octave. The resulting images highlight edges and other features.

```python
# Generate DOG images
dogImages = []
for octave_i in gaussianImages:
    dog_i = []
    for first_image, second_image in zip(octave_i[:-1], octave_i[1:]):
        dog_i.append(np.subtract(second_image, first_image))
    dogImages.append(dog_i)

# Convert dogImages to a list of arrays
dogImages = [np.array(dog_i) for dog_i in dogImages]

# Save the DoG images
for i, octave in enumerate(dogImages):
    for j, dog_image in enumerate(octave):
        path = f"Dog_Pyramid/octave_{i+1}_dog_{j+1}.jpg"
        cv.imwrite(path, (dog_image * 255).astype(np.uint8))
```

## Results

The generated Gaussian and DoG images are saved in the `Pyramid` and `Dog_Pyramid` directories, respectively. Each image is named according to its octave and position within the octave.
