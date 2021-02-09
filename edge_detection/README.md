# Edge detection

## Canny Edge Detector

<p float="left">
    <img src="./data/calling-edges.png" width="800" />
</p>

## HOG

<p float="left">
    <img src="./data/calling-hog.png" width="800" />
</p>

## CNN

### Model

The CNN model trained to classify portraits and landscapes reaches 0.96 test accuracy with:
* Training data: 13000 landscapes + 13000 portraits (0.2 validation split)
* Test data: 2000 landscapes + 2000 portraits

<p float="left">
    <img src="pix/cnn_accuracy_rgb.png" width="400" />
    <img src="pix/cnn_loss_rgb.png" width="400" />
</p>

### Filter

The filter of the second convolutional layer is shown below:

<p float="left">
    <img src="pix/separable_conv2d_1_stiched_filters_grayscale.png" width="800" />
</p>

### Activation

The activations of filter index 3 and 5 of the second convolutional layer are shown below for landscape:

<p float="left">
    <img src="pix/landscape_tensor.png" width="300" />
    <img src="pix/landscape_conv2d_1_filter_3.png" width="300" />
    <img src="pix/landscape_conv2d_1_filter_5.png" width="300" />
</p>

The activations of filter index 3 and 5 of the second convolutional layer are shown below for portrait:

<p float="left">
    <img src="pix/portrait_tensor.png" width="300" />
    <img src="pix/portrait_conv2d_1_filter_3.png" width="300" />
    <img src="pix/portrait_conv2d_1_filter_5.png" width="300" />
</p>

## References
* https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_canny/py_canny.html
* https://scikit-image.org/docs/dev/api/skimage.filters.html