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

The CNN model trained to classify portraits and landscapes reaches 0.96 test accuracy with:
* Training data: 13000 landscapes + 13000 portraits (0.2 validation split)
* Test data: 2000 landscapes + 2000 portraits

<p float="left">
    <img src="./pix/cnn_accuracy.png" width="400" />
    <img src="./pix/cnn_loss.png" width="400" />
</p>

## References
* https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_canny/py_canny.html