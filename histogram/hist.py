import cv2
import numpy as np
from matplotlib import pyplot as plt
import webcolors
import matplotlib.patches as mpatch
from scipy.spatial import Voronoi
import skimage.feature
import skimage.exposure

# todo
# how to crawl or download the dataset

def convert_bgr_to_hex(bgr_list):

    # BGR to RGB
    for i, bgr in enumerate(bgr_list):
        bgr_list[i] = np.flipud(bgr)

    # RGB to HEX
    hex_list = []
    for i, rgb in enumerate(bgr_list):
        hex_list.append(str(webcolors.rgb_to_hex(tuple(rgb))))

    return hex_list


def kmeans_bgr(img, K):

    # define criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    # 1. combined channel (or 3 channels? or grayscale?)
    # 2. BGR (or HSV?)
    # img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # reshape image: flatten
    img_flatten = img.reshape((-1, 3))

    # convert to np.float32
    img_reshaped = np.float32(img_flatten)

    # apply kmeans()
    ret, label, center = cv2.kmeans(img_reshaped, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # convert back into uint8 - BGR [0, 255]
    center = np.uint8(center)

    return label, center


def plot_kmeans_bgr_img(img, K, filename):

    # apply kmeans()
    label, center = kmeans_bgr(img, K)

    # average-colored image
    img_center = center[label.flatten()]
    img_center = img_center.reshape((img.shape))
    # save image
    cv2.imwrite(filename, img_center)

    # show image
    # cv2.imshow('Center', img_center)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def plot_kmeans_bgr_hist(img, K, filename):

    # apply kmeans()
    label, center = kmeans_bgr(img, K)

    plt.rcdefaults()
    fig, ax = plt.subplots()

    # BGR to HEX
    hex_list = convert_bgr_to_hex(center)

    # y-axis data
    y_pos = np.arange(len(hex_list))
    performance = [len(label[label == k]) for k in np.arange(K)]

    # bar chart
    bar_list = ax.barh(y_pos, performance, align='center')

    # update color for each bar with HEX
    for bar, hex in zip(bar_list, hex_list):
        bar.set_color(hex)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(hex_list)
    # ax.yaxis.set_tick_params(labelsize=10)
    ax.invert_yaxis()
    ax.set_xlabel('Number of pixels')
    ax.set_title('K-means clustering histogram')

    # save image
    plt.savefig(filename)

    # show image
    # plt.show()


def weighted_mean_kmeans_bgr(img, K):

    # apply kmeans()
    label, center = kmeans_bgr(img, K)

    sum = img.shape[0] * img.shape[1]
    weight = [len(label[label == k])/sum for k in np.arange(K)]
    weight = np.asarray(weight)

    # weighted mean - BGR
    mean_bgr = np.matmul(weight, center)
    mean_bgr = [int(bgr) for bgr in mean_bgr]

    # BGR to HEX
    mean_hex = convert_bgr_to_hex([mean_bgr])

    return mean_bgr, mean_hex


def kmeans_hsv(img, K):

    # define criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    # convert to HSV
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # reshape image: flatten
    img_flatten = img_hsv.reshape((-1, 3))

    # convert to np.float32
    img_reshaped = np.float32(img_flatten)

    # apply kmeans()
    ret, label, center = cv2.kmeans(img_reshaped, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # convert back into uint8 - BGR [0, 255]
    center = np.uint8(center)

    return label, center


def plot_kmeans_hsv_img(img, K, filename):

    # apply kmeans()
    label, center = kmeans_hsv(img, K)

    # average-colored image
    img_center = center[label.flatten()]
    img_center = img_center.reshape((img.shape))
    # save image
    cv2.imwrite(filename, img_center)

    # show image
    # cv2.imshow('Center', img_center)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def plot_kmeans_hsv_hist(img, K, filename):

    # apply kmeans()
    label, center = kmeans_hsv(img, K)

    plt.rcdefaults()
    fig, ax = plt.subplots()

    # BGR to HEX
    hex_list = convert_bgr_to_hex(center)

    # y-axis data
    y_pos = np.arange(len(hex_list))
    performance = [len(label[label == k]) for k in np.arange(K)]

    # bar chart
    bar_list = ax.barh(y_pos, performance, align='center')

    # update color for each bar with HEX
    for bar, hex in zip(bar_list, hex_list):
        bar.set_color(hex)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(hex_list)
    # ax.yaxis.set_tick_params(labelsize=10)
    ax.invert_yaxis()
    ax.set_xlabel('Number of pixels')
    ax.set_title('K-means clustering histogram')

    # save image
    plt.savefig(filename)

    # show image
    # plt.show()


def hist_gray(img):

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # parse GRAY
    # gray = img[:, :, 0]
    # cv2.imshow("Origin", gray_img)
    # cv2.imshow("Gray", gray)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])

    return hist


def plot_hist_gray(hist):

    # plt.hist(img.ravel(), 256, [0, 256])
    # plt.hist(hist, 256, [0, 256])
    # plt.show()

    plt.plot(hist, 'b')
    plt.show()


def hist_bgr(img):

    hist = []

    # parse BGR
    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    cv2.imshow("Origin", img)
    cv2.imshow("B", b)
    cv2.imshow("G", g)
    cv2.imshow("R", r)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # histogram in cv2
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        histr = cv2.calcHist([img], [i], None, [256], [0, 256])
        hist.append(histr)

    # histogram in numpy
    # cv2 is 40x faster than numpy
    # np.bincount() is 10x faster than np.histogram() for one-dimensional histograms
    # hist, bins = np.histogram(img.ravel(), 256, [0, 256])
    # hist = np.bincount(img.ravel(), minlength=256)

    return hist


def plot_hist_bgr(hist):

    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        plt.plot(hist[i], color = col)
        plt.xlim([0, 256])

    plt.show()


def hist_hsv(img):

    hist = []

    # convert BGR to HSV
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # parse HSV
    h, s, v = hsv_img[:, :, 0], hsv_img[:, :, 1], hsv_img[:, :, 2]
    cv2.imshow("Origin", hsv_img)
    cv2.imshow("H", h)
    cv2.imshow("S", s)
    cv2.imshow("V", v)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # histogram in cv2
    hsv = ('h', 's', 'v')
    for i, val in enumerate(hsv):
        histr = cv2.calcHist([hsv_img], [i], None, [256], [0, 256])
        hist.append(histr)

    return hist


def plot_hist_hsv(hist):

    # magenta for h, cyan for s, yellow for v
    color = ('m', 'c', 'y')
    for i, col in enumerate(color):
        plt.plot(hist[i], color = col)
        plt.xlim([0, 256])

    plt.show()


def hist_lab(img):

    hist = []

    # convert BGR to CIELAB
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # parse CIELAB
    l, a, b = lab_img[:, :, 0], lab_img[:, :, 1], lab_img[:, :, 2]
    cv2.imshow("Origin", lab_img)
    cv2.imshow("L", l)
    cv2.imshow("A (Red - Green)", a)
    cv2.imshow("B (Yellow - Blue)", b)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # histogram in cv2
    hsv = ('l', 'a', 'b')
    for i, val in enumerate(hsv):
        histr = cv2.calcHist([lab_img], [i], None, [256], [0, 256])
        hist.append(histr)

    return hist


def plot_hist_lab(hist):

    # cyan for l, r for a(red-green), yellow for b(yellow-blue)
    color = ('c', 'r', 'y')
    for i, col in enumerate(color):
        plt.plot(hist[i], color = col)
        plt.xlim([0, 256])

    plt.show()


def chi_squared_dist(hist, other_hist):

    # hist[0] = hist[1] = hist[2]
    # other_hist[0] = other_hist[1] = other_hist[2]
    num_pixels = int(sum(hist[0]))
    other_num_pixels = int(sum(other_hist[0]))

    # iterate through 3 histograms for 3 channels
    ones = np.ones((256, 1))
    for i in np.arange(3):
        # normalize: cv2.normalize(..., cv2.NORM_L2)
        # cv2.normalize(hist[i], hist[i], 1, 0, cv2.NORM_L2)
        # cv2.normalize(other_hist[i], other_hist[i], 1, 0, cv2.NORM_L2)

        # normalize: sum(bins) = 1 -> sum of all bins
        # avoid being divided by zero -> laplace smooth
        hist[i] = hist[i] + ones
        other_hist[i] = other_hist[i] + ones
        hist[i] = hist[i] / (num_pixels + 256)
        other_hist[i] = other_hist[i] / (other_num_pixels + 256)

    # chi-squared distance between two histograms
    dist = np.zeros((256, 1))
    for i in np.arange(3):
        dist = dist + ((hist[i] - other_hist[i])**2 / (hist[i] + other_hist[i]))

    return sum(dist)


def gaussian_mixture_model():
    # todo
    return


def get_hog(img):

    # ignore the blocks
    # only cells
    # without normalization of histogram

    # float32 array with normalization
    img_arr = np.float32(img) / 255.

    # calculate x and y image derivative (gradient)
    # for color images, the gradients of the three channels are evaluated
    gx = cv2.Sobel(img_arr, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(img_arr, cv2.CV_32F, 0, 1, ksize=1)

    # calculate gradient magnitude and direction (in degrees)
    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

    # normalizing a vector removes the scale
    # histograms are not affected by lighting variations

    # 8 orientations
    orientations = 8
    orientation_delta = 360/orientations
    pixels_per_cell = 8
    x_steps = int(img_arr.shape[1] / pixels_per_cell)
    y_steps = int(img_arr.shape[0] / pixels_per_cell)

    # gradient histogram
    hist = np.zeros(shape=(y_steps,x_steps, orientations))

    # iterate through cells
    for y_step in np.arange(y_steps):
        for x_step in np.arange(x_steps):

            print('cell coordinate', y_step, x_step)

            y_start = y_step * pixels_per_cell
            y_end = (y_start + pixels_per_cell) if (y_start + pixels_per_cell) <= img_arr.shape[0] else img_arr.shape[0]
            x_start = x_step * pixels_per_cell
            x_end = (x_start + pixels_per_cell) if (x_start + pixels_per_cell) <= img_arr.shape[1] else img_arr.shape[1]

            # inside one cell -> (8 x 8) pixels
            hist_cell = np.zeros(shape=(8))
            for y in np.arange(y_start, y_end):
                for x in np.arange(x_start, x_end):

                    print('pixel coordinate', y, x)

                    # get the maximum of the magnitude of gradients of the three channels
                    pixel_mag = mag[y, x]
                    pixel_max_mag_dim = np.argmax(pixel_mag, axis=0)

                    # the angle is the angle corresponding to the maximum magnitude
                    pixel_mag_max = mag[y, x, pixel_max_mag_dim]
                    pixel_angle_max = angle[y, x, pixel_max_mag_dim]

                    for orientation in np.arange(orientations):
                        if pixel_angle_max >= orientation*orientation_delta and pixel_angle_max < (orientation+1)*orientation_delta:
                            hist_cell[orientation] += pixel_mag_max

            # update the histogram
            hist[y_step, x_step] = hist_cell

    return hist


def get_hog_skimage(img):

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fd, img_hog = skimage.feature.hog(img_rgb, orientations=8, pixels_per_cell=(16, 16),
                                      cells_per_block=(1, 1), visualize=True, multichannel=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

    ax1.axis('off')
    ax1.imshow(img_rgb)
    ax1.set_title('input image')

    img_hog_rescaled = skimage.exposure.rescale_intensity(img_hog, in_range=(0, 10))

    ax2.axis('off')
    ax2.imshow(img_hog_rescaled)
    ax2.set_title('histogram of oriented gradients')
    plt.show()


def get_hog_cv2(img):

    winSize = (64, 64)
    blockSize = (16, 16)
    blockStride = (8, 8)
    cellSize = (8, 8)
    nbins = 9
    derivAperture = 1
    winSigma = 4.
    histogramNormType = 0
    L2HysThreshold = 2.0000000000000001e-01
    gammaCorrection = 0
    nlevels = 64

    hog_descriptor = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
                                       histogramNormType, L2HysThreshold, gammaCorrection, nlevels)

    winStride = (8, 8)
    padding = (8, 8)
    locations = ((10, 20),)

    hist = hog_descriptor.compute(img, winStride, padding, locations)

    return hist


def get_orb(img):

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create()

    # find the keypoints and descriptors with ORB
    kp, des = orb.detectAndCompute(img_gray, None)

    return kp, des


def plot_matcher_orb(img1, img2):

    kp1, des1 = get_orb(img1)
    kp2, des2 = get_orb(img2)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # match descriptors
    matches = bf.match(des1, des2)

    # sort them in the order of their distance
    matches = sorted(matches, key=lambda x: x.distance)

    # draw first 10 matches.
    img3 = cv2.drawMatches(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB), kp1,
                           cv2.cvtColor(img2, cv2.COLOR_BGR2RGB), kp2,
                           matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # save image
    # cv2.imwrite('./pix/orb_matcher_keypoints.jpg', img3)

    # show image
    plt.imshow(img3)
    plt.show()


def get_sift(img):

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp = sift.detect(img_gray, None)

    # computes the descriptors
    # kp is a list of keypoints
    # des is a numpy array of shape (number_of_keypoints × 128)
    kp, des = sift.compute(img_gray, kp)

    # draw keypoints
    # img_sift = cv2.drawKeypoints(img_gray, kp, img,
    #                              flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # save image
    # cv2.imwrite('./pix/sift_keypoints.jpg', img_sift)

    # show image
    # plt.imshow(img_sift)
    # plt.show()

    return kp, des


def plot_matcher_sift(img1, img2):

    kp1, des1 = get_sift(img1)
    kp2, des2 = get_sift(img2)

    # BFMatcher with default params
    bf = cv2.BFMatcher()

    # knn matches
    # matches = bf.knnMatch(des1, des2, k=2)

    # apply ratio test
    # good = []
    # for m, n in matches:
    #     if m.distance < 0.75 * n.distance:
    #         good.append([m])

    # cv.drawMatchesKnn expects list of lists as matches
    # img3 = cv2.drawMatchesKnn(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), kp1,
    #                           cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), kp2,
    #                           good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # match descriptors
    matches = bf.match(des1, des2)
    # sort them in the order of their distance
    matches = sorted(matches, key=lambda x: x.distance)
    # draw first 10 matches.
    img3 = cv2.drawMatches(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB), kp1,
                           cv2.cvtColor(img2, cv2.COLOR_BGR2RGB), kp2,
                           matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # save image
    # cv2.imwrite('./pix/sift_matcher_keypoints.jpg', img3)

    # show image
    plt.imshow(img3)
    plt.show()

if __name__ == '__main__':

    # load image
    thomas = cv2.imread('./data/mw02128.jpg')
    thomas1 = cv2.imread('./data/thomas.jpg')
    richard_iii = cv2.imread('./data/mw05305.jpg')

    # hog
    # hist = get_hog(thomas)

    # orb
    # plot_matcher_orb(thomas, thomas1)

    # sift
    # get_sift(thomas)
    # plot_matcher_sift(thomas, thomas1)

    # where is Waldo?
    # waldo = cv2.imread('./data/waldo.jpg')
    # scene = cv2.imread('./data/waldo_track.jpg')
    # plot_matcher_sift(waldo, scene)

    # kmeans
    # label, center = kmeans_bgr(thomas, 8)

    # plot kmeans - bgr
    plot_kmeans_bgr_img(richard_iii, 8, './output/richard8.png')
    plot_kmeans_bgr_hist(richard_iii, 8, './output/richardhist.png')

    plot_kmeans_bgr_img(thomas, 8, './output/thomas8.png')
    plot_kmeans_bgr_hist(thomas, 8, './output/thomashist.png')

    # weighted mean of kmeans - bgr
    # mean_bgr, mean_hex = weighted_mean_kmeans_bgr(thomas, 8)
    # print(mean_bgr)
    # print(mean_hex)

    # plot kmeans - hsv
    # plot_kmeans_hsv_img(thomas, 8, './output/thomas8hsv.png')
    # plot_kmeans_hsv_hist(thomas, 8, './output/thomashisthsv.png')

    # plot histogram - gray
    # hist = histogram_gray(img1)
    # plot_histogram_gray(hist)

    # plot histogram - bgr
    # hist = hist_bgr(thomas)
    # plot_hist_bgr(hist)

    # plot histogram - hsv
    # hist = hist_hsv(thomas)
    # plot_hist_hsv(hist)

    # plot histogram - cielab
    # hist = hist_lab(thomas)
    # plot_hist_lab(hist)

    # chi-squared distance
    # dist = chi_squared_dist(hist, other_hist)
    # print(dist)
