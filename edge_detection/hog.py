import cv2
import skimage.feature
import skimage.exposure
import os
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from colour import Color

infile = os.path.join('data', 'calling.png')
outfile = os.path.join('data', 'calling-hog.png')

img = cv2.imread(infile)
height, width, _ = img.shape

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fd, img_hog = skimage.feature.hog(img_rgb, orientations=4, pixels_per_cell=(16, 16),
                                  cells_per_block=(2, 2), visualize=True, multichannel=True)

# https://www.infobyip.com/detectmonitordpi.php
my_dpi = 192

plt.gca().set_axis_off()
plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
plt.margins(0,0)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())

ramp_colors = ['#000000', '#FFFFFF']
my_cmap = LinearSegmentedColormap.from_list( 'my_cmap', [Color(color).rgb for color in ramp_colors])

plt.figure(figsize=(width/my_dpi, height/my_dpi), dpi=my_dpi)
plt.imshow(img_hog, cmap=my_cmap, vmin=0, vmax=1)
# plt.imshow(img_hog, cmap=my_cmap)
plt.axis('off')
plt.savefig(outfile, dpi=my_dpi, bbox_inches = 'tight', pad_inches = 0)