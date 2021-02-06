import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

# infile = os.path.join('data', 'musician.jpg')
infile = os.path.join('data', 'calling.png')
outfile = os.path.join('data', 'calling-edges.png')

img = cv2.imread(infile, 0)
height, width = img.shape
# edges	= cv2.Canny(image, threshold1, threshold2)
edges = cv2.Canny(img, 100, 200)

# Save the edge image in the original shape
# https://www.infobyip.com/detectmonitordpi.php
my_dpi = 192

# With margin
# plt.figure(figsize=(width/my_dpi, height/my_dpi), dpi=my_dpi)
# plt.imshow(edges, cmap='gray')
# plt.axis('off')
# plt.savefig(outfile, dpi=my_dpi)

# Without margin
plt.gca().set_axis_off()
plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
plt.margins(0,0)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())

plt.figure(figsize=(width/my_dpi, height/my_dpi), dpi=my_dpi)
plt.imshow(edges, cmap='gray')
plt.axis('off')
plt.savefig(outfile, dpi=my_dpi, bbox_inches = 'tight', pad_inches = 0)

# Show the two images side by side
# plt.subplot(121),plt.imshow(img,cmap = 'gray')
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(edges,cmap = 'gray')
# plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
# plt.show()