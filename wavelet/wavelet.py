import os
import sys
import cv2 as cv
import numpy as np

def calc_texture_flow(indir, outdir, fn):

    img = cv.imread(cv.samples.findFile(os.path.join(indir, fn)))

    if img is None:
        print('Failed to load image file:', fn)
        sys.exit(1)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    h, w = img.shape[:2]

    eigen = cv.cornerEigenValsAndVecs(gray, 15, 3)
    eigen = eigen.reshape(h, w, 3, 2)  # [[e1, e2], v1, v2]
    flow = eigen[:, :, 2]

    vis = img.copy()
    vis[:] = (192 + np.uint32(vis)) / 2
    d = 12
    points = np.dstack(np.mgrid[d / 2:w:d, d / 2:h:d]).reshape(-1, 2)
    for x, y in np.int32(points):
        vx, vy = np.int32(flow[y, x] * d)
        cv.line(vis, (x - vx, y - vy), (x + vx, y + vy), (0, 0, 0), 1, cv.LINE_AA)

    # display
    # cv.imshow('flow', vis)
    # cv.waitKey()
    # cv.destroyAllWindows()

    # save
    cv.imwrite(os.path.join(outdir, fn), vis)


if __name__ == '__main__':

    calc_texture_flow(indir='data', outdir='output', fn='green_field.jpg')