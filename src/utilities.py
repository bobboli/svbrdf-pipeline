import os
from time import time
import numpy as np
from scipy import optimize
import sys
from matplotlib import pyplot as plt
import cv2
import math
import cmath
from sklearn.utils import check_random_state

_delta = 1e-9

def ImgInt2Float(img, dtype=np.float):
    return img.astype(dtype) / 255.0

def ImgFloat2Int(img):
    return (img*255.0).astype(np.uint8)

def ReadImg(fileName, gammaDecoding=True, grayScale=False):
    if os.path.splitext(fileName)[-1].lower() == '.pfm':
        assert gammaDecoding == False, 'You should expect colors stored in PFM to be in linear space!'
        return ReadPfm(fileName)

    if grayScale:
        imgData = cv2.imread(fileName, flags=cv2.IMREAD_GRAYSCALE)
    else:
        imgData = cv2.imread(fileName)
    if imgData is None:
        return None
    imgData = ImgInt2Float(imgData)
    if gammaDecoding:
        imgData = imgData ** 2.2
    if not grayScale:
        return imgData[:, :, ::-1]
    return imgData

def ReadImgAsUint8(fileName):
    return cv2.imread(fileName)[:, :, ::-1]

def WriteImgAsUint8(fileName, imgData):
    return cv2.imwrite(fileName, imgData[:,:,::-1])


# Note: All material properties should be stored in linear space (including "basecolor")
def WriteImg(fileName, imgData, gammaEncoding=True):
    if os.path.splitext(fileName)[-1].lower() == '.pfm':
        assert gammaEncoding == False, 'You should store colors in linear space to PFM files!'
        WritePfm(fileName, imgData)
        return
    if gammaEncoding:
        imgData = imgData ** (1.0 / 2.2)
    imgData = ImgFloat2Int(imgData)
    if len(imgData.shape) == 3:
        imgData = imgData[:, :, ::-1]
    cv2.imwrite(fileName, imgData)


def ReadPfm(fileName):
    # http://www.pauldebevec.com/Research/HDR/PFM/

    with open(fileName, 'rb') as file:
        channel_type = file.readline().strip()
        [xres, yres] = list(map(int, file.readline().strip().split()))
        byte_order = float(file.readline().strip())

        rawdata = np.fromfile(file, '>f' if byte_order >
                                            0 else '<f').astype(np.float64)

    return rawdata.reshape((yres, xres, -1))


def WritePfm(fileName, imgData):
    # http://www.pauldebevec.com/Research/HDR/PFM/
    imgData = imgData.astype(np.float32)
    with open(fileName, 'wb') as file:
        file.write(b'PF\n' if imgData.shape[2] == 3 else b'Pf\n')
        file.write(b'%d %d\n' % (imgData.shape[1], imgData.shape[0]))
        byte_order = imgData.dtype.byteorder
        file.write(b'1.0\n' if byte_order == '>' or byte_order ==
                               '=' and sys.byteorder == 'big' else b'-1.0\n')

        imgData.tofile(file)


def Rgb2Lum(img):
    return 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]


def Rgb2Pca(img):
    shape = img.shape
    mean = img.mean(axis=(0, 1))
    img_centered_flattened = (img - mean).reshape(-1, 3)
    N = img_centered_flattened.shape[0]
    C = np.matmul(np.transpose(img_centered_flattened), img_centered_flattened) / (N - 1)
    U, S, V = np.linalg.svd(C)
    return np.matmul(img_centered_flattened, U).reshape(shape)


def Clamp(img):
    return np.clip(img, 0.0, 1.0)


def PackNormalTangentMap(map):
    return np.clip((map + 1.0) / 2.0, 0.0, 1.0)


def UnpackNormalTangentMap(map):
    return np.clip(map * 2.0 - 1.0, -1.0, 1.0)


def MainAnisoAxis(x, theta, k=2, nMin=20):
    n = theta.shape[0]
    assert n >= nMin, 'Need at least {} samples to compute main anisotropic axis.'.format(nMin)
    # Trapezoidal rule
    x01 = (x[0] + x[-1]) / 2
    dtheta = np.concatenate(([theta[0]], theta[1:n] - theta[0:n - 1], [2 * np.pi - theta[-1]]))
    x = x * np.exp(-k * 1j * theta)
    xmid = np.concatenate(([x01 + x[0]], x[1:n] + x[0:n - 1], [x[-1] + x01])) * 0.5
    c2 = (xmid * dtheta).sum()
    angle = -np.angle(c2) * 0.5
    if angle < 0:
        return angle + np.pi
    else:
        return angle


# Compute orientation for each pixel, in range [0, pi),
def OriField(img, patchSize=7, eps=1e-7, coherenceThreshold=0.4):
    t0 = time()
    xPatchSize = patchSize
    yPatchSize = patchSize
    nx = img.shape[0]
    ny = img.shape[1]

    nxPatch = nx // xPatchSize if nx % xPatchSize == 0 else nx // xPatchSize + 1
    nyPatch = ny // yPatchSize if ny % yPatchSize == 0 else ny // yPatchSize + 1

    nxPad = nxPatch * xPatchSize - nx
    nxPad1 = nxPad // 2
    nxPad2 = nxPad - nxPad1
    nyPad = nyPatch * yPatchSize - ny
    nyPad1 = nyPad // 2
    nyPad2 = nyPad - nyPad1

    # Padding original image with 0 to satisfy the size of patch
    img = np.pad(img, ((nxPad1, nxPad2), (nyPad1, nyPad2), (0, 0)), 'constant')

    lum = np.uint8(Rgb2Lum(img) * 255.0)
    sobelx = cv2.Sobel(np.uint8(lum), cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(np.uint8(lum), cv2.CV_64F, 0, 1, ksize=3)

    patchOri = np.zeros((nxPatch, nyPatch))
    coherence = np.zeros((nxPatch, nyPatch))

    # Return in [0, pi)
    def computeTheta(idx):
        up = down = 0
        for s in range(xPatchSize):
            for t in range(yPatchSize):
                xPos = idx[0] * xPatchSize + s
                yPos = idx[1] * yPatchSize + t
                xGrad = sobelx[xPos][yPos]
                yGrad = sobely[xPos][yPos]
                up += 2 * xGrad * yGrad
                down += xGrad * xGrad - yGrad * yGrad
        return (np.arctan2(up, down) + np.pi) / 2

    for i in range(nxPatch):
        for j in range(nyPatch):
            patchOri[i][j] = computeTheta((i, j))

    def computeKp(idx):
        up = 0
        down = eps
        theta = patchOri[idx[0]][idx[1]]
        for s in range(xPatchSize):
            for t in range(yPatchSize):
                xPos = idx[0] * xPatchSize + s
                yPos = idx[1] * yPatchSize + t
                xGrad = sobelx[xPos][yPos]
                yGrad = sobely[xPos][yPos]
                up += np.abs(xGrad * np.cos(theta) + yGrad * np.sin(theta))
                down += np.sqrt(xGrad * xGrad + yGrad * yGrad)
        return up / down

    for i in range(nxPatch):
        for j in range(nyPatch):
            coherence[i][j] = computeKp((i, j))

    # plt.subplot(2, 3, 1)
    # show = patchOri / np.pi
    # mask = show > 0.5
    # show[mask] = (1 - show[mask]) * 2
    # show[np.logical_not(mask)] = show[np.logical_not(mask)] * 2
    # plt.imshow(show, cmap=plt.get_cmap('hsv'))
    # plt.subplot(2, 3, 2)
    # plt.imshow(coherence, cmap=plt.get_cmap('plasma'))
    # plt.show() # Show main orientation of each patch

    # TODO:Iterative optimization
    '''
    invalidPatch = coherence < coherenceThreshold

    def validNeighbors(idx):
        ret = []
        xmin = max(0, idx[0] - 1)
        xmax = min(nxPatch, idx[0] + 2)
        ymin = max(0, idx[1] - 1)
        ymax = min(nyPatch, idx[1] + 2)
        for x in range(xmin, xmax):
            for y in range(ymin, ymax):
                if not invalidPatch[x][y]:
                    ret += [[x, y]]
        return ret

    nInvalid = np.sum(invalidPatch)
    while nInvalid > 0:
        print(nInvalid, coherenceThreshold)
        xs, ys = np.where(invalidPatch)
        idxes = np.stack((xs, ys), axis=-1)
        np.random.shuffle(idxes)
        for idx in idxes:
            neighs = validNeighbors(idx)
            if len(neighs) >= 4:
                up = down = 0
                for n in neighs:
                    theta = patchOri[n[0]][n[1]]
                    up += np.sin(2 * theta)
                    down += np.cos(2 * theta)
                newTheta = np.arctan2(up, down) / 2
                patchOri[idx[0]][idx[1]] = newTheta if newTheta >= 0 else newTheta + np.pi
                coherence[idx[0]][idx[1]] = computeKp(idx)
                invalidPatch[idx[0]][idx[1]] = coherence[idx[0]][idx[1]] < coherenceThreshold

        nInvalidNew = np.sum(invalidPatch)
        if nInvalid - nInvalidNew <= 5:
            coherenceThreshold *= 0.95
        nInvalid = nInvalidNew

        if coherenceThreshold < 0.1:
            break

    plt.subplot(2, 3, 4)
    show = patchOri / np.pi
    #mask = show > 0.5
    #show[mask] = (1 - show[mask]) * 2
    #show[np.logical_not(mask)] = show[np.logical_not(mask)] * 2
    plt.imshow(show, cmap=plt.get_cmap('hsv'))
    plt.subplot(2, 3, 5)
    plt.imshow(coherence, cmap=plt.get_cmap('plasma'))
    plt.show()
    '''

    # TODO:Inter-patch smoothing

    # Bilinear interpolation
    # A pixel is a little square with unit side length. One located at (i, j) has an xy-coordinate of (i+0.5, j+0.5).

    # Complex representation of orientation: z = e^(j*2theta)
    patchOriComp = np.cos(patchOri * 2) + 1j * np.sin(patchOri * 2)
    imgOri = np.zeros((nx, ny))
    for i in range(nx):
        xPatch = (i + 0.5 + nxPad1) / xPatchSize - 0.5
        for j in range(ny):
            yPatch = (j + 0.5 + nyPad1) / yPatchSize - 0.5

            xPatch1 = max(math.floor(xPatch), 0)
            xPatch2 = min(math.ceil(xPatch), nxPatch - 1)

            # On the border where xPatch1==xPatch2, each weighted by 0.5 to sum up to a whole.
            dx1 = 0.5 if xPatch < xPatch1 else xPatch - xPatch1
            dx2 = 1 - dx1

            yPatch1 = max(math.floor(yPatch), 0)
            yPatch2 = min(math.ceil(yPatch), nyPatch - 1)

            dy1 = 0.5 if yPatch < yPatch1 else yPatch - yPatch1
            dy2 = 1 - dy1

            imgOri[i][j] = cmath.phase(
                dx2 * dy2 * patchOriComp[xPatch1][yPatch1] + dx2 * dy1 * patchOriComp[xPatch1][yPatch2] \
                + dx1 * dy2 * patchOriComp[xPatch2][yPatch1] + dx1 * dy1 * patchOriComp[xPatch2][yPatch2])

            if imgOri[i][j] < 0:
                imgOri[i][j] += 2 * np.pi
            imgOri[i][j] /= 2

    '''
    # plt.subplot(2, 3, 6)
    show = imgOri / np.pi
    mask = show > 0.5
    show[mask] = (1 - show[mask]) * 2
    show[np.logical_not(mask)] = show[np.logical_not(mask)] * 2
    plt.imshow(show, cmap=plt.get_cmap('hsv'))
    plt.show() # Show interpolated orientation for each pixel
    '''

    print("Orientation field computation done in {:.3f}s.".format(time() - t0))

    return imgOri


# BRIEF G-2 style
# TODO:改成参数可以输入数组
def Brief(img, patchSize=(), nbytes=(), sigma=(), seed=123456):
    # https://blog.csdn.net/hujingshuang/article/details/46910259
    # patchSize should be an odd number
    assert len(patchSize) == len(nbytes) and len(nbytes) == len(sigma)
    img_pca1 = Rgb2Pca(img)[:,:,0]
    size_y, size_x = img.shape[0:2]
    rng = check_random_state(seed)
    features = np.zeros((size_y, size_x, sum(nbytes)), dtype=np.uint8)
    tilesize_y = 500
    tilesize_x = 500

    nbytes_beg = 0

    for i in range(len(patchSize)):

        size_border = patchSize[i] // 2

        if sigma[i] == 0:
            img_bordered = cv2.copyMakeBorder(img_pca1, size_border, size_border, size_border, size_border, cv2.BORDER_REPLICATE)
        else:
            img_bordered = cv2.copyMakeBorder(cv2.GaussianBlur(img_pca1, (0, 0), sigma[i]), size_border, size_border, size_border, size_border, cv2.BORDER_REPLICATE)

        nbits = nbytes[i] * 8

        # Generate type-2 point pairs
        dy1, dx1 = np.round(np.clip(rng.normal(scale=patchSize[i] / 5, size=(2, nbits)), -size_border, size_border)).astype(
            np.int)
        dy2, dx2 = np.round(np.clip(rng.normal(scale=patchSize[i] / 5, size=(2, nbits)), -size_border, size_border)).astype(
            np.int)

        # Show the feature detector template
        # for i in range(nbits):
        #     plt.plot([dx1[i], dx2[i]], [dy1[i], dy2[i]])
        # plt.show()

        dy1 = dy1[np.newaxis, np.newaxis, :]
        dx1 = dx1[np.newaxis, np.newaxis, :]
        dy2 = dy2[np.newaxis, np.newaxis, :]
        dx2 = dx2[np.newaxis, np.newaxis, :]


        for iy_beg in range(0, size_y, tilesize_y):
            for ix_beg in range(0, size_x, tilesize_x):
                iy_end = iy_beg + tilesize_y
                ix_end = ix_beg + tilesize_x
                iy_end = min(iy_end, size_y)
                ix_end = min(ix_end, size_x)

                cur_tilesize_y = iy_end - iy_beg
                cur_tilesize_x = ix_end - ix_beg

                iy, ix = np.meshgrid(np.arange(iy_beg, iy_end), np.arange(ix_beg, ix_end), indexing='ij')

                y1 = iy[..., np.newaxis] + dy1 + size_border  # shape=(cur_tilesize_y, cur_tilesize_x, nbits)
                x1 = ix[..., np.newaxis] + dx1 + size_border
                y2 = iy[..., np.newaxis] + dy2 + size_border
                x2 = ix[..., np.newaxis] + dx2 + size_border
                # Note that features of 3 component for each pixel are adjacent
                features[iy_beg:iy_end, ix_beg:ix_end, nbytes_beg: nbytes_beg+nbytes[i]] = np.packbits(
                    img_bordered[(y1, x1)] > img_bordered[(y2, x2)], axis=-1)

                # print(ix_beg, ix_end, iy_beg, iy_end, i)

        nbytes_beg += nbytes[i]
    return features
