import numpy as np
import os
import cv2
from utilities import _delta, WriteImg, ImgFloat2Int, ImgInt2Float, Rgb2Lum
import matplotlib.pyplot as plt

zmin = 0
zmax = 255
n_z_values = zmax - zmin + 1
l_smooth = 50.0
weight = np.copy([z if z <= zmax // 2 else zmax - z for z in range(n_z_values)]) + 1e-3  # Shouldn't be 0


n_color_y = 15
n_color_x = 20

def MakeColorCard(path, tile_size=200):
    size_y = n_color_y * tile_size
    size_x = n_color_x * tile_size

    colorTable = np.random.rand(n_color_y, n_color_x, 3)
    img = np.zeros((size_y, size_x, 3))
    for iy in range(n_color_y):
        for ix in range(n_color_x):
            img[iy * tile_size: (iy + 1) * tile_size, ix * tile_size: (ix + 1) * tile_size] = colorTable[iy, ix]

    # WriteImg(os.path.join(path, 'colorcard.jpg'), img, False)
    return img

def FitResponseCurve(list_img, list_exposure_time, align=False):
    n_exposures = len(list_exposure_time)

    list_exposure_time = np.copy(list_exposure_time)
    log_t = np.log(list_exposure_time)

    n_pixels = n_color_y * n_color_x
    if align:
        list_img_aligned = np.copy(list_img)
        alignMTB = cv2.createAlignMTB()
        alignMTB.process(list_img, list_img_aligned)
        list_img = list_img_aligned
        del alignMTB
        del list_img_aligned

    list_img = [ImgFloat2Int(cv2.resize(ImgInt2Float(img), (n_color_x, n_color_y), interpolation=cv2.INTER_AREA)) for img in list_img]

    # Variables: n_z_values for g(z) | n_pixels values for lnEi
    n_variables = n_z_values + n_pixels
    # Constrain equations: n_exposures * n_pixels combinations for data-fitting | 1 equation to set middle value | n_z_values - 2 penalties for g''(z)
    n_constrains = n_exposures * n_pixels + 1 + n_z_values - 2
    response_curve = np.zeros((n_z_values, 3))

    # On three channels
    for c in range(3):

        Z = [img[..., c].flatten() for img in list_img]


        A = np.zeros((n_constrains, n_variables))
        b = np.zeros(n_constrains)

        k = 0
        # Data-fitting errors
        for i in range(n_pixels):
            for j in range(n_exposures):
                zij = Z[j][i]
                wij = weight[zij]
                A[k, zij] = wij
                A[k, n_z_values + i] = -wij
                b[k] = wij * log_t[j]
                k += 1

        # Fix the curve by setting its middle value to exposure value 0.5^2.2, rather than 1 in the original paper
        A[k, n_z_values//2] = 1
        b[k] = np.log(0.5**(2.2))
        k += 1

        # Smoothness penalties
        for z in range(zmin+1, zmax):
            wz = weight[z]
            A[k, z-1] = l_smooth * wz
            A[k, z] = -2 * l_smooth * wz
            A[k, z+1] = l_smooth * wz
            k += 1

        x = np.linalg.lstsq(A, b)[0]
        response_curve[:, c] = x[:n_z_values]
    #ShowResponseCurve(response_curve)

    return response_curve

def Img2Radiance(list_img, list_exposure_time, response_curve):

    list_log_t = np.log(np.copy(list_exposure_time))

    # KOLB, C., MITCHELL, D., AND HANRAHAN, P. A realistic camera model for computer graphics. In SIGGRAPH ’95 (1995).
    # Modern camera lens are designed to give a constant mapping between scene radiance and film irradiance among different pixels
    # So we directly regard irradiance as radiance, though differentiated by an area scale factor

    n_exposures = np.size(list_img, 0)
    size_y = np.size(list_img, 1)
    size_x = np.size(list_img, 2)

    radiance_map = np.zeros((size_y, size_x, 3))

    for c in range(3):
        g = response_curve[:, c]
        w_c = np.zeros((size_y, size_x))
        for p in range(n_exposures):
            w_cp = weight[list_img[p][..., c]]
            radiance_map[..., c] += (g[list_img[p][..., c]] - list_log_t[p]) * w_cp
            w_c += w_cp
        radiance_map[..., c] /= (w_c + _delta)

    return np.exp(radiance_map)

def FetchWeights(radiance_samples, exposure_time, response_curve):
    n_samples = radiance_samples.shape[0]
    log_exposure_value = np.log(radiance_samples) + np.log(exposure_time)
    pixel_vals = np.zeros((n_samples, 3), dtype=np.uint8)
    for c in range(3):
        pixel_vals[:, c] = np.clip(np.searchsorted(response_curve[:, c], log_exposure_value[:, c]), zmin, zmax).astype(
            np.uint8)
    return weight[pixel_vals] / zmax


def Radiance2Img(radiance, exposure_time, response_curve):
    size_y, size_x = radiance.shape[0:2]
    log_exposure_value = np.log(radiance+_delta) + np.log(exposure_time)
    img = np.zeros((size_y, size_x, 3), dtype=np.uint8)
    for c in range(3):
        img[:, :, c] = np.clip(np.searchsorted(response_curve[:, c], log_exposure_value[:, :, c]), zmin, zmax).astype(np.uint8)
    return img


def ShowResponseCurve(response_curve, log=True):
    if not log:
        response_curve = np.exp(response_curve)
    plt.plot(np.arange(n_z_values), response_curve[:, 0], c='r')
    plt.plot(np.arange(n_z_values), response_curve[:, 1], c='g')
    plt.plot(np.arange(n_z_values), response_curve[:, 2], c='b')
    #plt.plot(np.arange(n_z_values), response_curve.mean(axis=1), c='orange', linewidth=2)
    # Plot a gamma 2.2 curve
    x = np.arange(256)
    if not log:
        plt.plot(x, (x / 255) ** 2.2, c='k')
    else:
        plt.plot(x, np.log((x / 255) ** 2.2), c='k')
    plt.show()
    return

