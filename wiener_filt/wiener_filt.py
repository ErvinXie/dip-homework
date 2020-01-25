import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from scipy import signal


def gaussian_kern(size=6, std=1):
    if type(size) == tuple:
        x = size[0]
        y = size[1]
        gx = signal.gaussian(x, std=std).reshape(x, 1)
        gy = signal.gaussian(y, std=std).reshape(y, 1)
        re = np.outer(gx, gy)
    else:
        x = size
        g = signal.gaussian(x, std=std).reshape(x, 1)
        re = np.outer(g, g)
    return re / np.sum(re)


# plt.imshow(gaussian_kern(),interpolation='none')
# plt.show()
def get_idx(r, c):
    cc, rc = np.meshgrid(np.arange(c), np.arange(r))
    return np.vstack((rc.flatten(), cc.flatten()))


def gaussian_blur(path='in.jpg'):
    im = cv.imread(path).astype(np.float64) / 255
    nim = np.zeros_like(im)

    k = gaussian_kern(9, 3)
    kidx = get_idx(k.shape[0], k.shape[1])
    kidx -= (np.array(k.shape) // 2).reshape(-1, 1)

    nidx = get_idx(nim.shape[0], nim.shape[1]).T
    nidx = nidx.reshape(nim.shape[0], nim.shape[1], -1, 1)

    nca = nidx + kidx.reshape(1, 1, kidx.shape[0], kidx.shape[1])
    nca = np.where(nca < 0, 0, nca)
    nca = np.where(np.stack((nca[:, :, 0] < nim.shape[0], nca[:, :, 1] < nim.shape[1]), axis=2),
                   nca,
                   np.array([nim.shape[0] - 1, nim.shape[1] - 1]).reshape(1, 1, 2, 1))
    nim = np.sum(im[nca[:, :, 0], nca[:, :, 1]] * k.flatten().reshape(1, 1, -1, 1), axis=2)

    cv.imshow('test', im)
    cv.waitKey(0)
    cv.imshow('test', nim)
    cv.waitKey(0)


def gaussian_blur_fast(path='in.jpg'):
    im = cv.imread(path).astype(np.float64) / 255
    imf = np.fft.fft2(im, axes=(0, 1))

    k = gaussian_kern((im.shape[0], im.shape[1]), 3)
    kf = np.fft.fft2(k)

    kf = np.expand_dims(kf, 2)

    nim = imf * kf
    nim = np.fft.ifftshift(np.fft.ifft2(nim, axes=(0, 1)), axes=(0, 1)).real

    cv.imshow('im', im)
    cv.waitKey(0)
    cv.imshow('nim', nim)
    cv.waitKey(0)

    pass


gaussian_blur()
gaussian_blur_fast()
