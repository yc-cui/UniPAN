import math
import torch
import torch.nn as nn
import torch.nn.functional as func
from math import ceil
from math import floor
from torchvision.transforms import functional as func
from torchvision.transforms import InterpolationMode
from torch import nn, Tensor
from typing import Tuple
import math
import numpy as np
import scipy.ndimage.filters as ft


cdf23 = np.asarray(
    [0.5, 0.305334091185, 0, -0.072698593239, 0, 0.021809577942, 0, -0.005192756653, 0, 0.000807762146, 0,
     -0.000060081482])
cdf23 = [element * 2 for element in cdf23]


def interp23tap_torch(img, ratio, device):
    """
        A PyTorch implementation of the Polynomial interpolator Function.


        Parameters
        ----------
        img : Numpy Array
            Image to be scaled. The conversion in Torch Tensor is made within the function. Dimension: H, W, B
        ratio : int
            The desired scale. It must be a factor power of 2.
        device : Torch Device
            The device on which perform the operation.


        Return
        ------
        img : Numpy array
           The interpolated img.

    """

    assert ((2 ** (round(math.log(ratio, 2)))) == ratio), 'Error: Only resize factors power of 2'

    bs, b, r, c = img.shape

    base_coeff = np.expand_dims(np.concatenate([np.flip(cdf23[1:]), cdf23]), axis=-1)
    base_coeff = np.expand_dims(base_coeff, axis=(0, 1))
    base_coeff = np.concatenate([base_coeff] * b, axis=0)

    base_coeff = torch.from_numpy(base_coeff).to(device)

    for z in range(int(ratio / 2)):

        i1_lru = torch.zeros((bs, b, (2 ** (z + 1)) * r, (2 ** (z + 1)) * c), device=device, dtype=base_coeff.dtype)

        if z == 0:
            i1_lru[:, :, 1::2, 1::2] = img
        else:
            i1_lru[:, :, ::2, ::2] = img

        conv = nn.Conv2d(in_channels=b, out_channels=b, padding=(11, 0),
                         kernel_size=base_coeff.shape, groups=b, bias=False, padding_mode='circular')

        conv.weight.data = base_coeff
        conv.weight.requires_grad = False

        t = conv(torch.transpose(i1_lru, 2, 3))
        img = conv(torch.transpose(t, 2, 3))

    return img


def xcorr_torch(img_1, img_2, half_width):
    """
        A PyTorch implementation of Cross-Correlation Field computation.

        Parameters
        ----------
        img_1 : Torch Tensor
            First image on which calculate the cross-correlation. Dimensions: 1, 1, H, W
        img_2 : Torch Tensor
            Second image on which calculate the cross-correlation. Dimensions: 1, 1, H, W
        half_width : int
            The semi-size of the window on which calculate the cross-correlation

        Return
        ------
        L : Torch Tensor
            The cross-correlation map between img_1 and img_2

        """

    w = ceil(half_width)
    ep = 1e-20
    img_1 = img_1.double()
    img_2 = img_2.double()

    img_1 = func.pad(img_1, (w, w, w, w))
    img_2 = func.pad(img_2, (w, w, w, w))

    img_1_cum = torch.cumsum(torch.cumsum(img_1, dim=-1), dim=-2)
    img_2_cum = torch.cumsum(torch.cumsum(img_2, dim=-1), dim=-2)

    img_1_mu = (img_1_cum[:, :, 2 * w:, 2 * w:] - img_1_cum[:, :, :-2 * w, 2 * w:] -
                img_1_cum[:, :, 2 * w:, :-2 * w] + img_1_cum[:, :, :-2 * w, :-2 * w]) / (4 * w**2)
    img_2_mu = (img_2_cum[:, :, 2 * w:, 2 * w:] - img_2_cum[:, :, :-2 * w, 2 * w:] -
                img_2_cum[:, :, 2 * w:, :-2 * w] + img_2_cum[:, :, :-2 * w, :-2 * w]) / (4 * w**2)

    img_1 = img_1[:, :, w:-w, w:-w] - img_1_mu
    img_2 = img_2[:, :, w:-w, w:-w] - img_2_mu

    img_1 = func.pad(img_1, (w, w, w, w))
    img_2 = func.pad(img_2, (w, w, w, w))

    i2_cum = torch.cumsum(torch.cumsum(img_1**2, dim=-1), dim=-2)
    j2_cum = torch.cumsum(torch.cumsum(img_2**2, dim=-1), dim=-2)
    ij_cum = torch.cumsum(torch.cumsum(img_1 * img_2, dim=-1), dim=-2)

    sig2_ij_tot = (ij_cum[:, :, 2 * w:, 2 * w:] - ij_cum[:, :, :-2 * w, 2 * w:] -
                   ij_cum[:, :, 2 * w:, :-2 * w] + ij_cum[:, :, :-2 * w, :-2 * w])
    sig2_ii_tot = (i2_cum[:, :, 2 * w:, 2 * w:] - i2_cum[:, :, :-2 * w, 2 * w:] -
                   i2_cum[:, :, 2 * w:, :-2 * w] + i2_cum[:, :, :-2 * w, :-2 * w])
    sig2_jj_tot = (j2_cum[:, :, 2 * w:, 2 * w:] - j2_cum[:, :, :-2 * w, 2 * w:] -
                   j2_cum[:, :, 2 * w:, :-2 * w] + j2_cum[:, :, :-2 * w, :-2 * w])

    sig2_ii_tot = torch.clip(sig2_ii_tot, ep, sig2_ii_tot.max().item())
    sig2_jj_tot = torch.clip(sig2_jj_tot, ep, sig2_jj_tot.max().item())

    L = sig2_ij_tot / ((sig2_ii_tot * sig2_jj_tot) ** 0.5 + ep)

    return L


def fspecial_gauss(size, sigma):
    """
        Function to mimic the 'fspecial' gaussian MATLAB function

        Parameters
        ----------
        size : Tuple
            The dimensions of the kernel. Dimension: H, W
        sigma : float
            The frequency of the gaussian filter

        Return
        ------
        h : Numpy array
            The Gaussian Filter of sigma frequency and size dimension

        """

    m, n = [(ss - 1.) / 2. for ss in size]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def fir_filter_wind(hd, w):
    """
        Compute fir filter with window method

        Parameters
        ----------
        hd : float
            Desired frequency response (2D)
        w : Numpy Array
            The filter kernel (2D)

        Return
        ------
        h : Numpy array
            The fir Filter

    """

    hd = np.rot90(np.fft.fftshift(np.rot90(hd, 2)), 2)
    h = np.fft.fftshift(np.fft.ifft2(hd))
    h = np.rot90(h, 2)
    h = h * w

    return h


def nyquist_filter_generator(nyquist_freq, ratio, kernel_size):
    """
        Compute the estimeted MTF filter kernels.

        Parameters
        ----------
        nyquist_freq : Numpy Array or List
            The MTF frequencies
        ratio : int
            The resolution scale which elapses between MS and PAN.
        kernel_size : int
            The size of the kernel (Only squared kernels have been implemented).

        Return
        ------
        kernel : Numpy array
            The filter based on Modulation Transfer Function.

    """

    assert isinstance(nyquist_freq, (np.ndarray, list)), 'Error: GNyq must be a list or a ndarray'

    if isinstance(nyquist_freq, list):
        nyquist_freq = np.asarray(nyquist_freq)

    nbands = nyquist_freq.shape[0]

    kernel = np.zeros((kernel_size, kernel_size, nbands))  # generic kerenel (for normalization purpose)

    fcut = 1 / np.double(ratio)

    for j in range(nbands):
        alpha = np.sqrt(((kernel_size - 1) * (fcut / 2)) ** 2 / (-2 * np.log(nyquist_freq[j])))
        h = fspecial_gauss((kernel_size, kernel_size), alpha)
        hd = h / np.max(h)
        h = np.kaiser(kernel_size, 0.5)
        h = np.real(fir_filter_wind(hd, h))
        h = np.clip(h, a_min=0, a_max=np.max(h))
        h = h / np.sum(h)

        kernel[:, :, j] = h

    return kernel


def gen_mtf(ratio, sensor, kernel_size=41):
    """
        Compute the estimated MTF filter kernels for the supported satellites.

        Parameters
        ----------
        ratio : int
            The resolution scale which elapses between MS and PAN.
        sensor : str
            The name of the satellites which has provided the images.
        kernel_size : int
            The size of the kernel (Only squared kernels have been implemented).

        Return
        ------
        kernel : Numpy array
            The filter based on Modulation Transfer Function for the desired satellite.

        """
    nyquist_gains = []
    sensor = sensor.lower()
    if 'quickbird' in sensor:
        nyquist_gains = np.asarray([0.34, 0.32, 0.30, 0.22])  # Bands Order: B,G,R,NIR
    elif 'ikonos' in sensor:
        nyquist_gains = np.asarray([0.26, 0.28, 0.29, 0.28])  # Bands Order: B,G,R,NIR
    elif 'worldview-2' in sensor:
        nyquist_gains = [0.35, 0.35, 0.35, 0.35]
    elif 'worldview-3' in sensor:
        nyquist_gains = [0.355, 0.360, 0.365, 0.335]
    elif 'worldview-4' in sensor:
        nyquist_gains = [0.230, 0.230, 0.230, 0.230]
    elif 'gaofen-1' in sensor:
        # System Characterization Report on the Gaofen-1
        nyquist_gains = [0.250, 0.250, 0.140, 0.040]
    h = nyquist_filter_generator(nyquist_gains, ratio, kernel_size)

    return h


def local_corr_mask(img_in, ratio, sensor, device, kernel=8):
    """
        Compute the threshold mask for the structural loss.

        Parameters
        ----------
        img_in : Torch Tensor
            The test image, already normalized and with the MS part upsampled with ideal interpolator.
        ratio : int
            The resolution scale which elapses between MS and PAN.
        sensor : str
            The name of the satellites which has provided the images.
        device : Torch device
            The device on which perform the operation.
        kernel : int
            The semi-width for local cross-correlation computation.
            (See the cross-correlation function for more details)

        Return
        ------
        mask : PyTorch Tensor
            Local correlation field stack, composed by each MS and PAN. Dimensions: Batch, B, H, W.

        """

    pan = torch.unsqueeze(img_in[:, -1, :, :], dim=1)
    ms = img_in[:, :-1, :, :]

    mtf_kern = gen_mtf(ratio, sensor)[:, :, 0]
    mtf_kern = np.expand_dims(mtf_kern, axis=(0, 1))
    mtf_kern = torch.from_numpy(mtf_kern).type(torch.float32)
    pad = floor((mtf_kern.shape[-1] - 1) / 2)

    padding = nn.ReflectionPad2d(pad)

    depthconv = nn.Conv2d(in_channels=1,
                          out_channels=1,
                          groups=1,
                          kernel_size=mtf_kern.shape,
                          bias=False)

    depthconv.weight.data = mtf_kern
    depthconv.weight.requires_grad = False
    depthconv.to(device)
    pan = pan.to(device)
    ms = ms.to(device)
    pan = padding(pan)
    pan = depthconv(pan)
    mask = xcorr_torch(pan, ms, kernel)
    mask = 1.0 - mask

    return mask.float().to(device)


def d_rho(outputs, labels):
    x_corr = torch.clamp(xcorr_torch(outputs, labels, 8), min=-1.0, max=1.0)
    x = 1.0 - x_corr
    d_rho = torch.mean(x)
    return d_rho
