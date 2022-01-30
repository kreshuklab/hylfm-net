from pathlib import Path
from tifffile import imread as tiff_imread
import numpy as np
from scipy.signal import fftconvolve
from scipy import interpolate

def imread(p: Path):
    return tiff_imread(p).squeeze()

def normxcorr2(template, image, mode="full"):
    ########################################################################################
# Author: Ujash Joshi, University of Toronto, 2017                                     #
# Based on Octave implementation by: Benjamin Eltzner, 2014 <b.eltzner@gmx.de>         #
# Octave/Matlab normxcorr2 implementation in python 3.5                                #
# Details:                                                                             #
# Normalized cross-correlation. Similiar results upto 3 significant digits.            #
# https://github.com/Sabrewarrior/normxcorr2-python/master/norxcorr2.py                #
# http://lordsabre.blogspot.ca/2017/09/matlab-normxcorr2-implemented-in-python.html    #
########################################################################################
    """
    Input arrays should be floating point numbers.
    :param template: N-D array, of template or filter you are using for cross-correlation.
    Must be less or equal dimensions to image.
    Length of each dimension must be less than length of image.
    :param image: N-D array
    :param mode: Options, "full", "valid", "same"
    full (Default): The output of fftconvolve is the full discrete linear convolution of the inputs. 
    Output size will be image size + 1/2 template size in each dimension.
    valid: The output consists only of those elements that do not rely on the zero-padding.
    same: The output is the same size as image, centered with respect to the ‘full’ output.
    :return: N-D array of same dimensions as image. Size depends on mode parameter.
    """

    # If this happens, it is probably a mistake
    if np.ndim(template) > np.ndim(image) or \
            len([i for i in range(np.ndim(template)) if template.shape[i] > image.shape[i]]) > 0:
        print("normxcorr2: TEMPLATE larger than IMG. Arguments may be swapped.")

    template = template - np.mean(template)
    image = image - np.mean(image)

    a1 = np.ones(template.shape)
    # Faster to flip up down and left right then use fftconvolve instead of scipy's correlate
    ar = np.flipud(np.fliplr(template))
    out = fftconvolve(image, ar.conj(), mode=mode)
    
    image = fftconvolve(np.square(image), a1, mode=mode) - \
            np.square(fftconvolve(image, a1, mode=mode)) / (np.prod(template.shape))

    # Remove small machine precision errors after subtraction
    image[np.where(image < 0)] = 0

    template = np.sum(np.square(template))
    out = out / np.sqrt(image * template)

    # Remove any divisions by 0 or very close to 0
    out[np.where(np.logical_not(np.isfinite(out)))] = 0
    
    return out

def rectify_image(img, pitch, center_x, center_y, nnum=19):
    nnum_diff = np.floor(nnum/2)
    x_resample = np.concatenate((np.flip(np.arange(center_x, 0, step=-pitch/nnum)), 
                                np.arange(center_x, img.shape[1], step=pitch/nnum)))
    y_resample = np.concatenate((np.flip(np.arange(center_y, 0, step=-pitch/nnum)), 
                                np.arange(center_y, img.shape[0], step=pitch/nnum)))

    x_resample_center_init = np.where(x_resample == center_x)[0][0] - nnum_diff
    x_resample_init = x_resample_center_init - nnum*np.floor(x_resample_center_init/nnum) + nnum
    y_resample_center_init = np.where(y_resample == center_y)[0][0] - nnum_diff
    y_resample_init = y_resample_center_init - nnum*np.floor(y_resample_center_init/nnum) + nnum

    x_resample_Q = x_resample[int(x_resample_init):]
    y_resample_Q = y_resample[int(y_resample_init):]

    f = interpolate.interp2d(range(img.shape[1]), range(img.shape[0]), img, kind='cubic')
    img_resample = f(x_resample_Q, y_resample_Q)
    img_rect = img_resample[:int(nnum*np.floor((img_resample.shape[0] - y_resample_init)/nnum)),
                            :int(nnum*np.floor((img_resample.shape[1] - x_resample_init)/nnum))]
    return img_rect