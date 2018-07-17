# -*- coding: utf-8 -*-
__author__ = 'Domen Gorjup'
__version__ = '1.0'

"""
Generate print-ready speckle or line patterns to use in DIC applications.
"""

from itertools import product
from random import choice

import numpy as np
from numpy.random import multivariate_normal
from scipy.misc import imresize
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
from imageio import imwrite
from tqdm import tqdm
import piexif


def speckle(my, mx, D=3, shape=None, s=0, blur=0.6, value=1.):
    """
    Generates a random speckle in an image of given shape.
    """
    
    if not s:
        if D < 3:
            raise Exception('Set higher speckle diameter (D >= 3 px)!')
        polinom = np.array([ -4.44622133e-06,   1.17748897e-02,   2.58275794e-01, -0.65])
        s = np.polyval( polinom, D) # Empirično
    N = int(s * 300)
    
    if my == 0 and mx == 0 and shape is None:
        my = 2*D
        mx = 2*D
    
    mean = np.array([my, mx])
    cov = np.array([[s, 0], [0, s]])
    y, x = multivariate_normal(mean, cov, N).T
    x = np.round(x).astype(int)
    y = np.round(y).astype(int)
    yx = np.column_stack((y, x))
    
    # Velikost vzorca:
    dx = np.max(yx[:, 1]) - np.min(yx[:, 1])
    dy = np.max(yx[:, 0]) - np.min(yx[:, 0])
    d = np.mean(np.array([dx, dy]))
    #print(d)
    
    neustrezni_i = []
    
    if shape is None:
        slika = np.zeros((2*int(my), 2*int(mx)))
    else:
        slika = np.zeros(shape)
    
    for (y_, x_) in yx:
        try:
            slika[y_, x_] = value
        except:
            pass
        
    return gaussian_filter(slika, blur), d


def speckle_image(shape, D, size_randomness=1, position_randomness=1, speckle_blur=0.6, gridstep=2.2, n_unique=100):
    '''
    Generates an image of shape (w, d), populated with speckles of
    random shape and semi-random position and size. Generates and
    picks randomly from `n_unique` unique speckles.
    '''
    h, w = shape
    D = int(D)
    border = int(10*D) # to avoid overlap and clipping near borders 
    
    h += 2*border
    w += 2*border
    
    im = np.ones((int(h), int(w)))
    grid_size = D * gridstep
    
    xs = np.arange(border, w-border//2, grid_size).astype(int)
    ys = np.arange(border, h-border//2, grid_size).astype(int)
    
    speckle_coordinates = list(product(ys, xs))
    N = np.min([len(speckle_coordinates), n_unique])
    
    # Generate unique speckles
    speckles = []
    for i in range(N):
        Dr = np.clip(D + int(np.random.randn(1) * D * size_randomness*0.2), 2, 2*D)
        speckles.append(speckle(0, 0, Dr, blur=speckle_blur)[0])
    print('Random speckle generation complete.')
    
    for y, x in tqdm(speckle_coordinates):
        s = choice(speckles)
        s_shape = np.array(s.shape)
        dy, dx = (s_shape // 2).astype(int)
        
        x += int(np.random.randn(1)*(D*position_randomness*0.2))
        y += int(np.random.randn(1)*(D*position_randomness*0.2))
        
        sl = np.s_[y-dy:y+dy, x-dx:x+dx]
        
        im[sl]  -= s

    im = np.clip(im, 0, 1)

    im = im[border:-border, border:-border]
    
    return im


def save_image(path, image, dpi, exif=None):
    """ Saves a generated pattern image as JPEG, configured for printing """
    if path.split('.')[-1].lower() not in ['jpg', 'jpeg']:
        path += '.jpg'
     
    imwrite(path, np.uint8(image/np.max(image)*255), quality=100, dpi=(dpi, dpi))


def add_dpi_meta(path, dpi=300, comment=''):
    exif_dict = piexif.load(path)
    exif_dict["0th"][piexif.ImageIFD.XPComment] = comment.encode('utf16')
    exif_dict["0th"][piexif.ImageIFD.XResolution] = (dpi, 1)
    exif_dict["0th"][piexif.ImageIFD.YResolution] = (dpi, 1)
    exif_dict["0th"][piexif.ImageIFD.ResolutionUnit] = 2 # 1: no unit, 2: inch, 3: cm
    exif_bytes = piexif.dump(exif_dict)
    piexif.insert(exif_bytes, path)


def generate_and_save(height, width, dpi, speckle_diameter, path, size_randomness=0.5, 
                        position_randomness=0.5, speckle_blur=1, grid_step=1.2):
    """
    Generates a speckle image of given shape, speckle diameter etc. and saves
    it as a JPEG image to specified path, configured for printing.

    Parameters
    ----------
    height: float
        the height of output image in mm
    width: float
        the width of output image in mm
    dpi: float
        DPI setting for printing
    speckle_diameter: float
        average speckle diameter in mm
    path: str, None
        output file save path. If None, the file is named according 
        to speckle settings. Defaults to None.
    size_randomness: float
        a measure of speckle diameter randomness. 
        Should be in [0, 1] range.
    position_randomness: float
        a measure of speckle position deviation from regular grid. 
        Should be in [0, 1] range.
    speckle_blur: float
        sigma parameter of bluring Gaussian filter
    grid_step: float
        spacing of regular grid for speckle positioning, in terms
        of `speckle_diameter`.

    Returns
    -------
    image: (h, w), ndarray
        resulting speckle image (grayscale, [0, 1])
    """
    ppmm = dpi / 25.4
    w = int(np.round((width * ppmm)))
    h = int(np.round((height * ppmm)))
    D = np.ceil(speckle_diameter*ppmm)

    im = speckle_image((h, w), D, size_randomness, position_randomness, speckle_blur, grid_step)

    if path is None:
        path = f'speckle_{width}x{height}mm_D{speckle_diameter}mm_{dpi}DPI.jpg'

    # Add exif comment to image:
    image_comment = f'height: {height} mm\nwidth: {width} mm\ndpi: {dpi}\nD: {speckle_diameter} mm\n'\
                    f'size_randomness: {size_randomness}\nposition_randomness: {position_randomness}\n'\
                    f'speckle_blur: {speckle_blur}\ngrid_step: {grid_step}'
    
    save_image(path, im, dpi)
    add_dpi_meta(path, dpi, comment=image_comment)
    print(f'Image saved to {path}.')
    return im


def generate_lines(height, width, dpi, line_width, path, orientation='vertical', N_lines=None):
    """
    Generates a pattern of lines and saves it as a JPEG image to specified 
    path, configured for printing.

    Parameters
    ----------
    height: float
        the height of output image in mm
    width: float
        the width of output image in mm
    dpi: float
        DPI setting for printing
    line_width: float
        line width in mm
    path: str, None
        output file name.
    orientation: str
        line orientation: 'vertical' (default) or 'horizontal'.
    N_lines: float
        number of lines. If None, `line_width` is used. 
        Defaults to None.

    Returns
    -------
    image: (h, w), ndarray
        resulting image (grayscale, [0, 1])
    """

    ppmm = dpi / 25.4
    w = int(np.round((width * ppmm)))
    h = int(np.round((height * ppmm)))

    if N_lines is not None:
        if orientation == 'vertical':
            line_width = width // (2*N_lines)
        else:
            line_width = height // (2*N_lines)

    D = int(np.round(line_width * ppmm))

    im = np.ones((h, w), dtype=np.uint8) * 255
    if orientation == 'vertical':
        black_id = np.hstack( [np.arange(i*D, i*D+D) for i in range(0, w//D, 2)] )
        im[:, black_id] = 0
    else:
        black_id = np.hstack( [np.arange(i*D, i*D+D) for i in range(0, h//D, 2)] )
        im[black_id] = 0

    image_comment = f'{orientation} lines\nline width: {line_width}\n DPI: {dpi}'
    save_image(path, im, dpi)
    add_dpi_meta(path, dpi, comment=image_comment)
    print(f'Image saved to {path}.')
    return im
    

if __name__ == '__main__':
    slika = speckle_image((70, 70), 30, size_randomness=0.75, speckle_blur=1., gridstep=1.2)
    plt.figure(figsize=(7, 7))
    plt.imshow(slika, cmap='gray', interpolation='nearest', vmin=0, vmax=1)
    plt.show()