import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform
import pandas as pd
import os
from PIL import Image
from numpy import unravel_index
from math import exp
from random import random

from pdb import set_trace

def draw_hmap(x_gt, y_gt, shape=224, map_type=0):
    mask = np.zeros((shape, shape))
    if x_gt > shape or y_gt > shape:
        print('invalid coordinates labels ---- {}, {}'.format(x_gt, y_gt))

    if x_gt > -1 and y_gt > -1:
        if map_type == 0:
            mask[int(y_gt), int(x_gt)] = 255

        elif map_type == 1:
            # size x size
            size = 17
            for i in range(-int(size/2), int(size/2)+1):
                for j in range(-int(size/2), int(size/2)+1):
                    if abs(i) < 1 and abs(j) < 1:
                        mask[int(y_gt)+i, int(x_gt)+j] = 255
                    else:
                        try:
                            mask[int(y_gt)+i, int(x_gt)+j] = 255 - (i**2 + j**2)
                        except:
                            print(x_gt)
                            print(y_gt)

        else:
            mask = gaussion_hmap(x_gt, y_gt, shape=shape)

    return mask


def gaussion_hmap(x, y, shape=224):
    # Probability as a function of distance from the center derived
    # from a gaussian distribution with mean = 0 and stdv = 1
    scaledGaussian = lambda x : exp(-(1/2)*(x**2))
    
    isotropicMask = np.zeros((shape,shape))
    # scalor = random()*3+1
    scalor = 0.6
    boundary = 2
    x = int(x)
    y = int(y)
    
    for j in range(max(0, int(x-boundary)), min(int(x+boundary+1), int(shape))):
        for i in range(max(0, int(y-boundary)), min(int(y+boundary+1), int(shape))):
            # find euclidian distance from center of image (shape/2,shape/2)
            # and scale it to range of 0 to 2.5 as scaled Gaussian
            # returns highest probability for x=0 and approximately
            # zero probability for x > 2.5
            distanceFromLabel = np.linalg.norm(np.array([i-y,j-x]))
            distanceFromLabel = scalor * distanceFromLabel
            scaledGaussianProb = scaledGaussian(distanceFromLabel)
            isotropicMask[i,j] = np.clip(scaledGaussianProb*255,0,255)
            


    return np.round(isotropicMask)


def Gaussian(sigma=7):
    sigma_inp = 20#ref.hmGaussInp
    n = sigma_inp * 6 + 1
    g_inp = np.zeros((n, n))
    for i in range(n):
                    for j in range(n):
                                    g_inp[i, j] = np.exp(-((i - n / 2) ** 2 + (j - n / 2) ** 2) / (2. * sigma_inp * sigma_inp))
    if sigma == 7:
            return np.array([0.0529,  0.1197,  0.1954,  0.2301,  0.1954,  0.1197,  0.0529,
                                             0.1197,  0.2709,  0.4421,  0.5205,  0.4421,  0.2709,  0.1197,
                                             0.1954,  0.4421,  0.7214,  0.8494,  0.7214,  0.4421,  0.1954,
                                             0.2301,  0.5205,  0.8494,  1.0000,  0.8494,  0.5205,  0.2301,
                                             0.1954,  0.4421,  0.7214,  0.8494,  0.7214,  0.4421,  0.1954,
                                             0.1197,  0.2709,  0.4421,  0.5205,  0.4421,  0.2709,  0.1197,
                                             0.0529,  0.1197,  0.1954,  0.2301,  0.1954,  0.1197,  0.0529]).reshape(7, 7)
    elif sigma == n:
            return g_inp
    else:
            raise Exception('Gaussian {} Not Implement'.format(sigma))




def heatmap_to_coor(nparray):
    # max = np.argmax(nparray)
    # y = max // nparray.shape[1]
    # x = max % nparray.shape[1]
    y, x = unravel_index(nparray.argmax(), nparray.shape)
    return x, y


class CoorToHeatmap(object):
    """Convert coordinates to heatmap
    Args:

        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """
    def __init__(self, input_size=(1024, 1280), output_size=224):
        assert isinstance(output_size, (int, tuple))
        self.input_size = input_size
        self.output_size = output_size

    def __call__(self, coor):
        h, w = self.input_size

        coor = coor * [self.output_size / w, self.output_size / h]
        coor = np.round(coor)
        
        hmap = draw_hmap(coor[0], coor[1], shape=self.output_size, map_type=0)

        return hmap 

def generate_heatmap2(w, h, x_gt, y_gt):
    x_range = np.arange(start=0, stop=w, dtype=int)
    y_range = np.arange(start=0, stop=h, dtype=int)
    xx, yy = np.meshgrid(x_range, y_range)
    d2 = (xx - int(x_gt))**2 + (yy - int(y_gt))**2
    sigma = 2
    exponent = d2 / 2.0 / sigma / sigma
    heatmap = np.exp(-exponent)
    return heatmap

def accuracy_sum(outputs, labels):
    coor_outputs = []
    coor_labels = []
    list_acc_x = []
    list_acc_y = []
    for out in outputs:
        x, y = heatmap_to_coor(out.squeeze())
        coor_outputs.append((x / 224 * 1280, y / 224 * 1024))

    sum_acc_x = 0
    sum_acc_y = 0
    for idx, output in enumerate(coor_outputs):
        acc_x = (1 - abs(output[0] - labels[idx][0]) / 1280)
        acc_y = (1 - abs(output[1] - labels[idx][1]) / 1024)
        sum_acc_x += acc_x
        sum_acc_y += acc_y
        list_acc_x.append(acc_x)
        list_acc_y.append(acc_y)

    return sum_acc_x, sum_acc_y, list_acc_x, list_acc_y

def spike(hmap):
    x, y = hmap.squeeze().max(1)[0].max(0)[1].item(), hmap.squeeze().max(0)[0].max(0)[1].item()
    new_hmap = torch.zeros(hmap.shape)
    new_hmap[0, x, y] = 1
    return new_hmap

def crop(image, w_center, h_center, coor, scale, size=224):
    set_trace()
    h_image, w_image = np.array(image).shape
    w_center = int(w_center / size * w_image)
    h_center = int(h_center / size * h_image)
    if w_center - (scale/2) < 0:
        w_left = 0
        w_right = scale
        left_margin = 0
    elif w_center + (scale/2) > w_image:
        w_left = w_image - scale
        w_right = w_image
        left_margin = w_left
    else:
        w_left = w_center - int(scale/2)
        w_right = w_center + int(scale/2)
        left_margin = w_left

    if h_center - (scale/2) < 0:
        h_top = 0
        h_bottom = scale
        top_margin = 0
    elif h_center + (scale/2) > h_image:
        h_top = h_image -scale
        h_bottom = h_image
        top_margin = h_top
    else:
        h_top = h_center - int(scale/2)
        h_bottom = h_center + int(scale/2)
        top_margin = h_top
    hmap = torch.zeros(size, size)
    if h_top <= coor[0] < h_bottom and \
        w_left <= coor[1] < w_right:
            hmap[int((coor[0] - top_margin)  / (scale/size)), int((coor[1] - left_margin) / (scale/size))] = 1
    
    return image[h_top:h_bottom, w_left:w_right], hmap

def get_crop_hmap(image, coor, x_delta=0, y_delta=0, size=224):
    h_image, w_image = np.array(image).shape
    half = size // 2 
    x, y = coor
    x1 = x - half
    x2 = x + half
    y1 = y - half
    y2 = y + half
    x_peak = half - x_delta
    y_peak = half - y_delta

    if x1 < 0:
        x_peak = x_peak + x1
        x1 = 0
    elif x2 > w_image:
        x1 = x1 - (x2 - w_image)
        x_peak = x_peak + (x2 - w_image)
    

    if y1 < 0:
        y_peak = y_peak + y1
        y1 = 0
    elif y2 > h_image:
        y1 = y1 - (y2 - h_image)
        y_peak = y_peak + (y2 - h_image)
    
    return image[y1:y1+size, x1:x1+size], draw_hmap(x_peak, y_peak), x1, y1 

# Helper function to show a batch
def show_coor_batch(sample_batched):
    """Show image with landmarks for a batch of samples."""
    images_batch, coor_batch = \
        sample_batched['image'], sample_batched['coor_bc']

    grid_image = utils.make_grid(images_batch)
    grid_coor = utils.make_grid(coor_batch)
    grid = np.concatenate((grid_image.numpy(), grid_coor.numpy()), axis=1)
    plt.imshow(grid.transpose((1, 2, 0)))

# cutout from image
def cutout_image(image_np, coors, cut_size=40, random=True, scalar=4):
    half_size = cut_size // 2
    x, y = coors
    height = image_np.shape[0]
    width = image_np.shape[1]
    # x, y = np.clip(coors, half_size, height-half_size)

    if random:
        for i in range(max(0, y-half_size), min(height, y+half_size)):
            for j in range(max(0, x-half_size), min(width, x+half_size*scalar)):
                image_np[i, j] = np.random.randint(0, 255)
    else:
        image_np[max(0, y-half_size):min(height, y+half_size),
                 max(0, x-half_size):min(width, x+half_size*scalar)] = 0

# crop from image
def crop_image(image_np, coors, crop_size=224):
    image_pil = Image.fromarray(image_np)
    top_left = [max(0, x-crop_size//2) for x in coors[::-1]]
    lower_right = [x + crop_size for x in top_left]
    image_crop = image_pil.crop((*top_left, *lower_right))
    # image_crop, _ = crop(image_pil, *top_left, crop_size, crop_size)

    return image_crop

def zoom(image, x, y):
    image_zoom, _, x_shift, y_shift = get_crop_hmap_by_gt(image, [x, y])

    return image_zoom, x_shift, y_shift


if __name__ == "__main__":
    hmap = CoorToHeatmap()(np.array([1,2]).astype(int))
    print(hmap[:5, :10])
    set_trace()
    
