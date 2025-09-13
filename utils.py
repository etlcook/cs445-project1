import time
import os
import cv2
import argparse
import numpy as np
from scipy import signal
from math import ceil, floor
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def align_images(input_img_1, input_img_2, pts_img_1, pts_img_2,
                 save_images=False):
    
    # Load images
    im1 = cv2.imread(input_img_1)
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)

    im2 = cv2.imread(input_img_2)
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)

    # get image sizes
    h1, w1, b1 = im1.shape
    h2, w2, b2 = im2.shape

    # Get center coordinate of the line segment
    center_im1 = np.mean(pts_img_1, axis=0)
    center_im2 = np.mean(pts_img_2, axis=0)

    plt.close('all')

    # translate first so that center of ref points is center of image
    tx = int(np.around((w1 / 2 - center_im1[0]) * 2))

    if tx > 0:
        padding = np.zeros((im1.shape[0], tx, 3), dtype=im1.dtype)
        im1 = np.concatenate([padding, im1], axis=1)
    else:
        padding = np.zeros((im1.shape[0], -tx, 3), dtype=im1.dtype)
        im1 = np.concatenate([im1, padding], axis=1)

    ty = int(np.round((h1 / 2 - center_im1[1]) * 2))

    if ty > 0:
        padding = np.zeros((ty, im1.shape[1], 3), dtype=im1.dtype)
        im1 = np.concatenate([padding, im1], axis=0)
    else:
        padding = np.zeros((-ty, im1.shape[1], 3), dtype=im1.dtype)
        im1 = np.concatenate([im1, padding], axis=0)

    tx = int(np.around((w2 / 2 - center_im2[0]) * 2))

    if tx > 0:
        padding = np.zeros((im2.shape[0], tx, 3), dtype=im2.dtype)
        im2 = np.concatenate([padding, im2], axis=1)
    else:
        padding = np.zeros((im2.shape[0], -tx, 3), dtype=im2.dtype)
        im2 = np.concatenate([im2, padding], axis=1)

    ty = int(np.round((h2 / 2 - center_im2[1]) * 2))

    if ty > 0:
        padding = np.zeros((ty, im2.shape[1], 3), dtype=im2.dtype)
        im2 = np.concatenate([padding, im2], axis=0)
    else:
        padding = np.zeros((-ty, im2.shape[1], 3), dtype=im2.dtype)
        im2 = np.concatenate([im2, padding], axis=0)

    # downscale larger image so that lengths between ref points are the same
    len1 = np.linalg.norm(pts_img_1[0]-pts_img_1[1])
    len2 = np.linalg.norm(pts_img_2[0]-pts_img_2[1])
    dscale = len2 / len1

    if dscale < 1:
        width = int(im1.shape[1] * dscale)
        height = int(im1.shape[0] * dscale)
        dim = (width, height)
        im1 = cv2.resize(im1, dim, interpolation=cv2.INTER_LINEAR)
    else:
        width = int(im2.shape[1] * 1 / dscale)
        height = int(im2.shape[0] * 1 / dscale)
        dim = (width, height)
        im2 = cv2.resize(im2, dim, interpolation=cv2.INTER_LINEAR)

    # rotate im1 so that angle between points is the same
    theta1 = np.arctan2(-(pts_img_1[1, 1]-pts_img_1[0, 1]),
                        pts_img_1[1, 0]-pts_img_1[0, 0])
    theta2 = np.arctan2(-(pts_img_2[1, 1]-pts_img_2[0, 1]),
                        pts_img_2[1, 0]-pts_img_2[0, 0])
    dtheta = theta2-theta1
    rows, cols = im1.shape[:2]
    M = cv2.getRotationMatrix2D((cols/2, rows/2), dtheta*180/np.pi, 1)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((rows * sin) + (cols * cos))
    nH = int((rows * cos) + (cols * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cols/2
    M[1, 2] += (nH / 2) - rows/2

    im1 = cv2.warpAffine(im1, M, (nW, nH))

    # Crop images (on both sides of border) to be same height and width
    h1, w1, b1 = im1.shape
    h2, w2, b2 = im2.shape

    minw = min(w1, w2)
    brd = (max(w1, w2)-minw)/2
    if minw == w1:  # crop w2
        im2 = im2[:, int(ceil(brd)):int(-floor(brd)) if floor(brd) > 0 else None, :]
        tx = tx-int(ceil(brd))
    else:
        im1 = im1[:, int(ceil(brd)):int(-floor(brd)) if floor(brd) > 0 else None, :]
        tx = tx+int(ceil(brd))

    minh = min(h1, h2)
    brd = (max(h1, h2)-minh)/2
    if minh == h1:  # crop h2
        im2 = im2[int(ceil(brd)):int(-floor(brd)) if floor(brd) > 0 else None, :, :]
        ty = ty-int(ceil(brd))
    else:
        im1 = im1[int(ceil(brd)):int(-floor(brd)) if floor(brd) > 0 else None, :, :]
        ty = ty+int(ceil(brd))

    im1 = cv2.cvtColor(im1.astype(np.uint8), cv2.COLOR_RGB2BGR)
    im2 = cv2.cvtColor(im2.astype(np.uint8), cv2.COLOR_RGB2BGR)

    if save_images:
        output_img_1 = 'aligned_{}'.format(os.path.basename(input_img_1))
        output_img_2 = 'aligned_{}'.format(os.path.basename(input_img_2))
        cv2.imwrite(output_img_1, im1)
        cv2.imwrite(output_img_2, im2)

    return im1, im2


def prompt_eye_selection(image):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(image, cmap='gray')
    ax.set_title('Click on two points for alignment')
    ax.axis('off')
    
    points = []
    
    def onclick(event):
        if event.inaxes != ax:
            return
        if len(points) >= 2:
            return
            
        x, y = event.xdata, event.ydata
        points.append([x, y])
        
        # Plot the point
        ax.plot(x, y, 'r+', markersize=10)
        
        # If we have 2 points, draw a line between them
        if len(points) == 2:
            xs = [points[0][0], points[1][0]]
            ys = [points[0][1], points[1][1]]
            ax.plot(xs, ys, 'r-', linewidth=2)
            plt.draw()
            
        fig.canvas.draw()
        
        if len(points) >= 2:
            plt.close(fig)
    
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    
    return np.array(points, dtype=np.float32)


def crop_image(image, points):
    points = points.astype(int)
    ys = points[:,1]
    xs = points[:,0]
    if len(image.shape)==2:
        image = image[int(ys[0]):int(ys[1]), int(xs[0]):int(xs[1])]
    else:
        image = image[int(ys[0]):int(ys[1]), int(xs[0]):int(xs[1]),:]
    return image


def interactive_crop(image):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(image, cmap='gray')
    ax.set_title('Click upper-left and lower-right corner to crop')
    ax.axis('off')
    
    points = []
    return_object = {
        'cropped_image': None,
        'crop_bound': None
    }
    
    def onclick(event):
        if event.inaxes != ax:
            return
        if len(points) >= 2:
            return
            
        x, y = event.xdata, event.ydata
        points.append([x, y])
        
        ax.plot(x, y, 'r+', markersize=10)
        
        if len(points) == 2:
            # Draw rectangle
            rect_points = np.array(points)
            return_object['crop_bound'] = rect_points
            
            cropped = crop_image(image, rect_points)
            return_object['cropped_image'] = cropped
            
            ax.clear()
            ax.imshow(cropped, cmap='gray')
            ax.set_title('Cropped Image')
            ax.axis('off')
            
        fig.canvas.draw()
        
        if len(points) >= 2:
            plt.close(fig)
    
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    
    return return_object


def gaussian_kernel(sigma, kernel_half_size):
    '''
    Inputs:
        sigma = standard deviation for the gaussian kernel
        kernel_half_size = recommended to be at least 3*sigma
    
    Output:
        Returns a 2D Gaussian kernel matrix
    '''
    window_size = kernel_half_size*2+1
    gaussian_kernel_1d = signal.gaussian(window_size, std=sigma).reshape(window_size, 1)
    gaussian_kernel_2d = np.outer(gaussian_kernel_1d, gaussian_kernel_1d)
    gaussian_kernel_2d /= np.sum(gaussian_kernel_2d) # make sure it sums to one

    return gaussian_kernel_2d


def plot(array, filename=None):
    # plots gray images
    plt.imshow(array, cmap='gray') 
    plt.axis('off')
    if filename:
        array=np.clip(array,0,1)
        array=(array*255).astype(np.uint8)
        cv2.imwrite(filename, array)
        
        
def plot_spectrum(magnitude_spectrum):
    # A logarithmic colormap
    plt.figure()
    plt.imshow(magnitude_spectrum, norm=LogNorm(vmin=1/5))
    plt.colorbar()