import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

im = cv2.imread("mk-and-loki.jpg", cv2.IMREAD_GRAYSCALE)

##########

# Get dimensions of image
print(im.shape)

# Crop image
# im = im[1900:3200, 950:2050]

# Resize with given dimensions (NOTE: this takes column, row order)
# im = cv2.resize(im, (500, 600))

# Resize with given scalars fx=columns and fy=rows
# im = cv2.resize(im, (0,0), fx=0.8, fy = 0.8)

# Rotate and image a common amount (can't just put a number)
# im = cv2.rotate(im, cv2.ROTATE_180)

# Hijack pixel color values
# for i in range(200, 400):
#     for j in range(800, 1000):
#         im[i][j] = 0

# Apply low-pass filter


# Apply high-pass filter


##########


###################
### Testing #######

# resize to make smaller
# im = cv2.resize(im, (0,0), fx=0.5, fy=0.5)
im = cv2.resize(im, (0,0), fx=0.4, fy=0.4)
print(im.shape)

# Box kernel
# box_kernel = np.ones((5, 5), np.float32) / 25
# print(box_kernel)
# box_im = cv2.filter2D(im, -1, box_kernel)
# cv2.imshow('box blurred', box_im)

# Gauss kernel 
    # -> Sigma (stddev) determines blurriness, kernel half-width = (sigma * 3)
# gauss_im = cv2.GaussianBlur(im, (17, 17), sigmaX=3, sigmaY=3)
# cv2.imshow('gauss blurred', gauss_im)

# Gauss kernel
# gauss_im_2 = cv2.GaussianBlur(im, (17,17), sigmaX=2, sigmaY=2)
# cv2.imshow('gauss 2 blurred', gauss_im_2)

# More manual way of getting Gaus kernel:
sigma =2
ksize = int(np.ceil(sigma)*6+1)
gauss_kernel = cv2.getGaussianKernel(ksize, sigma) # 1D kernel
gauss_kernel = gauss_kernel*np.transpose(gauss_kernel) # 2D kernel by outer product
manual_gauss_im = cv2.filter2D(im, -1, gauss_kernel)
cv2.imshow("Manual Gauss", manual_gauss_im)
print(gauss_kernel)

# Get FFT magnitude
fftmag = np.abs(np.fft.fftshift(np.fft.fft2(im)))

# Display FFT 
    # -> Needed to apt install libxcb-cursor0 
plt.figure(figsize=(15,15))
plt.imshow(fftmag,norm=LogNorm(fftmag.min(),fftmag.max()),cmap='jet')
plt.show()


###################


###################



cv2.imshow('original', im)
cv2.waitKey(0)
cv2.destroyAllWindows()

