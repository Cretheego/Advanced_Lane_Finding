# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 19:55:11 2017

@author: Administrator
"""
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import numpy as np
import cv2
import Finding_the_Lines as FtL 
from   numpy import matrix

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Calculate directional gradient
    # Apply threshold
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    elif orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold     
    grad_binary = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1     
    return grad_binary

def mag_thresh(image, sobel_kernel=3, mag_thresh=(0, 255)):
    # Calculate gradient magnitude
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    mag_binary = (gradmag/scale_factor).astype(np.uint8) 
    binary_output = np.zeros_like(gradmag)
    mag_binary[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return mag_binary

def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Calculate gradient direction
    # Apply threshold
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    dir_binary =  np.zeros_like(absgraddir)
    dir_binary[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    return dir_binary

    
def color_thresholding(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    img = np.copy(img)
    # Convert to HSV color space and separate the V channel
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    h_channel = hsv[:,:,0]
    l_channel = hsv[:,:,1]
    s_channel = hsv[:,:,2]
    # Sobel x
    sobelx = cv2.Sobel(s_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    sobely = cv2.Sobel(s_channel, cv2.CV_64F, 0, 1) # Take the derivative in x
    abs_sobely = np.absolute(sobely) # Absolute x derivative to accentuate lines away from horizontal
    #scaled_sobel = np.uint8(255*abs_sobely/np.max(abs_sobely))    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary))
    
    absgraddir = np.arctan2(abs_sobely,abs_sobelx)
    
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (absgraddir == 1)] = 1

    return color_binary,combined_binary
    
# perspevtive transform
def persp_trans(img, src, dst, mtx, dist):
    # Pass in your image into this function
    # Write code to do the following steps
    # 1) Undistort using mtx and dist
    undist = cv2.undistort(img,mtx,dist,None,mtx) 
    # 2) Convert to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img_size = (undist.shape[1], undist.shape[0])
    print(gray.shape)
    if 1:
        # Grab the image shape
       
        # Given src and dst points, calculate the perspective transform matrix
        M = cv2.getPerspectiveTransform(src, dst)
        # Warp the image using OpenCV warpPerspective()
        warped = cv2.warpPerspective(undist, M, img_size)
        print(M)
    return warped,M

def draw_lines(img, lines, color=[255, 0, 0], thickness=3):
    cv2.line(img, (x1, y1), (x2, y2), color, thickness) 
    
def main():
    # Read in an image
    image = mpimg.imread('./calibration_wide/test_undist.jpg')
    # Choose a Sobel kernel size
    ksize = 3 # Choose a larger odd number to smooth gradient measurements
    nx = 9 # the number of inside corners in x
    ny = 6 # the number of inside corners in y
    dist_pickle = pickle.load( open( "./calibration_wide/wide_dist_pickle.p", "rb" ) )
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]

    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(20, 255))
    grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(20, 255))
    mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(30, 255))
    dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(0.9, 0.5))
    
    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    
    color_binary,combined = color_thresholding(image, s_thresh=(170, 255), sx_thresh=(20, 100))
    
    image = cv2.imread('./calibration_wide/test_undist.jpg')
    img_size = (image.shape[1], image.shape[0])
    # For source points I'm grabbing the outer four detected corners
    src = np.float32([[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],\
        [((img_size[0] / 6) - 10), img_size[1]],\
        [(img_size[0] * 5 / 6) + 60, img_size[1]],\
        [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
    print(src)
    dst = np.float32([[(img_size[0] / 4), 0],\
        [(img_size[0] / 4), img_size[1]],\
        [(img_size[0] * 3 / 4), img_size[1]],\
        [(img_size[0] * 3 / 4), 0]])
    
    top_down, perspective_M = persp_trans(image, src, dst, mtx, dist)   

    _,binary_warped = color_thresholding(top_down, s_thresh=(70, 255), sx_thresh=(20, 100))
    #print(binary_warped.shape)
    #binary_warped = abs_sobel_thresh(top_down, orient='x', sobel_kernel=ksize, thresh=(20, 255))
    
    ploty,left_fit,left_fitx,right_fit,right_fitx = FtL.find_lines(binary_warped,flag=0)
    #print("====")
    #print(ploty.shape)
    print('*-*-*-',np.hstack((left_fit,right_fit)).shape)
    
    ym_per_pix = 30/720
    xm_per_pix = 3.7/700
    left_curverad,right_curverad = MC.meas_cur(ploty,left_fit,left_fitx,right_fit,\
                                               right_fitx,ym_per_pix,xm_per_pix)
    print(left_curverad, 'm', right_curverad, 'm')
    #offset = ((rightx[0] - leftx[0]) / 2 + leftx[0] - image.shape[0] / 2) * xm_per_pix
    offset = (image.shape[0] / 2 - leftx[0]) * xm_per_pix
    print(offset,'m')
    #print(ploty)
    
    color_warp = MC.draw_back(binary_warped,left_fitx,right_fitx, ploty)
    Minv = matrix(perspective_M).I
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)
    #plt.imshow(result)
    
    cv2.line(image, (src[1][0], src[1][1]), (src[0][0], src[0][1]), [255, 0, 0], 3) 
    cv2.line(image, (src[2][0], src[2][1]), (src[3][0], src[3][1]), [255, 0, 0], 3) 
    cv2.line(image, (src[0][0], src[0][1]), (src[3][0], src[3][1]), [255, 0, 0], 3) 
    
    f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(32, 12))
    #f.tight_layout()
    ax1.imshow(image)
    ax1.set_title('Original Image', fontsize=20)
    ax2.imshow(binary_warped, cmap='gray')
    ax2.set_title('Thresholded Magnitude', fontsize=20)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    ax3.imshow(result)



if __name__ == '__main__':
    main()
    