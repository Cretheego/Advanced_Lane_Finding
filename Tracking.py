# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 00:50:13 2017

@author: Administrator
"""
from moviepy.editor import VideoFileClip
import instrument_function as infu
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import numpy as np
import cv2
from   numpy import matrix

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self,mtx,dist):
        # was the line detected in the last iteration?
        self.detected = True  
        # x values of the last n fits of the line
        self.recent_xfitted_left = [] 
        self.recent_xfitted_right = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx_left =  None 
        self.bestx_right =  None 
        #polynomial coefficients averaged over the last n iterations
        self.best_fit_left = np.array([False,False,False]) 
        self.best_fit_right = np.array([False,False,False])
        #polynomial coefficients for the most recent fit
        self.current_fit_left = []  
        self.current_fit_right = [] 
        self.current_xfit_left = []  
        self.current_xfit_right = [] 
        #radius of curvature of the line in some units
        self.radius_of_curvature_left = [] 
        self.radius_of_curvature_right = [] 
        #distance in meters of vehicle center from the line
        self.line_base_pos = [] 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx_left = [] 
        self.allx_right = [] 
        #y values for detected line pixels
        self.ally_left = []
        self.ally_right = []
        self.mtx = mtx
        self.dist = dist
        self.__n = []
        
    # Check if the detection makes sense   
    def sanity_check(self):
        parall = np.mean((np.array(self.radius_of_curvature_left[-1]) -\
                         np.array(self.radius_of_curvature_right[-1]))) 
 
        xm_per_pix=3.7/700
        distance = np.mean(list(map(lambda x: x[0]-x[1], zip(self.allx_right[-1], self.allx_left[-1]))))
        distance = (self.current_xfit_right[-1][0] - self.current_xfit_left[-1][0])*xm_per_pix
        print("parall :",parall)
        print("distance :",distance)
        
        if  np.shape(self.current_fit_left)[0] >= 2:
            self.diffs = np.sum(np.abs(self.current_fit_left[-1] - self.current_fit_left[-2]) + \
                     np.abs(self.current_fit_right[-1] - self.current_fit_right[-2]))
        if (parall < 8500 and (distance > 3.2 and distance < 4.5)) :
            self.detected = True
            self.__n.append(0)
        else:
            self.detected = False
            self.allx_left.pop()
            self.allx_right.pop()
            self.ally_left.pop()
            self.ally_right.pop()
            self.current_fit_left.pop()
            self.current_fit_right.pop()
            self.line_base_pos.pop()
            self.current_xfit_left.pop()  
            self.current_xfit_right.pop()
            self.__n.append(1) 
    
    # average the last n frame image to improve robust 
    def smoothing(self):
        # the number of frame used  
        backon = 8
        if len(self.__n) < backon:
            num = len(self.__n) - sum(self.__n)
        else:
            num = backon - sum(self.__n[-1:-backon-1:-1])
         
        if num == 0:
            # if successive num frame are bad frame then reset and 
            # start searching from scratch
            self.best_fit_left = np.array([False,False,False])  
            self.best_fit_right = np.array([False,False,False])  
            self.__n = []
        else:
            # take an average over num past measurements to obtain better  
            # lane position and fit coefficients  
            aver_y_left = []
            aver_x_left = []
            aver_y_right = []
            aver_x_right = []
            average_fit_left = []
            average_fit_right =[]

            for index in range(num):
                aver_y_left.extend((self.ally_left[-1-index].T)[0])
                aver_x_left.extend((self.allx_left[-1-index].T)[0])
               
                aver_y_right.extend((self.ally_right[-1-index].T)[0])
                aver_x_right.extend((self.allx_right[-1-index].T)[0])
                
                average_fit_left.append((self.current_fit_left[-1-index].T)[0])
                average_fit_right.append((self.current_fit_right[-1-index].T)[0])
            
            #left_fit = np.polyfit(np.array(aver_y_left), np.array(aver_x_left), 2)           
            #right_fit = np.polyfit(np.array(aver_y_right), np.array(aver_x_right), 2)
            
            self.bestx_left = aver_x_left
            self.bestx_right = aver_x_right
            self.best_fit_left = np.mean(average_fit_left,axis=0)  
            self.best_fit_right = np.mean(average_fit_right,axis=0)
            #self.best_fit_left = left_fit  
            #self.best_fit_right = right_fit
    
    # To find line pixels based on current fit coefficients when have
    # a new warped binary image from the next frame of video 
    def look_ahead_Filter(self, binary_warped):
        # get current fit coefficients
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fit = (self.current_fit_left[-1].T)[0]
        right_fit = (self.current_fit_right[-1].T)[0]

        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 40
        left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
        right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  
        
        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        if (leftx != [] and righty != []):
            left_fit = np.polyfit(lefty, leftx, 2)
            right_fit = np.polyfit(righty, rightx, 2)
            
            # Generate x and y values for plotting
            ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
            
            # append to current measurments and fit coefficients       
            self.allx_left.append(leftx[:,None])
            self.allx_right.append(rightx[:,None])
            
            self.ally_left.append(lefty[:,None])
            self.ally_right.append(righty[:,None])
            
            self.current_fit_left.append(left_fit[:,None])
            self.current_fit_right.append(right_fit[:,None])
            
            self.current_xfit_left.append(left_fitx[:,None])  
            self.current_xfit_right.append(right_fitx[:,None])
            
            ym_per_pix = 30/720
            xm_per_pix = 3.7/700
            self.meas_cur(ploty,left_fit,left_fitx,right_fit,right_fitx,ym_per_pix,xm_per_pix)
            
            # The offset of the center of the image from the left lane
            offset = (binary_warped.shape[0] / 2 - leftx[0]) * xm_per_pix
            self.line_base_pos.append(offset)
            
            self.sanity_check()
        else:
            self.detected = False
            self.__n.append(1)

            # converting pixel values to real world space
        
            
        
    # calculate radius of curvature
    def meas_cur(self,ploty,left_fit,leftx,right_fit,rightx,ym_per_pix=30/720,xm_per_pix=3.7/700):
        #y_eval = np.max(ploty)
        #left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
        #right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
            
        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
        
        # Calculate the new radii of curvature
        #left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        #right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
        left_curverad = ((1 + (2*left_fit_cr[0]*ploty*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*ploty*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
       
        # Now our radius of curvature is in meters
        self.radius_of_curvature_left.append(left_curverad[:,None])
        self.radius_of_curvature_right.append(right_curverad[:,None])
    
    # Find lane line from scratch
    def find_lines(self,binary_warped):
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(binary_warped.shape[0]/nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 40
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []
    
        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
            
        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
        
        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds] 
        print("-----",leftx)
        
        if  (leftx != [] and righty != []):
            # Fit a second order polynomial to each
            left_fit = np.polyfit(lefty, leftx, 2)
            right_fit = np.polyfit(righty, rightx, 2)
            # Generate x and y values for plotting
            ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
            
            self.allx_left.append(leftx[:,None])
            self.allx_right.append(rightx[:,None])
            
            
            self.ally_left.append(lefty[:,None])
            self.ally_right.append(righty[:,None])
            
            self.current_fit_left.append(left_fit[:,None])
            self.current_fit_right.append(right_fit[:,None])
            
            self.current_xfit_left.append(left_fitx[:,None])  
            self.current_xfit_right.append(right_fitx[:,None])
        
            ym_per_pix = 30/720
            xm_per_pix = 3.7/700
            self.meas_cur(ploty,left_fit,left_fitx,right_fit,right_fitx,ym_per_pix,xm_per_pix)
            offset = (binary_warped.shape[1] / 2 - (left_fitx[-1] + right_fitx[-1]) / 2) * xm_per_pix        
            self.line_base_pos.append(offset)
            
            self.sanity_check()
        else:
            self.detected = False
            self.__n.append(1)

    
    # Project the measurement back down onto the road     
    def draw_back(self,warped,perspective_M):
        ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
        
        if (self.best_fit_left.T == [np.array([False,False,False])]).any() or \
                (self.best_fit_right.T == [np.array([False,False,False])]).any():
            left_fitx = self.current_xfit_left[-1].T
            right_fitx = self.current_xfit_right[-1].T
        else:
            left_fit = (self.best_fit_left)
            right_fit = (self.best_fit_right)
            
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        # Create an image to draw the lines on left_fitx,right_fitx, ploty
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        
        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
        
        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
        
        Minv = matrix(perspective_M).I
        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, (color_warp.shape[1], color_warp.shape[0])) 

        return newwarp
        
    # Detection lane line     
    def lane_finding(self,image):
        img_size = (image.shape[1], image.shape[0])
        
        # For source points I'm grabbing the outer four detected corners
        src = np.float32([[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],\
            [((img_size[0] / 6) - 10), img_size[1]],\
            [(img_size[0] * 5 / 6) + 55, img_size[1]],\
            [(img_size[0] / 2 + 60), img_size[1] / 2 + 100]])
        
        dst = np.float32([[(img_size[0] / 4), 0],\
            [(img_size[0] / 4), img_size[1]],\
            [(img_size[0] * 3 / 4), img_size[1]],\
            [(img_size[0] * 3 / 4), 0]])
        
        top_down, perspective_M = infu.persp_trans(image, src, dst, self.mtx, self.dist)   
        
        _,binary_warped = infu.color_thresholding(top_down, s_thresh=(50, 255), sx_thresh=(30, 100),sobel_kernel=3)
  
        if (self.best_fit_left.T == [np.array([False,False,False])]).any() or \
                (self.best_fit_right.T == [np.array([False,False,False])]).any():
            self.find_lines(binary_warped)
            # smothing and set self.bestx ,self.best_fit
            if self.detected:
                self.bestx_left = self.allx_left[-1]
                self.bestx_right = self.allx_right[-1]
                self.best_fit_left = ((self.current_fit_left[-1]).T)[0]
                self.best_fit_right = ((self.current_fit_right[-1]).T)[0]
                #print("****", self.best_fit_left)
        else:
            self.look_ahead_Filter(binary_warped)
            # smothing and set self.bestx ,self.best_fit
            self.smoothing()
       
        # Combine the result with the original image
        newwarp = self.draw_back(binary_warped,perspective_M) 
        result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)
        
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        print('----',self.line_base_pos[-1])
        cv2.putText(result,'Radius of Curvature = ' +\
                    round(self.radius_of_curvature_left[-1][-1][0],4).astype('str')\
                    +'(m)',(200,100), font, 1.5,(255,255,255),2,cv2.LINE_AA)
        cv2.putText(result,'Vehicle is ' + round(self.line_base_pos[-1],4).astype('str')\
                    +'( left of center)',(200,200), font, 1.5,(255,255,255),2,cv2.LINE_AA)
        
        return result
    

dist_pickle = pickle.load( open( "./examples/calibration_wide/wide_dist_pickle.p", "rb" ) )
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

line_detected = Line(mtx,dist)
white_output = 'project_video.mp4'

clip1 = VideoFileClip(white_output)
lane_clip = clip1.fl_image(line_detected.lane_finding)
lane_clip.write_videofile("./new_project_video.mp4")
