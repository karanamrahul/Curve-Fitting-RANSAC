#!/usr/bin/env python3

import numpy as np
import cv2
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Least square solution for a parabola

def func(x, a, b , c):
    y = a * x ** 2 + b*x + c
    return y

    # Please use your current path or /docs/ball_video1.mp4
filepath1="docs/ball_video1.mp4"
filepath2="docs/ball_video2.mp4"

def LS(x, y):
    x=x[:,np.newaxis]
    y=y[:,np.newaxis]
    x = x[:,0]
    y = y[:,0]
    x_2 = np.power(x, 2)

    x = np.stack((x_2, x, np.ones((len(x)), dtype = int)), axis = 1)
    eig_val, _ = np.linalg.eig(np.matmul(x.T, x))
    if np.any(eig_val < 0):
         print(
             "[WARNING] This matrix should be positive semi definite i.e eigen_val > 0")
    return np.matmul(np.linalg.inv(np.matmul(x.T, x)), np.matmul(x.T, y))

def get_center(filepath):
    # Load the video 
    cap = cv2.VideoCapture(filepath)


    X=[]
    Y=[]
    while cap.isOpened():
        # Load image by each frame
        ret,frame = cap.read()
    # check if the frame is return is true or not
        if not ret:
         print("[WARNING] : FAIL")
         break
    #  Convert to grayscale and threshold

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        ret, thresh = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
    
        frame[thresh == 255] = 0

    # Find contours, draw on image and save
        contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # You can use the below method to display the contours.
        #cv2.drawContours(frame, contours, -1, (0,255,0), 3)
        #cv2.imwrite('result.png',frame)
        (cen_x,cen_y),radius = cv2.minEnclosingCircle(contours[1])
        X.append(int(cen_x))
        Y.append(int(cen_y))
    
    return X,Y


ball1_x,ball1_y=np.asarray(get_center(filepath1))
ball2_x,ball2_y=np.asarray(get_center(filepath2))

ls_sol_1=LS(ball1_x,ball1_y)

ls_sol_2=LS(ball2_x,ball2_y)


fig, axes = plt.subplots(1,2)
axes[0].plot(ball1_x, ball1_y, marker='^',alpha=0.7,c='m',label="Video-1")
axes[0].invert_yaxis()
axes[0].title.set_text('Trajectory for the video-1')
y_ls1 = [ls_sol_1[0]*i**2+ls_sol_1[1]*i+ls_sol_1[2] for i in ball1_x]
axes[0].plot(ball1_x, y_ls1,c='cyan',linestyle='dashdot',label="Standard Least Square")
axes[0].legend()
plt.xlabel("X-axis")
plt.ylabel('Y-axis')
     

axes[1].plot(ball2_x, ball2_y, marker='^',alpha=0.7,c='m',label="Video-2")
axes[1].invert_yaxis()
axes[1].title.set_text('Trajectory for the video-2')
y_ls2 = [ls_sol_2[0]*i**2+ls_sol_2[1]*i+ls_sol_2[2] for i in ball2_x]
axes[1].plot(ball2_x, y_ls2,c='cyan',linestyle='dashdot',label="Standard Least Square")
plt.legend()
plt.show()


