# Perception


Curve Fittings using Least squares, Total Least Squares and RANSAC.

Finding Homography Matrix using Singular Value Decompositon


├── Code
|  ├── Curve_fitting.py
|  ├── Homography_SVD.py
|  ├── balltrack.py
|  ├── Covariance.py
├── Docs
|  ├── ball_video1.mp4
|  ├── ball_video2.mp4
|  ├── dataset.xlsx
|  ├── Report.pdf
├── Results
|  |  ├── .png files

## Problem 1

A ball is thrown against a white background and a camera sensor is used to track its
trajectory. We have a near perfect sensor tracking the ball in video1 and the second
sensor is faulty and tracks the ball as shown in video2. Clearly, there is no noise added
to the first video whereas there is significant noise in the second video. Assuming that
the trajectory of the ball follows the equation of a parabola:

1. Use Standard Least Squares to fit curves to the given videos in each case. You
have to plot the data and your best fit curve for each case. Submit your code
along with the instructions to run it.



### Steps to Run the code

```
git clone --recursive https://github.com/karanamrahul/Perception.git 
cd Perception
python3 balltrack.py 

```

### Results

![alt test](https://github.com/karanamrahul/Perception/results/ball.png )


## Problem 2
In the above problem, we used the least squares method to fit a curve. However, if the
data is scattered, this might not be the best choice for curve fitting. In this problem, you
are given data for health insurance costs based on the person’s age. There are other
fields as well, but you have to fit a line only for age and insurance cost data. The data is
given in .csv file format and can be downloaded from here(dataset.xlsx).

1. Compute the covariance matrix (from scratch) and find its eigenvalues and
eigenvectors. Plot the eigenvectors on the same graph as the data. Refer to this
article for better understanding. 

2. Fit a line to the data using linear least square method, total least square method
and RANSAC. Plot the result for each method and explain
drawbacks/advantages for each. 


### Steps to Run the code

```
git clone --recursive https://github.com/karanamrahul/Perception.git 
cd Perception
python3 Covariance.py 
python3 Curve_fitting.py

```
### Results

##### Covariance 
![alt test](https://github.com/karanamrahul/Perception/results/covariance.png )
##### Least Squares 
![alt test](https://github.com/karanamrahul/Perception/results/LS.png )
##### Total Least Squares 
![alt test](https://github.com/karanamrahul/Perception/results/TLS.png )
##### RANSAC
![alt test](https://github.com/karanamrahul/Perception/results/RANSAC.png )


## Problem 3
![alt test](https://github.com/karanamrahul/Perception/docs/homography.png )

Compute Homography Matrix for the below given points.

Compute SVD of the Matrix A using python.

### Steps to Run the code

```
git clone --recursive https://github.com/karanamrahul/Perception.git 
cd Perception
python3 Homography_SVD.py 


```