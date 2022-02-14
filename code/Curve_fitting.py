
from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import math
from scipy import rand
from sklearn.metrics import mean_squared_error
from sympy import per

data = pd.read_excel(
    r'docs/ENPM673_hw1_linear_regression_dataset.xlsx')
# Separate the age and charges columns to individual numpy array
x = data['age'].to_numpy(int)
y = data['charges'].to_numpy(np.float64)
# Stack both the ages and charges
h_data = np.vstack((x, y)).T
std=[[x.std(),y.std()]]

# We add ones to the other column to represent the line equation y = m*x + b where the slope is
# be considered as 1
x_line = x[:, np.newaxis]
ones = np.ones(len(x_line))

ones = ones[:, np.newaxis]
# zeros=np.zeros(len(x))

x_line = np.hstack((x_line, ones))

# we use the below function to return two set of points from the x,y array.
def rand_sel(x, y):
    return np.array([[random.choice(x), random.choice(x)]]), np.array([[random.choice(y), random.choice(y)]])
# This method calculates the perpendicular distance between a point and line.
def per_dis(p_x,p_y,x,y):
    
    p1=np.asarray((p_x[0][0],p_y[0][0]))
    p2=np.asarray((p_x[0][1],p_y[0][1]))
    p3=np.asarray((x,y))
    return np.linalg.norm(np.cross(p2-p1, p1-p3))/np.linalg.norm(p2-p1)
    
def LS(x, y):
    eig_val, _ = np.linalg.eig(np.matmul(x.T, x))
    if np.any(eig_val < 0):
         print(
             "[WARNING] This matrix should be positive semi definite i.e eigen_val > 0")
    return np.matmul(np.linalg.inv(np.matmul(x.T, x)), np.matmul(x.T, y))


def TLS(x, y):

    x_mean = np.mean(x)
    y_mean = np.mean(y)
    u1 = x-x_mean
    u2 = y-y_mean
    u1 = u1[:, np.newaxis]
    u2 = u2[:, np.newaxis]
    U = np.hstack((u1, u2))
    U = U.T.dot(U)
    eig_u, vec_u = np.linalg.eig(U)
    sig_min = np.argmin(eig_u)
    min_u = vec_u[:, sig_min]
    return min_u

def RANSAC(x, y,threshold):
    n_best = 0
    n_iter=2000
    count=0
    max_inliers=0
    for i in range(0, n_iter):
        # Select subset of points from the dataset - 4
      P_x, P_y = rand_sel(x,y)
      count=0
      if P_x[0][0] != P_x[0][1]:
        # err = np.sqrt((y_old - P_y_new)**2/(len(x)-2))
        # sigma=np.std(P_y_new)
        for i in range(len(x)):
            dis=per_dis(P_x,P_y,x[i],y[i])
            if dis < threshold:
                count+=1
      if count > max_inliers:
            max_inliers=count
            n_best_x,n_best_y = P_x,P_y
    
    m=(n_best_y[0][1]-n_best_y[0][0]) / (n_best_x[0][1]-n_best_x[0][0]) 
    b= n_best_y[0][0] - m*n_best_x[0][0]

    return [m,b]

def plot_LS(x, h_data, ls_sol):
    # Scatter plot to show the age vs charges

      plt.scatter(h_data[:, 0], h_data[:, 1],marker='^',alpha=0.7,c='m')
      y_ls = [ls_sol[0]*i+ls_sol[1] for i in h_data[:,0]]
      err=mse(h_data[:,1],y_ls)
      print("The Mean Square Error for LS is :",err)
      plt.plot(x, y_ls,c='cyan',linestyle='dashdot',label="LS")
      plt.title('Least Squares best fit line ')
      plt.xlabel("Age")
      plt.ylabel('Charges')
      plt.legend()
      plt.show()

def plot_TLS(x, h_data, tls_sol):
    # Scatter plot to show the age vs charges
        y_tls = []
        plt.scatter(h_data[:, 0], h_data[:, 1],marker='^',alpha=0.7,c='m')
        x_ = np.mean(x)
        d = tls_sol[0]*x_+tls_sol[1]
        for i in range(0, len(x)):
            y_ = (d-(tls_sol[0]*x[i])) / tls_sol[1]
            y_tls.append(y_)
        err=mse(h_data[:,1],y_tls)
        print("The Mean Square Error for TLS is :",err)
        plt.plot(x, y_tls,color='green',linestyle='dashed',label="TLS")
        plt.title('Total Least Squares best fit line')
        plt.xlabel("Age")
        plt.ylabel('Charges')
        plt.legend()
        plt.show()

def plot_RANSAC(x, h_data, ran_sol):
    # Scatter plot to show the age vs charges

      plt.scatter(h_data[:, 0], h_data[:, 1],marker='^',alpha=0.7,c='m')
      y_ls = [ran_sol[0]*i+ran_sol[1] for i in h_data[:,0]]
      err=mse(h_data[:,1],y_ls)
      print("The Mean Square Error for RANSAC is :",err)
      plt.plot(x, y_ls,c='black',linestyle='dotted',label="RANSAC")
      plt.title('RANSAC best fit line ')
      plt.xlabel("Age")
      plt.ylabel('Charges')
      plt.legend()
      plt.show()


def mse(y_old,y_new):
    return mean_squared_error(y_old,y_new)
    
LS_sol = LS(x_line, h_data[:, 1])
TLS_sol = TLS(x, y)
plot_LS(x,h_data,LS_sol)
plot_TLS(x,h_data,TLS_sol)
ran_sol=RANSAC(x,h_data[:,1],8)
plot_RANSAC(x,h_data,ran_sol)
