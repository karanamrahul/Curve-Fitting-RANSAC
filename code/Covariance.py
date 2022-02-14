from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import seaborn as sns

plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (12, 8)

# Extract the excel data to a pandas dataframe
# Please change the filepath to your current path
data = pd.read_excel(
    r'docs/ENPM673_hw1_linear_regression_dataset.xlsx')

# Separate the age and charges columns to individual numpy array
x = data['age'].to_numpy(int)
y = data['charges'].to_numpy(np.float64)

# Stack both the ages and charges 
h_data = np.vstack((x, y)).T

#print(h_data.shape)
#  Covariance
def cov(x, y):
    x_, y_ = x.mean(), y.mean()
    return np.sum((x - x_)*(y - y_))/(len(x) - 1)

# Covariance matrix
def cov_mat(mat):
    return np.array([[cov(mat[0], mat[0]), cov(mat[0], mat[1])], 
                     [cov(mat[1], mat[0]), cov(mat[1], mat[1])]])

#Calculate covariance matrix for the health data h_data shape : (325 x 2) 
covar=cov_mat(h_data.T) 


# Now we center the matrix at the origin to show the eigen vectors along with the data
#h_data = h_data - np.mean(h_data, 0)
# We can see the covariance matrix as a heatmap
#sns.heatmap(covar,xticklabels=['age','charges'], yticklabels=['age','charges'])

cov_val, cov_v = np.linalg.eig(covar)

# Scatter plot to show the age vs charges
plt.scatter(h_data[:, 0], h_data[:, 1])


# Shifted the origin to show the spread of eigen vectors in the sample data
origin=[np.mean(h_data[:,0]),np.mean(h_data[:,1])]
eig1=cov_v[:,0]
eig2=cov_v[:,1]
print(cov_val)

plt.quiver(*origin, *eig1, color=['r'], scale=17,label='Eigen Vector-1')
plt.quiver(*origin, *eig2, color=['b'], scale=17,label='Eigen Vector-2')
plt.title('Covariance Matrix with eigen vectors')
plt.xlabel("Age")
plt.ylabel('Charges')
plt.legend()


plt.show()
