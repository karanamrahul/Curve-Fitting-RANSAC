import numpy as np
import sympy as sp
from sympy import Matrix
import math
import pprint

# Problem-4

# TODO: Find the Homography Matrix for the given two plane object and point correspondences.

# Corresponding points of the two planes

x = [5, 150, 150, 5]
x1 = x[0]
x2 = x[1]
x3 = x[2]
x4 = x[3]
x_p = [100, 200, 220, 100]
x_1 = x_p[0]
x_2 = x_p[1]
x_3 = x_p[2]
x_4 = x_p[3]
y = [5, 5, 150, 150]
y1 = y[0]
y2 = y[1]
y3 = y[2]
y4 = y[3]
y_p = [100, 80, 80, 200]
y_1 = y_p[0]
y_2 = y_p[1]
y_3 = y_p[2]
y_4 = y_p[3]

# The Matrix which we use to find the solution to the homogenous system of equations AX = 0.
A = np.matrix([[-x1, -y1,  -1, 0, 0, 0, x1*x_1, y1*x_1, x_1],
               [0, 0, 0, -x1, -y1, -1, x1*y_1, y1*y_1, y_1],
               [-x2, -y2, -1, 0, 0, 0, x2*x_2, y2*x_2, x_2],
               [0, 0, 0, -x2, -y2, -1, x2*y_2, y2*y_2, y_2],
               [-x3, -y3, -1, 0, 0, 0, x3*x_3, y3*x_3, x_3],
               [0, 0, 0, -x3, -y3, -1, x3*y_3, y3*y_3, y_3],
               [-x4, -y4, -1, 0, 0, 0, x4*x_4, y4*x_4, x_4],
               [0, 0, 0, -x4, -y4, -1, x4*y_4, y4*y_4, y_4]])
h11,h12 ,h13,h21,h22,h23,h31,h32,h33 = sp.symbols('h11 h12 h13 h21 h22 h23 h31 h32 h33')
X = [h11,h12 ,h13,h21,h22,h23,h31,h32,h33]
B=np.mat(np.zeros(8)).transpose()


# Singular Value Decomposition  SVD(A)= U*S*V
# Dimensions of A: m x n , U: m x m , S: m x n , V: n x n

# Class SVD 
# nullspace : This method is used to calculate the least square solution or find the solution for a homogenous system of equation.
# svd : This method will decompose A into U,Sigma and V matrices.
# p_inv : This method will return the pseudo inverse of non-square matrix

class svd:
    def __init__(self) -> None:
        pass
    def nullspace(self,mat):
         u = mat*mat.T
         v = mat.T*mat
         val_u,vec_u=np.linalg.eig(u)
         val_v,vec_v=np.linalg.eig(v)
        # Sorting eigen vectors based upon the decreasing order of eigen values
         idx = val_v.argsort()[::-1]   
         val_v = val_v[idx]
         vec_v = vec_v[:,idx]

         idx = val_u.argsort()[::-1]   
         val_u = val_u[idx]
         vec_u = vec_u[:,idx]

         if mat.shape[0] > mat.shape[1]:
            s_val=np.mat(np.sqrt(np.diag(val_u)))
            s_val=s_val[:,:mat.shape[1]]
         else:
             s_val=np.mat(np.sqrt(np.diag(val_v)))
             s_val=s_val[:mat.shape[0],:]
         return vec_v.T[-1]
        
    def svd(self,mat):
         u = mat*mat.T
         v = mat.T*mat
         val_u,vec_u=np.linalg.eig(u)
         val_v,vec_v=np.linalg.eig(v)

         idx = val_v.argsort()[::-1]   
         val_v = val_v[idx]
         vec_v = vec_v[:,idx]

         idx = val_u.argsort()[::-1]   
         val_u = val_u[idx]
         vec_u = vec_u[:,idx]

         if mat.shape[0] > mat.shape[1]:
            s_val=np.mat(np.sqrt(np.diag(val_u)))
            s_val=s_val[:,:mat.shape[1]]
         else:
             s_val=np.mat(np.sqrt(np.diag(val_v)))
             s_val=s_val[:mat.shape[0],:]
         return [vec_u,s_val,vec_v.T]

    def p_inv(self,mat):
         u = mat*mat.T
         v = mat.T*mat
         val_u,vec_u=np.linalg.eig(u)
         val_v,vec_v=np.linalg.eig(v)

         idx = val_v.argsort()[::-1]   
         val_v = val_v[idx]
         vec_v = vec_v[:,idx]

         idx = val_u.argsort()[::-1]   
         val_u = val_u[idx]
         vec_u = vec_u[:,idx]

         if mat.shape[0] > mat.shape[1]:
            s_val=np.mat(np.sqrt(np.diag(sorted(val_u,reverse=True))))
            s_val_inv=np.linalg.inv(s_val)
            s_val_inv=s_val[:mat.shape[0],:]
            s_val=s_val[:,:mat.shape[0]]
            
         else:
             s_val=np.mat(np.sqrt(np.diag(sorted(val_v,reverse=True))))
             s_val_inv=np.linalg.inv(s_val)
             s_val_inv=s_val[:,:mat.shape[0]]
             s_val=s_val[:mat.shape[0],:]
             
         
         return vec_v.dot(s_val_inv).dot(vec_u.T)




if __name__ == "__main__":
    s=svd()
    null=s.nullspace(A)
    u,s,v=s.svd(A)
    np.set_printoptions(formatter={'float_kind':'{:f}'.format})
    print("Homography Matrix ")
    print(null.reshape((3,3)))
   