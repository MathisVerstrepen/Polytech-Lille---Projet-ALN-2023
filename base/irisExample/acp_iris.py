import numpy as np
import scipy.linalg as nla
import matplotlib.pyplot as plt

from iris import *

np.set_printoptions(linewidth=160)

m, n = A.shape
for k in range (0,n) :
    A[:,k] = A[:,k] - np.average(A[:,k])

# Calcul d'ACP en deux dimensions par la méthode 
#   historique (matrice de covariance + calcul
#   des valeurs propres d'une mat. symétrique)

# C = 1/(n-1) * np.dot (np.transpose (A), A)

# lmbda, Q = nla.eigh (C)

# print(lmbda)
# print(Q)

# Q_max = np.empty ([n,2], dtype=np.float64)
# Q_max[:,0] = Q[:,3]
# Q_max[:,1] = Q[:,2]

# Chaque iris (= ligne de A) est un point dans R**4
# On projette chaque point 
# - dans le plan défini par les deux vecteurs propres
# - orthogonalement
# P = np.dot (A, Q_max)

# # Visualisation
# plt.scatter (P[0:50,0], P[0:50,1], color='blue')
# plt.scatter (P[50:100,0], P[50:100,1], color='green')
# plt.scatter (P[100:150,0], P[100:150,1], color='red')
# plt.show ()

# # Même calcul d'ACP par la méthode moderne 
# #   (SVD de la matrice A sans passer par la
# #    matrice de covariance)

U, Sigma, Vt = nla.svd(A, full_matrices=False)
V_max = np.transpose (Vt[0:2,:])

# # V_max contient les deux vecteurs singuliers droits
# #   associés aux deux valeurs singulières les plus
# #   grandes. Rq : V_max = Q_max !

# # On projette
P = np.dot (A, V_max)

# # Visualisation
plt.scatter (P[0:50,0], P[0:50,1], color='blue')
plt.scatter (P[50:100,0], P[50:100,1], color='green')
plt.scatter (P[100:150,0], P[100:150,1], color='red')
plt.show ()

