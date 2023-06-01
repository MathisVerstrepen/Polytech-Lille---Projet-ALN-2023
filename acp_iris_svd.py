import numpy as np
import matplotlib.pyplot as plt

from base.irisExample.iris import *
from functions.svd.svd import svd


np.set_printoptions(linewidth=160)

m, n = A.shape
for k in range(0, n):
    A[:, k] = A[:, k] - np.average(A[:, k])

# Calcul d'ACP par la méthode moderne
#   (SVD de la matrice A sans passer par la
#    matrice de covariance)

U, Sigma, Vt = svd(A)
Vt = np.flip (Vt, axis=0)
V_max = np.transpose (Vt[0:2,:])

# V_max contient les deux vecteurs singuliers droits
#   associés aux deux valeurs singulières les plus
#   grandes. Rq : V_max = Q_max !

# On projette
P = np.dot (A, V_max)

# Visualisation
plt.scatter (P[0:50,0], P[0:50,1], color='blue')
plt.scatter (P[50:100,0], P[50:100,1], color='green')
plt.scatter (P[100:150,0], P[100:150,1], color='red')
plt.show ()
