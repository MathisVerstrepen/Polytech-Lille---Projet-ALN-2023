import unittest
import numpy as np
import scipy.linalg as nla

from functions.eighvals.bissection import sturm_seq, count_sign_changes, bissection
from functions.eighvals.gershgorin import gershgorin_trig
from functions.eighvals.hessenberg import hessenberg
from functions.eighvals.puissance_inverse import puissance_inverse, lu_factor, solve_triangular
from functions.eighvals.eigh import eigh

from functions.svd.bidiagonal import bidiagonal, reflecteur
from functions.svd.svd import svd


class T1_Hessenberg(unittest.TestCase):
    def test_hessenberg_2x2_matrix(self):
        A = np.array([[2.0, 1.0],
                    [1.0, 3.0]])
        expected_H, expected_Q = nla.hessenberg(A, calc_q=True)
        
        H, Q = hessenberg(A, calc_q=True)
        A_verif = np.dot (Q, np.dot (H, np.transpose(Q)))
        
        np.testing.assert_array_almost_equal(np.absolute(H) , np.absolute(expected_H), decimal=5)
        np.testing.assert_array_almost_equal(np.absolute(Q), np.absolute(expected_Q), decimal=5)
        np.testing.assert_array_almost_equal(A_verif, A, decimal=5)
        
    def test_hessenberg_3x3_matrix(self):
        A = np.array([[0.37454, 0.95071, 0.00000],
                    [0.95071, 0.73199, 0.59865],
                    [0.00000, 0.59865, 0.15602]])
        expected_H, expected_Q = nla.hessenberg(A, calc_q=True)
        
        H, Q = hessenberg(A, calc_q=True)
        A_verif = np.dot (Q, np.dot (H, np.transpose(Q)))
        
        np.testing.assert_array_almost_equal(np.absolute(H) , np.absolute(expected_H), decimal=5)
        np.testing.assert_array_almost_equal(np.absolute(Q), np.absolute(expected_Q), decimal=5)
        np.testing.assert_array_almost_equal(A_verif, A, decimal=5)
        
class T2_Gershgorin(unittest.TestCase):
    def test_gershgorin_2x2_matrix(self):
        A = np.array([[2.0, 1.0],
                    [1.0, 3.0]])
        expected_eigenvalues_inter = np.array([1.0, 4.0])
        
        eigenvalues = gershgorin_trig(A)
        
        np.testing.assert_array_almost_equal(np.sort(eigenvalues), np.sort(expected_eigenvalues_inter), decimal=5)
        
    def test_gershgorin_3x3_matrix(self):
        A = np.array([[0.37454, 0.95071, 0.00000],
                    [0.95071, 0.73199, 0.59865],
                    [0.00000, 0.59865, 0.15602]])
        expected_eigenvalues_inter = np.array([-0.81737,  2.28135])
        
        eigenvalues = gershgorin_trig(A)
        
        np.testing.assert_array_almost_equal(np.sort(eigenvalues), np.sort(expected_eigenvalues_inter), decimal=5)
        
    def test_gershgorin_4x4_matrix(self):
        A = np.array([[0.37454, 0.95071, 0.00000, 0.00000],
                    [0.95071, 0.73199, 0.59865, 0.00000],
                    [0.00000, 0.59865, 0.15602, 0.7864],
                    [0.00000, 0.00000, 0.4532, 0.834756]])
        expected_eigenvalues_inter = np.array([-1.22903,  2.28135])
        
        eigenvalues = gershgorin_trig(A)
        
        np.testing.assert_array_almost_equal(np.sort(eigenvalues), np.sort(expected_eigenvalues_inter), decimal=5)
        
        
class T3_Bissection(unittest.TestCase):

    def test_bissection_2x2_diagonal_matrix(self):
        A = np.array([[2.0, 1.0],
                    [1.0, 3.0]])
        binf, bsup = (1, 4)
        precision = 1e-5

        eigenvalues = bissection(A, binf, bsup, precision)
        expected_eigenvalues = np.linalg.eigvalsh(A)

        np.testing.assert_array_almost_equal(np.sort(eigenvalues), np.sort(expected_eigenvalues), decimal=5)

    def test_bissection_3x3_tridiagonal_symmetric_matrix(self):
        A = np.array([[0.37454, 0.95071, 0.00000],
                    [0.95071, 0.73199, 0.59865],
                    [0.00000, 0.59865, 0.15602]])
        binf, bsup = (-0.81737, 2.28135)
        precision = 1e-5

        eigenvalues = bissection(A, binf, bsup, precision)
        expected_eigenvalues = np.linalg.eigvalsh(A)

        np.testing.assert_array_almost_equal(np.sort(eigenvalues), np.sort(expected_eigenvalues), decimal=5)

    def test_bissection_4x4_tridiagonal_symmetric_matrix(self):
        A = np.array([[0.37454, 0.95071, 0.00000, 0.00000],
                    [0.95071, 0.73199, 0.59865, 0.00000],
                    [0.00000, 0.59865, 0.15602, 0.15599],
                    [0.00000, 0.00000, 0.15599, 0.05808]])
        binf, bsup = (-0.81737, 2.28135)
        precision = 1e-5

        eigenvalues = bissection(A, binf, bsup, precision)
        expected_eigenvalues = np.linalg.eigvalsh(A)

        np.testing.assert_array_almost_equal(np.sort(eigenvalues), np.sort(expected_eigenvalues), decimal=5)
        
    def test_hessenberg_4x4_matrix(self):
        A = np.array([[0.37454, 0.95071, 0.00000, 0.00000],
                    [0.95071, 0.73199, 0.59865, 0.00000],
                    [0.00000, 0.59865, 0.15602, 0.15599],
                    [0.00000, 0.00000, 0.15599, 0.05808]])
        expected_H, expected_Q = nla.hessenberg(A, calc_q=True)
        
        H, Q = hessenberg(A, calc_q=True)
        A_verif = np.dot (Q, np.dot (H, np.transpose(Q)))
        
        np.testing.assert_array_almost_equal(np.absolute(H) , np.absolute(expected_H), decimal=5)
        np.testing.assert_array_almost_equal(np.absolute(Q), np.absolute(expected_Q), decimal=5)
        np.testing.assert_array_almost_equal(A_verif, A, decimal=5)

class T4_SturmSeq(unittest.TestCase):

    def test_tridiagonal_matrix(self):
        A = np.array([
            [1, -1, 0 ,0],
            [-1, 2, -1, 0],
            [0, -1, 3, -1],
            [0, 0, -1, 4]
        ])
        mu = 5
        result = sturm_seq(A, mu)
        expected = np.array([1, -4, 11, -18, 7]) # Calculé à la main
        np.testing.assert_array_almost_equal(result, expected)
        

    def test_tridiagonal_matrix2(self):
        A = np.array([
            [2, 1, 0],
            [1, 3, 1],
            [0, 1, 4]
        ])
        mu = 1
        result = sturm_seq(A, mu)
        expected = np.array([1, 1, 1, 2]) # Calculé à la main
        np.testing.assert_array_almost_equal(result, expected)
        
    def test_tridiagonal_matrix_3(self):
        A = np.array([
            [1, 1, 0 ,0],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 1, 1]
        ])
        mu = 2
        result = sturm_seq(A, mu)
        expected = np.array([1, -1, 0, 0, 0]) # Calculé à la main
        np.testing.assert_array_almost_equal(result, expected)

class T5_CountSignChange(unittest.TestCase):
    
    def test_count_sign_changes_1(self):
        det = np.array([1, 1, -3, 5, -2, 0, -8, 7, 11, -5, 4])
        result = count_sign_changes(det)
        expected = 6 # Calculé à la main
        self.assertEqual(result, expected)
        
    def test_count_sign_changes_2(self):
        det = np.array([1, -7, 3, -4, 1, -10, 6, 0, -9, 8, -2])
        result = count_sign_changes(det)
        expected = 9 # Calculé à la main
        self.assertEqual(result, expected)
        
    def test_count_sign_changes_3(self):
        det = np.array([1, 0, 7, -1, -8, 0, 2, 9, -3, 0, -5])
        result = count_sign_changes(det)
        expected = 3 # Calculé à la main
        self.assertEqual(result, expected)
        
class T6_puissance_inverse(unittest.TestCase):
    
    def test_puissance_inverse_1(self):
        A = np.array([[1, 2, 0],
                    [2, 2, 4],
                    [0, 4, 5]])
        
        eigs, eigenvec = nla.eig(A)
        eigs = np.real(eigs)
        
        for (idx, eig) in enumerate(eigs):
            v = puissance_inverse(A, eig)
            expected = eigenvec[:, idx]
            np.testing.assert_array_almost_equal(np.absolute(v), np.absolute(expected), decimal=5)
            
    def test_puissance_inverse_2(self):
        A = np.array([[8, 2, 0],
                    [4, 1, 4],
                    [0, 4, 3]])
        
        eigs, eigenvec = nla.eig(A)
        eigs = np.real(eigs)
        
        for (idx, eig) in enumerate(eigs):
            v = puissance_inverse(A, eig)
            expected = eigenvec[:, idx]
            np.testing.assert_array_almost_equal(np.absolute(v), np.absolute(expected), decimal=5)
            
    def test_puissance_inverse_3(self):
        A = np.array([[2, 2, 0],
                    [4, 9, 5],
                    [0, 1, 3]])
        
        eigs, eigenvec = nla.eig(A)
        eigs = np.real(eigs)
        
        for (idx, eig) in enumerate(eigs):
            v = puissance_inverse(A, eig)
            expected = eigenvec[:, idx]
            np.testing.assert_array_almost_equal(np.absolute(v), np.absolute(expected), decimal=5)
            
class T7_solve_triangular(unittest.TestCase):
        
        def test_solve_triangular_1(self):
            A = np.array([[1, 2, 0],
                        [2, 2, 4],
                        [0, 4, 5]])
            
            b = np.array([1, 2, 3])
            
            x = solve_triangular(A, b, lower=True, overwrite_b=False, unit_diagonal=True)
            expected = nla.solve_triangular(A, b, lower=True, overwrite_b=False, unit_diagonal=True)
            
            np.testing.assert_array_almost_equal(np.absolute(x), np.absolute(expected), decimal=5)
            
            
        def test_solve_triangular_2(self):
            A = np.array([[8, 2, 0],
                        [4, 1, 4],
                        [0, 4, 3]])
            
            b = np.array([1, 2, 3])
            
            x = solve_triangular(A, b, lower=True, overwrite_b=False, unit_diagonal=True)
            expected = nla.solve_triangular(A, b, lower=True, overwrite_b=False, unit_diagonal=True)
            
            np.testing.assert_array_almost_equal(np.absolute(x), np.absolute(expected), decimal=5)
            
        def test_solve_triangular_3(self):
            A = np.array([[2, 2, 0],
                        [4, 9, 5],
                        [0, 1, 3]])
            
            b = np.array([1, 2, 3])
            
            x = solve_triangular(A, b, lower=True, overwrite_b=False, unit_diagonal=True)
            expected = nla.solve_triangular(A, b, lower=True, overwrite_b=False, unit_diagonal=True)
                        
            np.testing.assert_array_almost_equal(np.absolute(x), np.absolute(expected), decimal=5)
            
class T8_lu_factor(unittest.TestCase):
    
     def test_lu_factor_1(self):
         A = np.array([[1, 2, 0],
                     [2, 2, 4],
                     [0, 4, 5]])
        
         expected = nla.lu_factor(A)
         result = lu_factor(A)
        
         np.testing.assert_array_almost_equal(np.absolute(result[0]), np.absolute(expected[0]), decimal=5)
         np.testing.assert_array_almost_equal(np.absolute(result[1]), np.absolute(expected[1]), decimal=5)
        
     def test_lu_factor_2(self):
         A = np.array([[8, 2, 0],
                     [4, 1, 4],
                     [0, 4, 3]])
        
         expected = nla.lu_factor(A)
         result = lu_factor(A)
        
         np.testing.assert_array_almost_equal(np.absolute(result[0]), np.absolute(expected[0]), decimal=5)
         np.testing.assert_array_almost_equal(np.absolute(result[1]), np.absolute(expected[1]), decimal=5)
        
     def test_lu_factor_3(self):
         A = np.array([[2, 2, 0],
                     [4, 9, 5],
                     [0, 1, 3]])
        
         expected = nla.lu_factor(A)
         result = lu_factor(A)
        
         np.testing.assert_array_almost_equal(np.absolute(result[0]), np.absolute(expected[0]), decimal=5)
         np.testing.assert_array_almost_equal(np.absolute(result[1]), np.absolute(expected[1]), decimal=5)

class T9_eigh(unittest.TestCase):
    
    def test_eigh_1(self):
        A = np.array([[1, 2, 0],
                    [2, 2, 4],
                    [0, 4, 5]])
        
        expected = nla.eigh(A)
        result = eigh(A)
        
        np.testing.assert_array_almost_equal(np.absolute(result[0]), np.absolute(expected[0]), decimal=5)
        np.testing.assert_array_almost_equal(np.absolute(result[1]), np.absolute(expected[1]), decimal=5)
        
        
    def test_eigh_2(self):
        A = np.array([[1, 2, 0],
                    [2, 1, 2],
                    [0, 2, 1]])
        
        expected = nla.eigh(A)
        result = eigh(A)
        
        np.testing.assert_array_almost_equal(np.absolute(result[0]), np.absolute(expected[0]), decimal=5)
        np.testing.assert_array_almost_equal(np.absolute(result[1]), np.absolute(expected[1]), decimal=5)
        
    def test_eigh_3(self):
        A = np.array([[1, 2, 0],
                    [2, 9, 2],
                    [0, 2, 1]])
        
        expected = nla.eigh(A)
        result = eigh(A)
        
        np.testing.assert_array_almost_equal(np.absolute(result[0]), np.absolute(expected[0]), decimal=5)
        np.testing.assert_array_almost_equal(np.absolute(result[1]), np.absolute(expected[1]), decimal=5)


class T10_bidiagonal(unittest.TestCase):
    
   def test_bidiagonal_2x2_matrix(self):
       A = np.array([[2.0, 1.0],
                   [1.0, 3.0]])
       
       # Calculated using sympy
       expected_B = np.array([[2.23606797749979, 2.23606797749979], 
                              [4.44089209850063e-16, -2.23606797749979]], dtype=np.float64)
       B = bidiagonal(A)[0]
       
       np.testing.assert_array_almost_equal(np.absolute(B) , np.absolute(expected_B), decimal=5)
        
   def test_bidiagonal_3x3_matrix(self):
       A = np.array([[0.37454, 0.95071, 0.00000],
                   [0.95071, 0.73199, 0.59865],
                   [0.00000, 0.59865, 0.15602]])
        
        # Calculated using sympy
       expected_B = np.array([[1.02182665638551, 1.17052999371909, 1.11022302462516e-16], 
                              [-2.22044604925031e-16, 0.743243859729525, 0.405605377987709], 
                              [0, 1.66533453693773e-16, 0.306099862342017]], dtype=np.float64)
       B = bidiagonal(A)[0]
        
       np.testing.assert_array_almost_equal(np.absolute(B) , np.absolute(expected_B), decimal=5)
        
   def test_bidiagonal_4x4_matrix(self):
       A = np.array([[0.37454, 0.95071, 0.04355, 0.4520],
                   [0.95071, 0.73199, 0.59865, 0.97654],
                   [0.45324, 0.59865, 0.15602, 0.15599],
                   [0.453245, 0.786345, 0.15599, 0.05808]])
       
        # Calculated using sympy
       expected_B = np.array([[1.20622852035798, 1.81199394669257, -1.11022302462516e-16, -2.77555756156289e-16],
                              [-2.77555756156289e-17, 0.457361274925780, 0.411175860999764, 5.89805981832114e-17],
                              [-8.32667268468867e-17, -1.66533453693773e-16, -0.458098299783882, -0.368381215847480],
                              [0, -1.94289029309402e-16, 0, -0.0620279739286370]], dtype=np.float64)
       B = bidiagonal(A)[0]
        
       np.testing.assert_array_almost_equal(np.absolute(B) , np.absolute(expected_B), decimal=5)
        
class T11_reflecteur(unittest.TestCase):
    
    def test_reflecteur_1(self):
        v = np.array([[1],
                      [2],
                      [3]])
        
        expected = np.array([[1,  4,  6],
                            [4,  7, 12],
                            [6, 12, 17]])
        result = reflecteur(3, v)
        
        np.testing.assert_array_almost_equal(np.absolute(result) , np.absolute(expected), decimal=5)
        
    def test_reflecteur_2(self):
        v = np.array([[1],
                      [2],
                      [3],
                      [4]])
        
        expected = np.array([[1,  4,  6,  8],
                            [4,  7, 12, 16],
                            [6, 12, 17, 24],
                            [8, 16, 24, 31]])
        result = reflecteur(4, v)
        
        np.testing.assert_array_almost_equal(np.absolute(result) , np.absolute(expected), decimal=5)
        
    def test_reflecteur_3(self):
        v = np.array([[1],
                      [2],
                      [3],
                      [4],
                      [5]])
        
        expected = np.array([[1,  4,  6,  8, 10],
                            [4,  7, 12, 16, 20],
                            [6, 12, 17, 24, 30],
                            [8, 16, 24, 31, 40],
                            [10, 20, 30, 40, 49]])
        result = reflecteur(5, v)
        
        np.testing.assert_array_almost_equal(np.absolute(result) , np.absolute(expected), decimal=5)
        
class T12_svd(unittest.TestCase):
    
    def test_svd_1(self):
        A = np.array([[-7/10, -109/25, 1/50], 
                      [7/10, -31/25, 209/50],
                     [23/10, -49/25, 161/50], 
                     [-23/10, -91/25, 49/50]])
        
        expected = nla.svd(A, full_matrices=False)
        result = svd(A)
        
        np.testing.assert_array_almost_equal(np.absolute(result[0]), np.absolute(expected[0]), decimal=4)
        np.testing.assert_array_almost_equal(np.absolute(result[1]), np.absolute(np.diag(np.sort(expected[1]))), decimal=4)
        np.testing.assert_array_almost_equal(np.absolute(np.flip(result[2], axis=0)), np.absolute(expected[2]), decimal=4) 
        # flip because of the sign of the eigenvectors

if __name__ == '__main__':
    unittest.main()
