import unittest
from librerias import *

class testLibrerias(unittest.TestCase):

    def testEsCuadrada(self):
        #cuadrada
        m = np.array([[1,2],
                      [5,4]])
        self.assertTrue(esCuadrada(m))
        #no cuadrada
        m = np.array([[1,2,3],
                      [5,4,3]])
        self.assertFalse(esCuadrada(m))
        #vector
        m = np.array([[1,2,3]])
        self.assertFalse(esCuadrada(m))

    def testTriangSup(self):
        m1 = np.array([[1,2],
                      [5,4]])
        m1copy = m1.copy()
        m2 = np.array([[1,2,3],
                      [5,4,3],
                      [2,4,5]])
        m2copy = m2.copy()
        m1expected = np.array([[0,2],
                      [0,0]])
        m2expected = np.array([[0,2,3],
                      [0,0,3],
                      [0,0,0]])
        np.testing.assert_array_equal(triangSup(m1),m1expected)
        np.testing.assert_array_equal(m1,m1copy)
        np.testing.assert_array_equal(triangSup(m2),m2expected)
        np.testing.assert_array_equal(m2,m2copy)

    def testDiagonal(self):
        m1 = np.array([[1,2],
                      [5,4]])
        m1copy = m1.copy()
        m2 = np.array([[1,2,3],
                      [5,4,3],
                      [2,4,5]])
        m2copy = m2.copy()
        m1expected = np.array([[1,0],
                              [0,4]])
        m2expected = np.array([[1,0,0],
                              [0,4,0],
                              [0,0,5]])
        np.testing.assert_array_equal(diagonal(m1),m1expected)
        np.testing.assert_array_equal(m1,m1copy)
        np.testing.assert_array_equal(diagonal(m2),m2expected)
        np.testing.assert_array_equal(m2,m2copy)

    def testTriangInf(self):
        m1 = np.array([[1,2],
                      [5,4]])
        m1copy = m1.copy()
        m2 = np.array([[1,2,3],
                      [5,4,3],
                      [2,4,5]])
        m2copy = m2.copy()
        m1expected = np.array([[0,0],
                      [5,0]])
        m2expected = np.array([[0,0,0],
                      [5,0,0],
                      [2,4,0]])
        np.testing.assert_array_equal(triangInf(m1),m1expected)
        np.testing.assert_array_equal(m1,m1copy)
        np.testing.assert_array_equal(triangInf(m2),m2expected)
        np.testing.assert_array_equal(m2,m2copy)

    def testTraza2x2(self):
        m1 = np.array([[3,4],
                       [12,7]])
        
        self.assertEqual(traza(m1), 10)
    
    def testTraza4x4(self):
        m1 = np.array([[3,4,2,1],
                       [12,7,5,6],
                       [5,6,2,8],
                       [8,9,10,-10]])
        
        self.assertEqual(traza(m1), 2)

    def testTrazaSinDiagonal(self):
        m1 = np.array([[0,4,2,1],
                       [12,0,5,6],
                       [5,6,0,8],
                       [8,9,10,0]])
        
        self.assertEqual(traza(m1), 0)

    def testTraspuesta2x2(self):
        m = np.array([[1,2],
                     [3,4]])
        mCopy = m.copy()
        expected = np.array([[1,3],
                            [2,4]])
        
        np.testing.assert_array_equal(traspuesta(m),expected)
        np.testing.assert_array_equal(m,mCopy)

    def testTraspuesta3x2(self):
        m = np.array([[1,2],
                     [3,4],
                     [4,5]])
        mCopy = m.copy()
        expected = np.array([[1,3,4],
                            [2,4,5]])
        
        np.testing.assert_array_equal(traspuesta(m),expected)
        np.testing.assert_array_equal(m,mCopy)

    def testTraspuestaSimetrica(self):
        m = np.array([[1,2,3],
                     [2,4,5],
                     [3,5,7]])
        
        np.testing.assert_array_equal(traspuesta(m),m)

    
    def testSimetricaTrue(self):

        m = np.array([[1,2,3],
                     [2,4,5],
                     [3,5,7]])
        
        self.assertTrue(esSimetrica(m))

    def testSimetricaFalse(self):

        m = np.array([[1,2,5],
                     [2,4,3],
                     [5,5,10]])
        
        self.assertFalse(esSimetrica(m))    
        
    def testCalcularAXMatrizCuadrada(self):
        m = np.array([[1,2],
                      [3,1]])

        x = np.array([2,2])

        np.testing.assert_array_equal(calcularAx(m,x),np.array([6,8]))


    def testCalcularAXMatrizNoCuadrada(self):
        m = np.array([[1,2],
                      [3,1],
                      [2,-2]])

        x = np.array([1,3])

        np.testing.assert_array_equal(calcularAx(m,x),np.array([7,6,-4]))


    def testIntercambiarFila(self):
        m = np.array([[1,2,3],
                     [4,5,6],
                     [7,8,9]])
        expected = np.array([[4,5,6],
                            [1,2,3],
                            [7,8,9]])  

        intercambiarFilas(m,0,1)
        np.testing.assert_array_equal(m,expected)

    def testIntercambiarFila2VecesVuelveOriginal(self):
        m = np.array([[1,2,3],
                     [4,5,6],
                     [7,8,9]])
        expected = np.array([
                            [1,2,3],
                            [4,5,6],
                            [7,8,9]])  

        intercambiarFilas(m,0,1)
        intercambiarFilas(m,0,1)
        np.testing.assert_array_equal(m,expected)

    def testIntercambiarFilaVariasVeces(self):
        m = np.array([[1,2,3],
                     [4,5,6],
                     [7,8,9]])
        expected = np.array([[7,8,9],
                            [1,2,3],
                            [4,5,6]])  

        intercambiarFilas(m,0,1)
        intercambiarFilas(m,0,2)
        np.testing.assert_array_equal(m,expected)


    def testSumarFilaMultiploMultiplo1(self):
        m = np.array([[1,2,3],
                     [2,1,2],
                     [3,3,3]])

        expected = np.array([[4,5,6],
                     [2,1,2],
                     [3,3,3]])

        sumarFilaMultiplo(m,0,2,1)

        np.testing.assert_array_equal(m,expected)

    
    def testSumarFilaMultiploMultiploPositivo(self):
        m = np.array([[1,2,3],
                     [2,1,2],
                     [3,3,3]])

        expected = np.array([[1,2,3],
                     [6,9,14],
                     [3,3,3]])

        sumarFilaMultiplo(m,1,0,4)

        np.testing.assert_array_equal(m,expected)

    def testSumarFilaMultiploMultiploNegativo(self):
        m = np.array([[1,2,3],
                     [2,1,2],
                     [3,3,3]])

        expected = np.array([[1,2,3],
                     [2,1,2],
                     [-1,1,-1]])

        sumarFilaMultiplo(m,2,1,-2)

        np.testing.assert_array_equal(m,expected)

    def testEsDominanteTrue(self):
        m = np.array([[5,2,1],
                      [3,7,1],
                      [0,0,1]])
        
        self.assertTrue(esDiagonalmenteDominante(m))

    def testEsDominanteFalse(self):
        m = np.array([[5,2,1],
                      [3,2,1],
                      [0,0,1]])
        
        self.assertFalse(esDiagonalmenteDominante(m))

    def testMatrizCirculanteVector2(self):
        v = np.array([1,2])
        expected = np.array([[1,2],[2,1]])
        np.testing.assert_array_equal(matrizCirculante(v),expected)


    def testMatrizCirculanteVector4(self):
        v = np.array([0,2,1,5])
        expected = np.array([[0,2,1,5],
                             [5,0,2,1],
                             [1,5,0,2],
                             [2,1,5,0]])
        np.testing.assert_array_equal(matrizCirculante(v),expected)

    def testVandermondeVector2(self):
        v = np.array([2,3])
        expected = np.array([[1,1],
                             [2,3]])
        np.testing.assert_array_equal(matrizVandermonde(v),expected)

    def testVandermondeVector3(self):
        v = np.array([2,3,4])
        expected = np.array([[1,1,1],
                             [2,3,4],
                             [4,9,16]])
        np.testing.assert_array_equal(matrizVandermonde(v),expected)

    def testHilbertN2(self):
        expected = np.array([[1,1/2],
                             [1/2,1/3]])
        
        np.testing.assert_array_almost_equal(matrizHilbert(2),expected)
    
    def testHilbertN3(self):
        expected = np.array([[1,1/2,1/3],
                             [1/2,1/3,1/4],
                             [1/3,1/4,1/5]])
        np.testing.assert_array_almost_equal(matrizHilbert(3),expected)




if __name__ == '__main__':
    unittest.main()