import unittest
import pytest
import numpy as np
from operacionesMatricesVectores import *

class TestOperacionesMatematicas(unittest.TestCase):

# Pruebas para InversoAditivo
    
    def test_inverso_aditivo_vector(self):
        vector = [(1+2j), (3+4j), (5+6j)]
        inverso_esperado = [(-1-2j), (-3-4j), (-5-6j)]
        inverso_obtenido = InversoAditivoVectorComplejo(vector)
        self.assertEqual(inverso_esperado, inverso_obtenido)

        vector = [(0+0j), (0+0j), (0+0j)]
        inverso_esperado = [(0+0j), (0+0j), (0+0j)]
        inverso_obtenido = InversoAditivoVectorComplejo(vector)
        self.assertEqual(inverso_esperado, inverso_obtenido)

# Pruebas para SumaVectoresComplejos

    def test_suma_vectores(self):
        vector1 = [(1+2j), (3+4j), (5+6j)]
        vector2 = [(7+8j), (9+10j), (11+12j)]
        suma_esperada = [(8+10j), (12+14j), (16+18j)]
        suma_obtenida = SumaVectoresComplejos(vector1, vector2)
        self.assertEqual(suma_esperada, suma_obtenida)

        vector1 = [(0+0j), (0+0j), (0+0j)]
        vector2 = [(1+1j), (1+1j), (1+1j)]
        suma_esperada = [(1+1j), (1+1j), (1+1j)]
        suma_obtenida = SumaVectoresComplejos(vector1, vector2)
        self.assertEqual(suma_esperada, suma_obtenida)

# Pruebas para multiplicacion_escalar_vector

    def test_multiplicacion_escalar_vector(self):
        escalar = 3
        vector = [(1+2j), (3+4j), (5+6j)]
        resultado_esperado = [(3+6j), (9+12j), (15+18j)]
        resultado_obtenido = MultiEscalarComplejo(escalar, vector)
        self.assertEqual(resultado_esperado, resultado_obtenido)

        escalar = 0
        vector = [(1+2j), (3+4j), (5+6j)]
        resultado_esperado = [(0+0j), (0+0j), (0+0j)]
        resultado_obtenido = MultiEscalarComplejo(escalar, vector)
        self.assertEqual(resultado_esperado, resultado_obtenido)

# Pruebas para SumaMatricesComplejas

    def test_suma_matrices_complejas(self):
        # Caso positivo
        matriz1 = [[(1, 2), (3, 4)], [(5, 6), (7, 8)]]
        matriz2 = [[(8, 7), (6, 5)], [(4, 3), (2, 1)]]
        resultado_esperado = [[(9, 9), (9, 9)], [(9, 9), (9, 9)]]
        self.assertEqual(SumaMatricesComplejas(matriz1, matriz2), resultado_esperado)

        # Caso negativo
        matriz1 = [[(1, 2), (3, 4)], [(5, 6), (7, 8)]]
        matriz2 = [[(8, 7), (6, 5)]]
        with self.assertRaises(ValueError):
            SumaMatricesComplejas(matriz1, matriz2)

# Pruebas para InversoAditivoMatriz

    def test_inverso_aditivo_matriz_compleja(self):
        # Caso positivo
        matriz = [[(1, 2), (3, 4)], [(5, 6), (7, 8)]]
        resultado_esperado = [[(-1, -2), (-3, -4)], [(-5, -6), (-7, -8)]]
        self.assertEqual(InversoAditivoMatrizCompleja(matriz), resultado_esperado)

        # Caso negativo
        matriz = [[(1, 2), (3, 4)], [(5, 6), (7, 8)]]
        resultado_esperado = [[(-1, -2), (-3, -4)], [(-5, -6), (-7, -9)]]
        with self.assertRaises(AssertionError):
            self.assertEqual(InversoAditivoMatrizCompleja(matriz), resultado_esperado)

# Pruebas para Transpuesta Matriz

    def test_transpose_complex_matrix(self):
        # Caso positivo
        matriz = [[(1, 2), (3, 4)], [(5, 6), (7, 8)]]
        resultado_esperado = [[(1, 2), (5, 6)], [(3, 4), (7, 8)]]
        self.assertEqual(transpose_complex_matrix(matriz), resultado_esperado)
    def test_conjugada_matriz_vector(self):
        matriz = [[1+1j, 2+2j], [3+3j, 4+4j]]
        vector = [1+1j, 2+2j]
        conjugada_matriz, conjugada_vector = ConjugadaMatrizVector(matriz, vector)
        self.assertEqual(conjugada_matriz, [[(1-1j), (2-2j)], [(3-3j), (4-4j)]])
        self.assertEqual(conjugada_vector, [(1-1j), (2-2j)])

# Pruebas para Multiplicar Matrices Complejas        
    def test_multiplicar_matrices_complejas(self):
        matriz1 = [[1+2j, 3+4j], [5+6j, 7+8j]]
        matriz2 = [[9+10j, 11+12j], [13+14j, 15+16j]]
        resultado = multiplicar_matrices_complejas(matriz1, matriz2)
        self.assertEqual(resultado, [[-197+404j, -241+508j], [-505+716j, -617+924j]])

# Pruebas para Accion de matriz sobre vector        
    def test_accion_matriz_vector(self):
        matriz = [[1+1j, 2+2j], [3+3j, 4+4j]]
        vector = [1+1j, 2+2j]
        resultado = accion_matriz_vector(matriz, vector)
        self.assertEqual(resultado, [(3+3j), (7+7j)])

# Pruebas para Producto Interno           
    def test_producto_interno(self):
        v1 = [1+1j, 2+2j]
        v2 = [3+3j, 4+4j]
        resultado = producto_interno(v1, v2)
        self.assertEqual(resultado, (22+0j))

# Pruebas para Norma Vector          
    def test_norma_vector(self):
        v = [1+1j, 2+2j]
        resultado = norma_vector(v)
        self.assertEqual(resultado, 3.1622776601683795)

# Pruebas para Distancia entre Vectores           
    def test_distancia_entre_vectores(self):
        v1 = [1+1j, 2+2j]
        v2 = [3+3j, 4+4j]
        resultado = distancia_entre_vectores(v1, v2)
        self.assertEqual(resultado, 3.1622776601683795)

# Pruebas para Valores y vectores propios           
    def test_valores_vectores_propios(self):
        matriz = [[1+0j, 2+0j], [3+0j, 4+0j]]
        valores, vectores = valores_vectores_propios(matriz)
        self.assertTrue(np.allclose(valores, [-0.37228132+0.j, 5.37228132+0.j]))
        self.assertTrue(np.allclose(vectores, [[-0.82456484, -0.41597356], [ 0.56576746, -0.90937671]]))

# Pruebas para Matriz Unitaria    
    def test_es_matriz_unitaria(self):
        # Caso de prueba 1: matriz unitaria
        matriz1 = [[1, 0], [0, 1]]
        self.assertTrue(es_matriz_unitaria(matriz1))
        
        # Caso de prueba 2: matriz no unitaria
        matriz2 = [[1, 1], [1, 1]]
        self.assertFalse(es_matriz_unitaria(matriz2))

# Pruebas para Matriz Hermitiana          
    def test_es_matriz_hermitiana(self):
        # Caso de prueba 1: matriz hermitiana
        matriz1 = [[1+0j, 2-3j], [2+3j, 4+0j]]
        self.assertTrue(es_matriz_hermitiana(matriz1))
        
        # Caso de prueba 2: matriz no hermitiana
        matriz2 = [[1, 2+1j], [2-1j, 4]]
        self.assertFalse(es_matriz_hermitiana(matriz2))
        
# Pruebas para Producto tensorial de matrices      
    def test_producto_tensor_matrices(self):
        # Caso de prueba 1: matrices de 2x2
        matriz1 = [[1, 2], [3, 4]]
        matriz2 = [[5, 6], [7, 8]]
        resultado1 = [[5, 6, 10, 12], [7, 8, 14, 16], [15, 18, 20, 24], [21, 24, 28, 32]]
        self.assertEqual(producto_tensor(matriz1, matriz2), resultado1)

# Pruebas para Producto tensorial de vectores      
    def test_producto_tensor_vectores(self):
        vector1 = [1, 2, 3]
        vector2 = [4, 5, 6]
        
        resultado_esperado = [4, 5, 6, 8, 10, 12, 12, 15, 18]
        
        self.assertEqual(producto_tensor_vectores(vector1, vector2), resultado_esperado)

if __name__ == '__main__':
    unittest.main()