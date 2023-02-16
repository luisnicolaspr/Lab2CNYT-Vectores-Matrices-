import numpy as np

#INVERSO ADITIVO DE UN VECTOR COMPLEJO

def InversoAditivoVectorComplejo(vector):

    inverso = []
    
    for element in vector:
        inverso.append(-1*element)
    
    return inverso

#SUMA DE VECTORES COMPLEJOS

def SumaVectoresComplejos(vector1,vector2):
    
    resultadoSuma = []

    for i in range (len(vector1)):
        resultadoSuma.append(vector1[i] + vector2[i])
        return resultadoSuma

#MULTIPLICACIÓN ESCALAR POR VECTOR COMPLEJO

def MultiEscalarComplejo (escalar,vector):
    resultadoMultiplicacion = []

    for nComplejo in vector:
        resultadoMultiplicacion.append(complex(escalar*nComplejo.real, escalar*nComplejo.imag))
        return resultadoMultiplicacion

#SUMA DE DOS MATRICES COMPLEJAS

def SumaMatricesComplejas (matriz1,matriz2):
    filas = len(matriz1)
    columnas = len(matriz1[0])
    resultado = [[0 for j in range(columnas)] for i in range(filas)]

    for i in range(filas):
        for j in range(columnas):
            real = matriz1[i][j][0] + matriz2[i][j][0]
            imag = matriz1[i][j][1] + matriz2[i][j][1]
            resultado[i][j] = (real,imag)
        return resultado

#INVERSA ADITIVA DE UNA MATRIZ COMPLEJA

def InversoAditivoMatrizCompleja (matriz):
    filas = len(matriz)
    columnas = len(matriz[0])
    resultado = [[0 for j in range(columnas)] for i in range(filas)]

    for i in range(filas):
        for j in range(columnas):
            real = -matriz[i][j][0]
            imag = -matriz[i][j][1]
            resultado[i][j] = (real,imag)
        return resultado

#TRANSPUESTA MATRIZ VECTOR

def transpose_complex_matrix(matrix):
    rows = len(matrix)
    cols = len(matrix[0])
    result = [[0 for j in range(rows)] for i in range(cols)]
    
    for i in range(rows):
        for j in range(cols):
            result[j][i] = matrix[i][j]
    
    return result

def transpose_complex_vector(vector):
    return [vector[i] for i in range(len(vector))]

# Ejemplo de uso de la transpuesta de una matriz
matrix = [[(1, 2), (3, 4)], [(5, 6), (7, 8)]]
result = transpose_complex_matrix(matrix)

for row in result:
    print(row)

# Ejemplo de uso de la transpuesta de un vector
vector = [(1, 2), (3, 4), (5, 6)]
result = transpose_complex_vector(vector)

print(result)


#CONJUGADA DE MATRIZ Y VECTOR

def ConjugadaMatrizVector(matriz, vector):
    conjugada_matriz = [[complex(x.real, -x.imag) for x in row] for row in matriz]
    conjugada_vector = [complex(x.real, -x.imag) for x in vector]
    return (conjugada_matriz, conjugada_vector)


#ADJUNTA DE UN VECTOR Y DE UNA MATRIZ

adj_matrix = np.conj(matrix)
adj_vector = np.conj(vector) 

#MULTIPLICACIÓN DE DOS MATRICES(MISMO TAMAÑO)

def multiplicar_matrices_complejas(matriz1, matriz2):
    """
    Multiplica dos matrices de números complejos con el mismo tamaño y devuelve el resultado.
    """
    filas_matriz1 = len(matriz1)
    columnas_matriz1 = len(matriz1[0])
    filas_matriz2 = len(matriz2)
    columnas_matriz2 = len(matriz2[0])

    if columnas_matriz1 != filas_matriz2:
        raise ValueError("Las matrices no tienen el mismo tamaño.")

    # Creamos una matriz de ceros para almacenar el resultado
    resultado = [[0 + 0j] * columnas_matriz2 for _ in range(filas_matriz1)]

    # Multiplicamos las matrices
    for i in range(filas_matriz1):
        for j in range(columnas_matriz2):
            for k in range(filas_matriz2):
                resultado[i][j] += matriz1[i][k] * matriz2[k][j]

    return resultado

#ACCION DE UNA MATRIZ SOBRE UN VECTOR

def accion_matriz_vector(matriz, vector):
    """
    Calcula la acción de una matriz sobre un vector y devuelve el vector resultante.
    """
    filas_matriz = len(matriz)
    columnas_matriz = len(matriz[0])
    longitud_vector = len(vector)

    if columnas_matriz != longitud_vector:
        raise ValueError("La matriz y el vector no tienen dimensiones compatibles.")

    # Creamos un vector de ceros para almacenar el resultado
    resultado = [0 + 0j] * filas_matriz

    # Multiplicamos la matriz por el vector
    for i in range(filas_matriz):
        for j in range(columnas_matriz):
            resultado[i] += matriz[i][j] * vector[j]

    return resultado

#PRODUCTO INTERNO DE DOS VECTORES

def producto_interno(v1, v2):
    """
    Calcula el producto interno de dos vectores y devuelve el resultado.
    """
    if len(v1) != len(v2):
        raise ValueError("Los vectores no tienen la misma dimensión.")
    
    resultado = sum([v1[i] * v2[i].conjugate() for i in range(len(v1))])
    
    return resultado

#NORMA DE UN VECTOR

def norma_vector(v):
    """
    Calcula la norma de un vector y devuelve el resultado.
    """
    resultado = (sum([abs(v[i])**2 for i in range(len(v))]))**(1/2)
    
    return resultado

#DISTANCIA ENTRE DOS VECTORES

def distancia_entre_vectores(v1, v2):
    """
    Calcula la distancia entre dos vectores y devuelve el resultado.
    """
    if len(v1) != len(v2):
        raise ValueError("Los vectores no tienen la misma dimensión.")
    
    diferencia = [v1[i] - v2[i] for i in range(len(v1))]
    resultado = norma_vector(diferencia)
    
    return resultado

#VALORES Y VECTORES PROPIOS 

import numpy as np

def valores_vectores_propios(matriz):
    """
    Calcula los valores y vectores propios de una matriz y devuelve los resultados.
    """
    valores_propios, vectores_propios = np.linalg.eig(matriz)
    
    return valores_propios, vectores_propios

#VERIFICACIÓN DE MATRIZ UNITARIA

def es_matriz_unitaria(matriz):
    """
    Verifica si una matriz es unitaria y devuelve True o False.
    """
    adjunta = [[matriz[j][i].conjugate() for j in range(len(matriz))] for i in range(len(matriz[0]))]
    producto = multiplicar_matrices_complejas(matriz, adjunta)
    identidad = [[1 + 0j if i == j else 0
    for i in range(len(matriz))] for j in range(len(matriz[0]))]

    if es_matriz_unitaria(producto):
            return True
    else:
        return False

#MATRIZ HERMITIANA

def es_matriz_hermitiana(matriz):
    """
    Verifica si una matriz es Hermitiana y devuelve True o False.
    """
    # Calcula la matriz adjunta de la matriz original
    adjunta = [[matriz[j][i].conjugate() for j in range(len(matriz))] for i in range(len(matriz[0]))]
    
    # Verifica si la matriz original es igual a su matriz adjunta
    if matriz == adjunta:
        return True
    else:
        return False

#PRODUCTO TENSORIAL MATRIZ

import numpy as np

def producto_tensor(matriz1, matriz2):
    """
    Calcula el producto tensor de dos matrices y devuelve el resultado.
    """
    resultado = np.kron(matriz1, matriz2)
    
    return resultado

#PRODUCTO TENSORIAL VECTOR

def producto_tensor_vectores(vector1, vector2):
    """
    Calcula el producto tensor de dos vectores y devuelve el resultado.
    """
    resultado = [vector1[i] * vector2[j] for i in range(len(vector1)) for j in range(len(vector2))]
    
    return resultado


