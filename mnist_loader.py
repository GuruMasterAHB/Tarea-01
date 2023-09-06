import pickle
import gzip

import numpy as np

# Esta función se encargará de importar los datos de MNIST que vienen comprimidos
# los desencapsula y les asigna variables a los tres sets que se tienen
def load_data(): 
    f = gzip.open('mnist.pkl.gz', 'rb') # Abre el archivo
    # Los datos se separan en tres partes: trainig_data, validation_data y test_data
    # Se usa latin1 como encoder para que funciones con python 3.x
    training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    f.close()
    return (training_data, validation_data, test_data) # devuelve cada variable con sus datos asociados al paquete

def load_data_wrapper():
    tr_d, va_d, te_d = load_data() # Llama a la función anterior y renombra las variables
    # tr_d[0] corresponde al conjunto de imágenes con las que se va a entrenar
    # tr_d[1] es la etiqueta de cada imagen
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]] # Transforma la matriz de la imagen de 28*28 a una matriz de 784*1
    training_results = [vectorized_result(y) for y in tr_d[1]] # codifica y como one-hot encoding para cada elemento de tr_d[1]
    training_data = zip(training_inputs, training_results) # enpaqueta ambas matrices juntas

    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]] # transforma la imagen en una matriz de 784*1
    # no termino de entender por qué no tiene etiqueta asociada
    # no tiebe una matriz one-hot asociada a validation_results
    validation_data = zip(validation_inputs, va_d[1]) 

    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]] # transorma la imagen en una matriz de 84*1
    test_data = zip(test_inputs, te_d[1]) # no tiene matriz one-hot asociada porque son datos de test
    return (training_data, validation_data, test_data)

# Matriz one-hot
def vectorized_result(j):
    e = np.zeros((10, 1)) # vector de 10*1 lleno de ceros
    e[j] = 1.0 # vuelve el cero de la posición j en un 1.0
    return e