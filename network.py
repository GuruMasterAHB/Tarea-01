import random

import numpy as np

class Network(object): # clase para representar una RNA

    def __init__(self, sizes):
        self.num_layers = len(sizes) # número de capas que tiene la red
        self.sizes = sizes # lista (ejemplo: [3,7,2,6,...]) con el número de neuronas que tiene cada capa (que es cada índice de la lista)

        ''' Se crea una lista de matrices, cada una de dimensión y*1
            - Empieza en 1 porque la primera capa no necesita bias
            - self.bias[0] es una matriz de y*1 asociada a la capa 1
            - self.bias[1] es una matriz de y*1 asociada a la capa 2'''
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        
        ''' Lista de matrices de pesos entre las neuronas de la capa actual y la capa anterior
            zip(sizes[:-1],sizes[1:])
                - sizes[:-1] una lista sin la última capa
                - sizes[1:] una lista sin la primera capa
            '''
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    # Toma un valor de entrada y aplica la función sigmoide a cada una de las capas de la red
    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b) 
        return a

    # Stochastic gradient descent
    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        
        # training_data tiene dos parámetros (imagen, etiqueta) manipulados en mnist_loader
        training_data = list(training_data) # lista de tuplas [(imagen_0,etiqueta_0),(imagen_1,etiqueta_1),(imagen_2,etiqueta_2)...]
        n = len(training_data) # ¿cuántas tuplas hay?

        if test_data: # si hay un set de datos de prueba entonces:
            test_data = list(test_data) # vuelve los datos de prueba una lista
            n_test = len(test_data) # ¿cuántos datos de prueba hay?

        for j in range(epochs): # se da un número de épocas
            random.shuffle(training_data) # para cada época, los datos de entrenamiento se mezclan aleatoriamente
            mini_batches = [
                training_data[k:k+mini_batch_size] # se crean mini-batches del tamaño igual al argumento asociado en SGD(..., mini_batch_size,...)
                for k in range(0, n, mini_batch_size)] # va de 0 hasta la última tupla y el tamaño del paso es mini_batch_zise
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {} : {} / {}".format(j,self.evaluate(test_data),n_test))
            else:
                print("Epoch {} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y) # para cada tupla del mini_batch aplica backpropagation 
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)] # suma todos los gradientes de los buas
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)] # suma todos los gradientes de los pesos
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activations = [x] 
        zs = [] 
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b # z = (w • x) + b
            zs.append(z) # guarda cada z en la lista zs
            activation = sigmoid(z) # activación σ(z)
            activations.append(activation) # guarda las activaciones σ(z)

        # Calcula el error de la útlima capa
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta # gradiente del bias de la capa de salida
        nabla_w[-1] = np.dot(delta, activations[-2].transpose()) # gradiente del peso de la capa de salida


        for l in range(2, self.num_layers): # itera de las capas ocultas, desde la 2 hasta la penúltima capa
            z = zs[-l] # suma de todas las entradas de la neurona
            sp = sigmoid_prime(z) # llama a la derivada de la función sigmoide
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp # calcula el error de las capas yendo hacia atrás
            nabla_b[-l] = delta # se guarda el error de cada capa
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        return (output_activations-y)

# función sigmoide para activación
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

# derivada de la función sigmoide: dσ/dz
def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))