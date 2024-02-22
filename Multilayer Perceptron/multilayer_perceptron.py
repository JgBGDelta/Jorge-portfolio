from cmath import sqrt
import time
import numpy as np
import pandas as pd
import os

# Global variables
n_neurons_per_layer = [784,16,16,10]
learning_rate = 0.5
epochs = 50
activation_function = "Sigmoid"
save_name = 'testing_model'
model_to_load_name = "testing_model"
stochastic = False

# Supervised artificial nueural network that recognises the MNIST num dataset by Jorge Barroso García

#region Tests

def forward_prop_test():
    """ Passed -> Forward propagation working as expected."""
    layers = []
    layers.append(Layer(0,2))
    layers.append(Layer(2,2))
    layers.append(Layer(2,1))
    # (input_neurons, output_neurons)
    layers[1].weights = np.zeros((2, 2))
    layers[1].biases = np.zeros((1, 2))
    layers[2].weights = np.zeros((2, 1))
    layers[2].biases = np.zeros((1, 1))

    layers[1].weights[0][0] = 1
    layers[1].weights[0][1] = 2
    layers[1].weights[1][0] = 3
    layers[1].weights[1][1] = 4

    layers[1].biases[0][0] = 1
    layers[1].biases[0][1] = 2

    layers[2].weights[0][0] = 1
    layers[2].weights[1][0] = 2
    
    layers[2].biases[0][0] = 1


    print(layers[1].weights)
    print(layers[1].biases)
    print(layers[2].weights)
    print(layers[2].biases)

    data = np.array([1,1])
    print(forward_propagation(data,layers))

# endregion

class Layer:
    """ Clase que representa una capa de la red neuronal. Cada capa tiene
        un número de inputs y un número de neurons. Los weights y biases
        de cada capa se inicializan aleatoriamente."""
    def __init__(self, n_input, n_neurons):
        # Inicializar los weights y biases
        self.input_layer = (n_input == 0)

        if not self.input_layer:
            self.weights = 0.01 * np.random.rand(n_input, n_neurons)
            self.biases = np.zeros((1, n_neurons))
        else:
            self.weights = None
            self.biases = None

    def forward(self, inputs):
        """ Introduce los inputs a la capa y devuelve el resultado de la
            función de activación ReLU."""
        if not self.input_layer: 
            self.output = np.dot(inputs, self.weights) + self.biases
            if activation_function == "ReLU":
                self.output = ReLU(self.output)
            elif activation_function == "Sigmoid":
                self.output = Sigmoid(self.output)
            
        else:
            self.output = np.array([inputs])

def forward_propagation(data_piece,layers):
    """ Introduce la imagen dada a la red neuronal y devuelve los resultados
        de la última capa de la red"""
    layers[0].forward(data_piece)
    for i in range(1,len(layers)):
        layers[i].forward(layers[i-1].output)
    return layers[-1].output


def gradient_descent(gradient_vectors,learning_rate,layers):
    """ Se aplican los vectores gradientes dados y se actualizan los weights
        y biases de cada capa."""
    gradient_vectors_weights,gradient_vectors_biases = gradient_vectors
    for i in range(1,len(layers)):
        layers[i].weights -= learning_rate * gradient_vectors_weights[i-1]
        layers[i].biases -= learning_rate * gradient_vectors_biases[i-1]

def backpropagation(layers,error_vector):
    """ Obtiene y devuelve el vector gradiente de la red neuronal para
        cada capa haciendo las derivadas parciales de la función de coste.
        Como función de coste se usa: la media de los errores al cuadrado."""
    
    deltas = []
    # Calcular los deltas o errores de cada capa
    for i in range(len(layers)-1,0,-1):
        if i == len(layers)-1:
            # Output 
            dError = 2* np.sqrt(error_vector)
            if activation_function == "ReLU":
                dOutput = dRelu(layers[i].output)
            elif activation_function == "Sigmoid":
                dOutput = dSigmoid(layers[i].output)
            deltas.insert(0,dError*dOutput)
        else:
            # Hidden layers
            wl1 = layers[i+1].weights
            deltaL1 = deltas[0].T
            if activation_function == "ReLU":
                dOutput = dRelu(layers[i].output)
            elif activation_function == "Sigmoid":
                dOutput = dSigmoid(layers[i].output)
            mat_mul = np.dot(wl1,deltaL1)
            deltas.insert(0,mat_mul.T*dOutput)
            
    # Calcular el vector gradiente de cada capa
    gradient_vectors_weights = []
    gradient_vectors_biases = []
    for i in range(len(layers)-1,0,-1):
        # Hidden layers and output layer
        deltaL = deltas[i-1].T
        aLm1 = layers[i-1].output
        gradient_weights = np.dot(deltaL,aLm1).T
        gradient_biases = deltaL.T
        gradient_vectors_weights.insert(0,gradient_weights)
        gradient_vectors_biases.insert(0,gradient_biases)

    return [gradient_vectors_weights,gradient_vectors_biases]

def train(epochs, learning_rate, layers, dataset):
    """ Se entrena la red neuronal con el dataset MNIST. Se hace un forward
        propagation con cada imagen del dataset y se calcula el vector
        gradiente con backpropagation. Se aplica el vector gradiente con
        gradient descent. Se repite epochs veces."""
    print(f"Training the neural network with {len(dataset)} images")
    # Total training -> Updates the W&B after the whole dataset
    if not stochastic:
        for epoch in range(epochs):
            total_gradient_vectors = []
            total_error = 0
            for index,data_piece in enumerate(dataset):
                label,data = data_piece
                data = np.array(data,dtype=float)
                # Forward propagation
                output = forward_propagation(data,layers)

                #Error calculation
                expected_output = get_expected_output(label)
                total_error += get_error(output,expected_output)
                total_error_vector = get_error_vector(output,expected_output)

                # Backpropagation
                gradient_vectors = backpropagation(layers,total_error_vector)
                if len(total_gradient_vectors) == 0:
                    total_gradient_vectors = gradient_vectors
                else:
                    for i in range(len(total_gradient_vectors[0])):
                        total_gradient_vectors[0][i] += gradient_vectors[0][i]
                        total_gradient_vectors[1][i] += gradient_vectors[1][i]
                    

            
            print(f"Error: {total_error}")
            print(f"Epoch {epoch}/{epochs} finished - Avg Error: {total_error/len(dataset)}")  
            
            if epoch == epochs//2:
                learning_rate = learning_rate/2
            if epoch == epochs//4:
                learning_rate = learning_rate/2
            # Average the gradient vectors
            for i in range(len(total_gradient_vectors[0])):
                total_gradient_vectors[0][i] = total_gradient_vectors[0][i]/len(dataset)
                total_gradient_vectors[1][i] = total_gradient_vectors[1][i]/len(dataset)
            # Gradient descent
            gradient_descent(total_gradient_vectors,learning_rate,layers)
            
    
    # Stochastic training -> Updates the W&B after each data piece
    else:
        for epoch in range(epochs//len(dataset)):
            for index,data_piece in enumerate(dataset):
                label,data = data_piece
                data = np.array(data,dtype=float)
                # Forward propagation
                output = forward_propagation(data,layers)

                #Error calculation
                expected_output = get_expected_output(label)
                total_error = get_error(output,expected_output)
                total_error_vector = get_error_vector(output,expected_output)

                # Backpropagation
                gradient_vectors = backpropagation(layers,total_error_vector)
                
                # Gradient descent
                gradient_descent(gradient_vectors,learning_rate,layers)
                if epoch ==0 and index == 0:
                    print(f"Error: {total_error/len(dataset)}")
                epoch_number = epoch*len(dataset)+index+1
                if epoch_number % 1000 == 0:
                    print(f"Epoch {epoch_number}/{epochs} - Error: {total_error/len(dataset)}")


def test(layers,testset):
    """ Se testea la red neuronal con el testset MNIST. Se hace un forward
        propagation con cada imagen del testset y se calcula el error
        de la red neuronal. Se devuelve el porcentaje de aciertos."""
    print(f"Testing the neural network with {len(testset)} images")
    correct = 0
    for data_piece in testset:
        label,data = data_piece
        data = np.array(data,dtype=float)
        output = forward_propagation(data,layers)
        if np.argmax(output) == label and output[0][np.argmax(output)] > 0.55:
            correct += 1
    print(f"Correct: {correct}/{len(testset)}")
    print(f"Percentage: {correct/len(testset)*100}%")

def get_expected_output(label):
    """ Devuelve un array de 10 elementos con todos los elementos a 0 excepto
        el elemento de la posición label que es 1. Ejemplo: label=3 -> [0,0,0,1,0,0,0,0,0,0]"""
    expected_output = np.zeros(10)
    expected_output[label] = 1
    return expected_output

def save(layers,model_name="model"):
    """ Crea un nuevo archivo de texto con los weights y biases de cada capa
        La primera línea específica la estructura de capas.
        Las siguientes líneas son los weights y biases de cada capa. Ejemplo:
        784,16,16,10
        W1: W00,W01,W10,W11
        B1: B0,B1
        W2: ...
        """
    
    with open(f"{model_name}.txt", "w") as file:
        # Escribir la estructura de capas en la primera línea
        layers_structure = ",".join(str(n) for n in n_neurons_per_layer)
        file.write(layers_structure + "\n")
        
        # Escribir los weights y biases de cada capa
        for layer in layers[1:]:
            # Escribir los weights
            weights_line = "W" + str(layers.index(layer) + 1) + ": "
            weights_line += ",".join(str(w) for w in layer.weights.flatten())
            file.write(weights_line + "\n")
            
            # Escribir los biases
            biases_line = "B" + str(layers.index(layer) + 1) + ": "
            biases_line += ",".join(str(b) for b in layer.biases.flatten())
            file.write(biases_line + "\n")
    
def load(model_name="model"):
    """ Crea una red neuronal con la estructura y los weights y biases
        especificados en el archivo de texto.
        """
    layers = []
    with open(f"{model_name}.txt", "r") as file:
        # Leer la estructura de capas
        layers_structure = file.readline().strip().split(",")
        
        # Crear las capas de la red neuronal
        for i in range(len(layers_structure)-1):
            layers.append(Layer(int(layers_structure[i]), int(layers_structure[i+1])))
        
        # Leer los weights y biases de cada capa
        for layer in layers:
            # Leer los weights
            weights_line = file.readline().strip().split(": ")[1].split(",")
            weights = np.array([float(w) for w in weights_line]).reshape(layer.weights.shape)
            layer.weights = weights
            
            # Leer los biases
            biases_line = file.readline().strip().split(": ")[1].split(",")
            biases = np.array([float(b) for b in biases_line]).reshape(layer.biases.shape)
            layer.biases = biases
    
    return layers

def ReLU(x):
    """ Función de activación ReLU. Devuelve 0 si x es negativo, x si es
        positivo."""
    return np.maximum(0,x)

def dRelu(x):
    """ Derivada de la función de activación ReLU. Devuelve 0 si x es negativo,
        1 si es positivo."""
    return np.where(x <= 0, 0, 1)

def Sigmoid(x):
    """ Función de activación Sigmoid. Devuelve el valor de la función
        sigmoide en x."""
    return 1/(1+np.exp(-x))

def dSigmoid(x):
    """ Derivada de la función de activación Sigmoid. Devuelve el valor de
        la derivada de la función sigmoide en x."""
    return Sigmoid(x)*(1-Sigmoid(x))

def get_error(output,expected_output):
    """ Devuelve la suma de los errores cuadráticos entre el array output de la red neuronal
        y el array output esperado."""
    return np.sum((output-expected_output)**2)

def get_error_vector(output,expected_output):
    """ Devuelve el vector resultado de aplicar el cuadrado de la diferencia
        entre el output y el expected_output."""
    return (output-expected_output)**2

def get_mnist_dataset():
    """ Devuelve el dataset MNIST. El dataset MNIST contiene 60000 imágenes
        de 28x28 píxeles (784 Input neurons) de números centrados, escritos
        a mano en blanco y negro del 1 al 10 """
    train_set = []
    test_set = []

    current_path = os.path.dirname(os.path.abspath(__file__))
    train_csv_path= os.path.join(current_path, 'mnist_num_train.csv')
    test_csv_path = os.path.join(current_path,'mnist_num_test.csv')

    with open(train_csv_path, 'r') as file:
        lines = file.readlines()
        for line in lines[1:]:
            train_set.append((int(line[0]),line.strip().split(',')[1:]))

    with open(test_csv_path, 'r') as file:
        lines = file.readlines()
        for line in lines[1:]:
            test_set.append((int(line[0]),line.strip().split(',')[1:]))

    return train_set,test_set

def main():
    """ Obtiene el dataset MNIST, crea la red neuronal, la entrena y la
        testea. Muestra los resultados por pantalla. Guarda el modelo
        si se especifica en la variable global save_model"""
    
    # Obtener los datos
    start_time = time.time()
    print("Reading the MNIST set")
    train_set, test_set = get_mnist_dataset()
    print("Data read. Time taken: " + str(time.time()-start_time))

    # Crear la red neuronal por capas o la obtiene de un archivo de texto
    layers = []
    if model_to_load_name == "":
        # Input layer
        layers.append(Layer(0,n_neurons_per_layer[0]))
        # Hidden layers and output layer
        for i in range(1,len(n_neurons_per_layer)):
            layers.append(Layer(n_neurons_per_layer[i-1],n_neurons_per_layer[i]))
    else:
        print(f"Model {model_to_load_name} loaded")
        layers = load(model_to_load_name)
    
    # Entrenar la red neuronal
    start_time = time.time()
    print("Training the neural network")
    train(epochs,learning_rate,layers,train_set)
    print("Training finished. Time taken: " + str(time.time()-start_time))

    # Testear la red neuronal
    start_time  = time.time()
    print("Testing the neural network")
    test(layers,test_set)
    print("Testing finished. Time taken: " + str(time.time()-start_time))

    # Guardar el modelo
    if save_name != "":
        start_time = time.time()
        print("Saving the model")
        save(layers,save_name)
        print("Model saved. Time taken: " + str(time.time()-start_time))

main()
