# May 2025 --> Supervised ANN that recognises the MNIST number dataset by Jorge Barroso García
import time
import random
import matplotlib.pyplot as plt
import numpy as np
import os

# GLOBAL VARIABLES 
network_structure = [784,16,16,10]
learning_rate = 0.1
dynamic_learning_rate = True
stochastic = True
epochs = 3
activation_function = "sigmoid" #sigmoid/relu
# Si save_name == "" no se guarda el modelo
# Si save_name == "descriptive" el nombre del modelo es la info de entrenamiento del modelo
save_name = '' 
# Si model_to_load_name == "" no se carga el modelo
model_to_load_name = 'Stochastic_LR_0.1_DynamicLR_Act_sigmoid_Epochs10'
train_model = False
visualize_results = True


#region Tests (no testing)

#PASSED
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
#region Visualizacion

def predict_single(layers, data_vec):
    x = np.array(data_vec, dtype=float)
    out = forward_propagation(x, layers).flatten()  # (10,)
    pred = int(np.argmax(out))
    conf = float(out[pred])
    return out, pred, conf

class OneByOneViewer:
    def __init__(self, layers, dataset):
        self.layers = layers
        self.dataset = dataset
        self.idx = 0

        # Figura con dos paneles: imagen y barras de salida
        self.fig, (self.ax_img, self.ax_bar) = plt.subplots(1, 2, figsize=(9, 4.8))
        self.fig.canvas.manager.set_window_title("MNIST viewer")
        plt.subplots_adjust(wspace=0.35)

        # Preparar barra
        self.bars = self.ax_bar.bar(range(10), np.zeros(10))
        self.ax_bar.set_xticks(range(10))
        self.ax_bar.set_ylim(0, 1)   # salidas en [0,1] con Sigmoid
        self.ax_bar.set_xlabel("Clase")
        self.ax_bar.set_ylabel("Salida (neurona)")
        self.pred_text = self.ax_bar.text(0.5, 1.02, "", transform=self.ax_bar.transAxes,
                                          ha="center", va="bottom", fontsize=11)

        # Conectar eventos
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)

        self.draw()
        plt.show()

    def clamp_index(self):
        n = len(self.dataset)
        if self.idx < 0: self.idx = 0
        if self.idx >= n: self.idx = n - 1

    def draw(self):
        self.clamp_index()
        label, data = self.dataset[self.idx]
        out, pred, conf = predict_single(self.layers, data)

        # Imagen
        img = np.array(data, dtype=float).reshape(28, 28)
        self.ax_img.clear()
        self.ax_img.imshow(img, cmap='gray')
        self.ax_img.axis('off')
        self.ax_img.set_title(f"Index={self.idx}  |  y={label} |  Success={('T' if label==pred else 'F')}" )

        # Barras de salida (última capa)
        for i, b in enumerate(self.bars):
            b.set_height(out[i])
            b.set_color('C0')  # reset color
        # resalta la predicción
        self.bars[pred].set_color('C3')

        # texto arriba del gráfico
        self.pred_text.set_text(f"pred={pred}  conf={conf:.3f}")

        # título general con ayuda
        self.fig.suptitle("Flechas para moverse",
                          fontsize=10)
        self.fig.canvas.draw_idle()

    def next(self):
        self.idx += 1
        self.draw()

    def prev(self):
        self.idx -= 1
        self.draw()

    def on_key(self, event):
        if event.key in ('right', ' '):   # flecha derecha o espacio
            self.next()
        elif event.key in ('left', 'backspace'):
            self.prev()

    def on_scroll(self, event):
        if event.button == 'up':
            self.next()
        elif event.button == 'down':
            self.prev()

#endregion
#region Red

class Layer:
    """ Clase que representa una capa de la red neuronal. Cada capa tiene
        un número de inputs y un número de neurons. Los weights y biases
        de cada capa se inicializan aleatoriamente."""
    def __init__(self, n_input, n_neurons, activation=None):
        # Inicializar los weights y biases
        self.input_layer = (n_input == 0)
        if activation==None:
            self.activation = activation_function
        else:
            self.activation = activation
        if not self.input_layer:
            
            self.weights = np.random.randn(n_input, n_neurons) * np.sqrt(2.0 / n_input)
            # Recordatorio J: Inicializacion pequeña y positiva: Cambiada para ayudar a ReLU a no morirse
            # self.weights = 0.01 * np.random.rand(n_input, n_neurons)
            self.biases = np.zeros((1, n_neurons))
        else:
            self.weights = None
            self.biases = None

    def forward(self, inputs):
        """ Introduce los inputs a la capa y devuelve el resultado de la
            función de activación ReLU."""
        if not self.input_layer: 
            self.z = np.dot(inputs, self.weights) + self.biases
            if self.activation == "relu":
                self.output = ReLU(self.z)
            elif self.activation == "sigmoid":
                self.output = Sigmoid(self.z)
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

def backpropagation(layers,output,expected_output):
    """ Obtiene y devuelve el vector gradiente de la red neuronal para
        cada capa haciendo las derivadas parciales de la función de coste.
        Como función de coste se usa MSE: la media de los errores al cuadrado."""
    
    deltas = []
    # Calcular los deltas o errores de cada capa
    for i in range(len(layers)-1,0,-1):
        activation = layers[i].activation
        if i == len(layers)-1:
            # Output 
            dError = 2 * (output-expected_output)
            if activation == "relu":
                dOutput = dRelu(layers[i].z)
            elif activation == "sigmoid":
                dOutput = dSigmoid(layers[i].z)
            deltas.insert(0,dError*dOutput)
        else:
            # Hidden layers
            dError = np.dot(deltas[0],layers[i+1].weights.T)
            if activation == "relu":
                dOutput = dRelu(layers[i].z)
            elif activation == "sigmoid":
                dOutput = dSigmoid(layers[i].z)
            deltas.insert(0,dError*dOutput)
            
            
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

                # Backpropagation
                # Recordatorio Jorge: Se pasa el output y el expected_output en lugar del error para mantener el signo en la derivada del backprop
                gradient_vectors = backpropagation(layers,output,expected_output)
                if len(total_gradient_vectors) == 0:
                    total_gradient_vectors = gradient_vectors
                else:
                    for i in range(len(total_gradient_vectors[0])):
                        total_gradient_vectors[0][i] += gradient_vectors[0][i]
                        total_gradient_vectors[1][i] += gradient_vectors[1][i]
                    
            print(f"Error: {total_error}")
            print(f"Epoch {epoch+1}/{epochs} finished - Avg Error: {total_error/len(dataset)}")  
            
            # Dynamic learning rate
            if dynamic_learning_rate:
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
        for epoch in range(epochs):
            total_error_epoch = 0.0
            for index, data_piece in enumerate(dataset):
                label, data = data_piece
                data = np.array(data, dtype=float)

                # Forward propagation
                output = forward_propagation(data, layers)

                # Error calculation
                expected_output = get_expected_output(label)
                err = get_error(output, expected_output)
                total_error_epoch += err

                # Backprop + gradient descent
                gradient_vectors = backpropagation(layers, output, expected_output)
                gradient_descent(gradient_vectors, learning_rate, layers)

                # Print cada 1000 datapieces
                if (index + 1) % 1000 == 0:
                    avg_so_far = total_error_epoch / (index + 1)
                    print(f"Epoch {epoch+1}/{epochs}  |  {index+1}/{len(dataset)}  |  Avg Error so far: {avg_so_far:.4f}")

            # Dynamic learning rate
            if dynamic_learning_rate:
                if epoch == epochs//2:
                    learning_rate = learning_rate/2
                if epoch == epochs//4:
                    learning_rate = learning_rate/2
                    
            # Fin de Epoch
            avg_epoch = total_error_epoch / len(dataset)
            print(f"Epoch {epoch+1}/{epochs} finished - Avg Error: {avg_epoch:.4f}")

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
        if np.argmax(output) == label:
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
        - Si model_name == "descriptive" el nombre del modelo es la info del modelo
        - La primera línea describe el training del modelo (estocastico/LR/DynamicLR/epochs entrenados)
        - La segunda linea da el Activation Function
        - La tercera línea específica la estructura de capas.
        Las siguientes líneas son los weights y biases de cada capa. Ejemplo:
        784,16,16,10
        W1: W00,W01,W10,W11
        B1: B0,B1
        W2: ...
        """
    
    if model_name == 'descriptive':
        # construir nombre con info clave
        model_name = ""
        model_name += "Stochastic_" if stochastic else "Batch_"
        model_name += f"LR_{learning_rate}_"
        model_name += "DynamicLR_" if dynamic_learning_rate else ""
        model_name += f"Act_{activation_function}_"
        model_name += f"Epochs{epochs}"


    with open(f"{model_name}.txt", "w") as file:

        # Escribir la información de entrenamiento del modelo en la primera línea
        training_info = (
            f"Info de entrenamiento del modelo: "
            f"Stochastic training: {stochastic}, "
            f"Learning Rate: {learning_rate}, "
            f"Dynamic LR: {True}, "
            f"Epochs: {epochs}"
        )
        file.write(training_info + "\n")
        file.write(f"Activation Function:{activation_function}\n")
        # Escribir la estructura de capas en la segunda línea
        layers_structure = ",".join(str(n) for n in network_structure)
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
    """ Crea una red neuronal con la estructura y los weights y biases y activation
        especificados en el archivo de texto.
        """
    layers = []
    with open(f"{model_name}.txt", "r") as file:
        # Ignorar la info de entrenamiento:
        file.readline()
        activation_f = file.readline().split(":")[1].strip()
        # Leer la estructura de capas
        layers_structure = file.readline().strip().split(",")
        
        # Crear las capas de la red neuronal
        for i in range(len(layers_structure)-2):
            layers.append(Layer(int(layers_structure[i]), int(layers_structure[i+1]),activation=activation_f))
        # La ultima capa siempre sigmoid
        layers.append(Layer(int(layers_structure[len(layers_structure)-2]), int(layers_structure[len(layers_structure)-1]),activation="sigmoid"))

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
    return np.maximum(0,x)

def dRelu(x):
    return np.where(x <= 0, 0, 1)

def Sigmoid(x):
    x = np.clip(x, -40, 40) #Clip para evitar overflow en ReLU
    return 1.0/(1.0+np.exp(-x))

def dSigmoid(x):
    s = Sigmoid(x)
    return s*(1.0-s)

def get_error(output,expected_output):
    """ Devuelve la suma de los errores cuadráticos entre el array output de la red neuronal
        y el array output esperado."""
    return np.sum((output-expected_output)**2)

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

def normalize(dataset):
    """ Normaliza el dataset para que cada dato este contenido entre 0 y 1"""
    for d,datapiece in enumerate(dataset):
        for i,pixel in enumerate(datapiece[1]):
            dataset[d][1][i] = float(pixel)/255.0


def main():
    """ Obtiene el dataset MNIST, crea la red neuronal, la entrena y la
        testea. Muestra los resultados por pantalla. Guarda el modelo
        si se especifica en la variable global save_model"""
    
    # Obtener los datos
    start_time = time.time()
    print("Reading the MNIST set...")
    train_set, test_set = get_mnist_dataset()
    normalize(train_set)
    normalize(test_set)
    print("Data read and normalized. Time taken: " + str(time.time()-start_time))

    # Crear la red neuronal por capas o la obtiene de un archivo de texto
    layers = []
    if model_to_load_name == "":
        # Input layer
        layers.append(Layer(0,network_structure[0]))
        # Hidden layers and output layer
        for i in range(1,len(network_structure)-1):
            layers.append(Layer(network_structure[i-1],network_structure[i]))
        # La ultima capa siempre sigmoid
        layers.append(Layer(network_structure[len(network_structure)-2],network_structure[len(network_structure)-1],activation="sigmoid"))
    else:
        print(f"Model {model_to_load_name} loaded")
        layers = load(model_to_load_name)
    
    if train_model:
        # Entrenar la red neuronal
        start_time = time.time()
        print("Training the neural network...")
        train(epochs,learning_rate,layers,train_set)
        print("Training finished. Time taken: " + str(time.time()-start_time))

    # Testear la red neuronal
    start_time  = time.time()
    print("Testing the neural network...")
    test(layers,test_set)
    print("Testing finished. Time taken: " + str(time.time()-start_time))

    # Guardar el modelo
    if save_name != "":
        start_time = time.time()
        print("Saving the model")
        save(layers,save_name)
        print("Model saved. Time taken: " + str(time.time()-start_time))

    if visualize_results:
        OneByOneViewer(layers, test_set)

main()

#endregion

