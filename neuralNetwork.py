import numpy as np
import layers
from matplotlib import pyplot as plt
np.set_printoptions(suppress=True)


class NeuralNetwork:

    def __init__(self, input: layers.InputLayer, output: layers.OutputLayer):
        self.expectedOutputs = None
        self.inputData = None
        self.inputLayer = input
        self.outputLayer = output

    # FORWARD PROPAGATION
    def forwardPropagate(self, layer: layers.Layer):
        if type(layer.previousLayer) is layers.InputLayer:
            layer.results()
            return 1
        if self.forwardPropagate(layer.previousLayer) == 1:
            layer.results()
            return 1
        else:
            self.forwardPropagate(layer.previousLayer)

    # BACKWARD PROPAGATION
    def backwardPropagate(self, expected, l_rate):
        currentLayer = self.outputLayer

        # calculating output layer error
        currentLayer.error = currentLayer.values - expected

        # updating output layer weights
        currentLayer.weights += -l_rate * np.dot(currentLayer.error, currentLayer.previousLayer.values.T)
        currentLayer.bias += -l_rate * currentLayer.error

        if type(currentLayer.previousLayer) is not layers.InputLayer:
            currentLayer.previousLayer.error = np.dot(currentLayer.weights.T, currentLayer.error) * (currentLayer.previousLayer.values * (1 - currentLayer.previousLayer.values))
            # currentLayer.previousLayer.error = np.dot(currentLayer.weights.T, currentLayer.error)
            # update weights
            currentLayer.previousLayer.weights += -l_rate * np.dot(currentLayer.previousLayer.error, currentLayer.previousLayer.previousLayer.values.T)
            currentLayer.previousLayer.bias += -l_rate * currentLayer.previousLayer.error

            currentLayer = currentLayer.previousLayer
            while type(currentLayer.previousLayer) is not layers.InputLayer:
                currentLayer.previousLayer.error = np.dot(currentLayer.weights.T, currentLayer.error) * (currentLayer.previousLayer.values * (1 - currentLayer.previousLayer.values))
                # currentLayer.previousLayer.error = np.dot(currentLayer.weights.T, currentLayer.error)
                # update weights
                currentLayer.previousLayer.weights += -l_rate * np.dot(currentLayer.previousLayer.error,currentLayer.previousLayer.previousLayer.values.T)
                currentLayer.previousLayer.bias += -l_rate * currentLayer.previousLayer.error

                currentLayer = currentLayer.previousLayer

    # Neural network training
    def train(self, data, output, repetitions, learning_rate):

        for iteration in range(repetitions):
            # progress bar
            if (iteration / repetitions) * 100 % 1 == 0:
                print(((iteration / repetitions) * 100), "%")

            for line in range(len(data)):
                self.inputLayer.values = np.array([data[line]]).T
                self.forwardPropagate(self.outputLayer)
                self.backwardPropagate(np.array([output[line]]).T, learning_rate)

    def predict(self, data):
        for line in range(len(data)):
            self.inputLayer.values = np.array([data[line]]).T

            self.forwardPropagate(self.outputLayer)
            plt.imshow(data[line].reshape(28, 28), cmap="Greys")
            plt.show()
            print(self.outputLayer.values)

    def predictNormalization(self, data, max, min):
        for line in range(len(data)):
            self.inputLayer.values = np.array([data[line]]).T

            self.forwardPropagate(self.outputLayer)
            results = self.outputLayer.values

            for i in range(len(results)):
                results[i] = results[i] * (max[i] - min[i]) + min[i]

            print(results)

