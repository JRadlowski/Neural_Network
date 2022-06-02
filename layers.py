import numpy as np

def activationFunction(x):
    return 1 / (1 + np.exp(-x))
    # return x


# Parent class
class Layer:
    def __init__(self):
        self.weights = None
        self.previousLayer = None
        self.values = None
        self.error = None
        self.bias = None

    def results(self):
        return None

class InputLayer(Layer):

    def __init__(self, trainingInput):
        super().__init__()
        self.values = trainingInput.T


class HiddenLayer(Layer):
    np.random.seed(1)

    def results(self):
        self.values = activationFunction(self.bias + np.dot(self.weights, self.previousLayer.values))
        # self.values = self.bias + np.dot(self.weights, self.previousLayer.values)

    def __init__(self, neurons, previousL: Layer):
        super().__init__()
        self.values = None
        self.previousLayer = previousL
        self.weights = 2 * np.random.random((neurons, len(self.previousLayer.values))) - 1
        self.bias = np.zeros((neurons, 1))
        self.results()


class OutputLayer(Layer):
    np.random.seed(1)

    # Activation function

    def results(self):
        self.values = activationFunction(self.bias + np.dot(self.weights, self.previousLayer.values))
        # self.values = self.bias + np.dot(self.weights, self.previousLayer.values)

    def __init__(self, neurons, previousL: Layer):
        super().__init__()
        self.previousLayer = previousL

        self.weights = 2 * np.random.random((neurons, len(self.previousLayer.values))) - 1
        self.bias = self.bias = np.zeros((neurons, 1))
        self.results()
