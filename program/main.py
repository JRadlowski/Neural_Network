import layers
import neuralNetwork
import numpy as np
import csv
import pandas as pd

np.set_printoptions(suppress=True)


def fileOpeningInt(name):
    with open(name, newline='') as file:
        reader = csv.reader(file, quoting=csv.QUOTE_NONNUMERIC,
                            delimiter=';')

        # storing all the rows in an output list
        output = []
        for row in reader:
            output.append(row[:])

    return np.array(output).astype(int)


def fileOpeningFloat(name):
    with open(name, newline='') as file:
        reader = csv.reader(file, quoting=csv.QUOTE_NONNUMERIC,
                            delimiter=';')

        # storing all the rows in an output list
        output = []
        for row in reader:
            output.append(row[:])

    return np.array(output).astype(float)


def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, 10))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y


if __name__ == '__main__':
    # ------------- MNIST--------------
    # data = pd.read_csv('data/train.csv')
    # data = np.array(data)
    # m, n = data.shape
    # print(m)
    #
    # data_dev = data[0:40000].T
    # expectedOutputs = data_dev[0]
    # expectedOutputs = np.array([expectedOutputs])
    # expectedOutputs = one_hot(expectedOutputs).T
    # learningData = data_dev[1:n].T
    #
    # data_test = data[41000:41010].T
    # predictiondata = data_test[1:n].T
    # results = data_test[0]

    # LAYER BUILDING
    # inputLayer = layers.InputLayer(np.array([learningData[0]]))
    # hiddenLayer = layers.HiddenLayer(100, inputLayer)
    # hiddenLayer2 = layers.HiddenLayer(30, hiddenLayer)
    # outputLayer = layers.OutputLayer(10, hiddenLayer2)
    #
    # neuralnetwork = neuralNetwork.NeuralNetwork(inputLayer, outputLayer)
    # neuralnetwork.train(learningData, expectedOutputs, 4, 0.01)
    #
    # neuralnetwork.predict(learningData)
    # print(expectedOutputs)

    # ------------- POWER CONSUMPTION -------------------
    # Import data
    data = fileOpeningFloat("data/prad/learningData.csv")
    learningData = data[0:100]
    data2 = fileOpeningFloat("data/prad/expectedOutput.csv")
    expectedOutputs = data2[0:100]

    # Data normalization of inputs
    maxValues = learningData.max(axis=0)
    minValues = learningData.min(axis=0)

    for column in learningData:
        for i in range(len(column)):
            column[i] = (column[i] - minValues[i]) / (maxValues[i] - minValues[i])

    # Data normalization of outputs
    outputmaxValues = expectedOutputs.max(axis=0)
    outputminValues = expectedOutputs.min(axis=0)

    for column in expectedOutputs:
        for i in range(len(column)):
            column[i] = (column[i] - outputminValues[i]) / (outputmaxValues[i] - outputminValues[i])

    # Neural network building
    inputLayer = layers.InputLayer(np.array([learningData[0]]))
    hiddenLayer = layers.HiddenLayer(3, inputLayer)
    hiddenLayer2 = layers.HiddenLayer(3, hiddenLayer)
    outputLayer = layers.OutputLayer(2, hiddenLayer2)

    neuralnetwork = neuralNetwork.NeuralNetwork(inputLayer, outputLayer)
    neuralnetwork.train(learningData, expectedOutputs, 10000, 0.01)

    # neuralnetwork.predict(learningData)
    neuralnetwork.predictNormalization(learningData, outputmaxValues, outputminValues)
