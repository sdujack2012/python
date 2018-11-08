import numpy as np


def sigmoid(x, derivative=False):
    return x*(1-x) if derivative else 1/(1+np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x, True)


class NeuralNetwork:
    def __init__(self, layers, learningRate):
        self.layers = layers
        self.learningRate = learningRate
        self.weights = []

        for i in range(1, len(layers)):
            self.weights.append(np.random.rand(
                self.layers[i-1], self.layers[i]))

    def train(self, input, target):
        self.outputs = [input]
        for i in range(0, len(self.weights)):
            self.outputs.append(
                sigmoid(np.dot(self.outputs[i], self.weights[i])))

        print(self.outputs[len(self.outputs)-1].tolist())
        numberOfLayer = len(self.weights)
        self.output_deltas = [None]*numberOfLayer
        self.output_deltas[numberOfLayer-1] = (
            target - self.outputs[numberOfLayer]) * sigmoid_derivative(self.outputs[numberOfLayer])
        self.weights[numberOfLayer-1] += self.learningRate * \
            np.dot(self.outputs[numberOfLayer-1].T,
                   self.output_deltas[numberOfLayer-1])

        for i in range(numberOfLayer-1, 0, -1):
            self.output_deltas[i-1] = np.dot(self.output_deltas[i], self.weights[i].T) * \
                sigmoid_derivative(self.outputs[i])
            self.weights[i-1] += self.learningRate * \
                np.dot(self.outputs[i-1].T, self.output_deltas[i-1])


def readTrainingImages():
    with open("train-images.idx3-ubyte", "rb") as f:
        f.read(16)
        byteArray = f.read()
        images = []
        n = 28*28

        for i in range(0, len(byteArray), n):
            images.append(list(map(float, byteArray[i:i + n])))

        return np.array(images)/255


def readTrainingLabels():
    with open("train-labels.idx1-ubyte", "rb") as f:
        f.read(8)
        byteArray = f.read()
        labels = []
        for i in range(0, len(byteArray)):
            label = [0] * 10
            label[byteArray[i]] = 1
            labels.append(label)

        return np.array(labels)


def readTestingImages():
    with open("t10k-images.idx3-ubyte", "rb") as f:
        f.read(16)
        byteArray = f.read()
        n = 28*28
        for i in range(0, len(byteArray), n):
            yield byteArray[i:i + n]


def readTestingLabels():
    with open("t10k-labels.idx1-ubyte", "rb") as f:
        f.read(8)
        byteArray = f.read()
        for i in range(0, len(byteArray), 1):
            yield byteArray[i:i + 1]


input = readTrainingImages()
target = readTrainingLabels()
network = NeuralNetwork([28*28, 600, 10], 0.5)
network.train(input, target)

for x in range(0, 10):
    network.train(input, target)


# # network.feedforward()
