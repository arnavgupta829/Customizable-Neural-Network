import numpy as np
import pandas as pd

def sigmoid(a):
    return (1/(1+(np.exp(-a))))

def sigmoidDerivative(a):
    return np.multiply(sigmoid(a), (1-sigmoid(a)))

class NeuralNetwork:
    def __init__(self, inputNos, outputNos, totalLayers, neuronNos):
        self.inputNos = inputNos
        self.outputNos = outputNos
        self.totalLayers = totalLayers
        self.hiddenLayers = totalLayers-2
        self.neuronNos = neuronNos

        layers = [(inputNos)]
        for i in range(0, totalLayers-2):
            layers.append(neuronNos)
        layers.append(outputNos)

        self.bias = [np.random.rand(x)/100 for x in layers[1:]]
        self.weights = [np.random.rand(y,x)/100 for (x,y) in zip(layers[:-1], layers[1:])]
        self.outputs = [np.zeros(x) for x in layers]
        self.activationDerivative = [np.zeros(x) for x in layers]

    def feedforward(self, inputVector):
        self.outputs[0] = inputVector

        for i in range(1, self.totalLayers):
            temp = np.dot(self.weights[i-1], self.outputs[i-1]) + self.bias[i-1]
            self.activationDerivative[i] = sigmoidDerivative(temp)
            self.outputs[i] = (sigmoid(temp))

    def backprop(self, inputVector, outputVectors, learningRate):
        del_weights = [np.zeros(w.shape) for w in self.weights]
        del_bias = [np.zeros(b.shape) for b in self.bias]

        delta = np.multiply((outputVectors-self.outputs[-1]), self.activationDerivative[-1])
        del_bias[-1] = delta
        del_weights[-1] = np.dot(np.matrix(delta).transpose(), np.matrix(self.outputs[-2]))
        
        for l in range(2, self.totalLayers):
                delta = np.multiply((np.dot(self.weights[-l+1].transpose(), delta)), self.activationDerivative[-l])
                del_bias[-l] = delta
                del_weights[-l] = np.dot(np.matrix(delta).transpose(), np.matrix(self.outputs[-l-1]))

        for i in range(0, len(self.weights)):
            self.weights[i] += del_weights[i]
        for i in range(0, len(self.bias)):
            self.bias[i] += del_bias[i]

    def iteration(self, inputVectors, outputVectors, learningRate, nos):
        for i in range(0, nos):
            for j in range(0, len(inputVectors)):
                self.feedforward(inputVectors[j])
                self.backprop(inputVectors[j], outputVectors[j], learningRate)


def main():
    data = pd.read_csv(r'/home/arnavgupta829/Desktop/ELL409/Assignment1/mnist_train_m.csv')
    raw = []
    for i in data.columns:
        raw.append(data[i])
    raw_np = np.asarray(raw)
    raw_np = raw_np.transpose()
    inputVectors = raw_np[0:, 1:785]
    inputVectors = (inputVectors-np.mean(inputVectors))/256
    outputValues = []
    for i in range(0, len(inputVectors)):
        outputValues.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        outputValues[i][raw_np[i][0]] = 1
    outputVectors = np.asarray(outputValues)

    nn = NeuralNetwork(784, 10, 3, 256)
    print("Iterating-------------------------------------")
    nn.iteration(inputVectors, outputVectors, 0.03, 100)
    print("Done Iterating--------------------------------")

    print("Trying ff after iteration---------------------")
    test_data = pd.read_csv(r'/home/arnavgupta829/Desktop/ELL409/Assignment1/mnist_test.csv')
    raw_test = []
    for i in test_data.columns:
        raw_test.append(test_data[i])
    raw_test_np = np.asarray(raw_test)
    raw_test_np = raw_test_np.transpose()
    raw_test_np = (raw_test_np - np.mean(raw_test_np))/256
    test_outputs = []
    for i in range(0, len(raw_test_np)):
        nn.feedforward(raw_test_np[i])
        outList = nn.outputs[-1]
        maxVal = 0
        for j in range(0, len(outList)):
            if(outList[j]>outList[maxVal]):
                maxVal = j
        test_outputs.append(maxVal)
    print(test_outputs)

    s = "\n".join([str(x) for x in test_outputs])
    with open("submission.csv", "w+") as f:
        f.write(s)

if __name__ == "__main__":
    main()