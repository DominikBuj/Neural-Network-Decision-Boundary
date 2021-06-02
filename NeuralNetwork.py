from ActivationFunctions import uActivationFunctionDerivative
import numpy as np

randomGenerator = np.random.default_rng()

def splitIntoBatches(data, batchSize):
    for i in range(0, len(data), batchSize):
        yield data[i:i+batchSize]

class LinearLayer():
    def __init__(self, previousLayer, nextLayer):
        self.samples = None
        self.weights = randomGenerator.random(size=(previousLayer+1, nextLayer))
        self.gradient = None
    
    def forward(self, samples, activationFunction):
        samples = np.c_[samples, np.ones(shape=(samples.shape[0], 1)) * -1.0]        
        self.samples = samples
        signals = self.samples @ self.weights
        return signals
    
    def backward(self, gradient, activationFunction):
        self.gradient = gradient
        errors = self.gradient @ np.transpose(self.weights[:-1])
        return errors
    
    def adjust(self, learningRate):
        deltaWeights = learningRate * np.transpose(self.samples) @ self.gradient
        self.weights = self.weights + deltaWeights

class ActivationLayer():
    def __init__(self):
        self.signals = None
    
    def forward(self, signals, activationFunction):
        self.signals = signals
        outputs = activationFunction(self.signals)
        return outputs
    
    def backward(self, errors, activationFunction):
        derivatives = uActivationFunctionDerivative(activationFunction, self.signals)
        gradient = derivatives * errors
        return gradient
    
    def adjust(self, learningRate):
        pass

class NeuralNetwork():
    def __init__(self, hiddenLayerSize, batchSize, initialLearningRate, numOfEpochs):
        self.batchSize = batchSize
        self.initialLearningRate = initialLearningRate
        self.numOfEpochs = numOfEpochs
        self.initLayers(hiddenLayerSize)
    
    def addLayer(self, previousLayer, nextLayer):
        self.layers.append(LinearLayer(previousLayer, nextLayer))
        self.layers.append(ActivationLayer())
    
    def initLayers(self, hiddenLayerSize):
        hiddenLayerWidth, hiddenLayerDepth = hiddenLayerSize
        self.layers = []
        layersSizes = None
        
        if hiddenLayerDepth > 0:
            layersSizes = [(2, 2), (2, hiddenLayerWidth)]
            for hiddenLayer in range(hiddenLayerDepth-1): layersSizes.append((hiddenLayerWidth, hiddenLayerWidth))
            layersSizes.append((hiddenLayerWidth, 2))
        else:
            layersSizes = [(2, 2), (2, 2)]
        
        for layerSize in layersSizes: self.addLayer(layerSize[0], layerSize[1])
    
    def adjustLabels(self, labels):
        labels = np.r_[[labels], [np.ones(len(labels))]]
        for j in range(labels.shape[1]):
            if labels[0, j] == 1.0: labels[1, j] = 0.0
            else: labels[1, j] = 1.0
        return np.transpose(labels)
    
    def forward(self, samples, activationFunction):
        for layer in self.layers: samples = layer.forward(samples, activationFunction)
        return samples
    
    def backward(self, errors, activationFunction):
        for layer in self.layers[::-1]: errors = layer.backward(errors, activationFunction)
    
    def adjust(self, learningRate):
        for layer in self.layers: layer.adjust(learningRate)
    
    def train(self, data, activationFunction):
        samples, labels = data
        labels = self.adjustLabels(labels)
        
        epoch = 0
        learningRate = self.initialLearningRate
        while epoch < self.numOfEpochs and learningRate > 0.0:
            for samplesBatch, labelsBatch in zip(splitIntoBatches(samples, self.batchSize), splitIntoBatches(labels, self.batchSize)):
                outputs = self.forward(samplesBatch, activationFunction)
                errors = 2.0 * (labelsBatch - outputs)
                self.backward(errors, activationFunction)
                self.adjust(learningRate)
            epoch += 1
            learningRate -= 0.0001
    
    def evaluate(self, data, activationFunction, errorFunction):
        samples, labels = data
        labels = self.adjustLabels(labels)
        outputs = self.forward(samples, activationFunction)
        return errorFunction(labels, outputs)