from ActivationFunctions import uActivationFunctionDerivative
import numpy as np

randomGenerator = np.random.default_rng()

def splitIntoBatches(data, batchSize):
    for i in range(0, len(data), batchSize):
        yield data[i:i+batchSize]

class Neuron():
    def __init__(self, batchSize, initialLearningRate, numOfEpochs):
        self.weights = randomGenerator.random(size=3)
        self.batchSize = batchSize
        self.initialLearningRate = initialLearningRate
        self.numOfEpochs = numOfEpochs
    
    def train(self, data, activationFunction):
        samples, labels = data
        samplesStar = np.c_[samples, np.ones(len(samples)) * -1.0]
        
        epoch = 0
        learningRate = self.initialLearningRate
        while epoch < self.numOfEpochs and learningRate > 0.0:
            for samplesStarBatch, labelsBatch in zip(splitIntoBatches(samplesStar, self.batchSize), splitIntoBatches(labels, self.batchSize)):
                signals = self.weights @ np.transpose(samplesStarBatch)
                derivatives = uActivationFunctionDerivative(activationFunction, signals)
                outputs = activationFunction(signals)
                errors = labelsBatch - outputs
                deltaWeights = learningRate * errors * derivatives @ samplesStarBatch
                self.weights = self.weights + deltaWeights
            epoch += 1
            learningRate -= 0.001
    
    def evaluate(self, data, activationFunction, errorFunction):
        samples, labels = data
        samplesStar = np.c_[samples, np.ones(len(samples)) * -1.0]
        signals = self.weights @ np.transpose(samplesStar)
        outputs = activationFunction(signals)
        return errorFunction(labels, outputs)

