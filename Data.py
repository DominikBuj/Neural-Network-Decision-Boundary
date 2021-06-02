import numpy as np

randomGenerator = np.random.default_rng()

class Mode():
    def __init__(self, mean, scale, label, numOfSamples):
        self.samples = randomGenerator.normal(
            loc=(randomGenerator.uniform(mean[0], mean[1]), randomGenerator.uniform(mean[0], mean[1])),
            scale=randomGenerator.uniform(scale[0], scale[1]),
            size=(numOfSamples, 2))
        self.labels = np.full(shape=(numOfSamples,), fill_value=label)

class Class():
    def __init__(self, mean, scale, label, numOfSamples, numOfModes):
        self.label = label
        self.modes = [Mode(mean, scale, label, numOfSamples) for mode in range(numOfModes)]
    
    def getSamples(self):
        samples = np.empty(shape=(0, 2))
        for mode in self.modes: samples = np.vstack((samples, mode.samples))
        return samples
    
    def getLabels(self):
        labels = np.array([])
        for mode in self.modes: labels = np.hstack((labels, mode.labels))
        return labels

    def getData(self):
        return np.c_[self.getSamples(), np.transpose(self.getLabels())]