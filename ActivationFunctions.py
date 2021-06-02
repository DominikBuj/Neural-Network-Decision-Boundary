import numpy as np
import math

def heaviside(signal):
    if signal < 0.0: return 0.0
    else: return 1.0

uHeaviside = np.frompyfunc(heaviside, 1, 1)

BETA = -6.0

def sigmoid(signal):
    return 1.0 / (1.0 + np.exp(BETA * signal))

uSigmoid = np.frompyfunc(sigmoid, 1, 1)

def sin(signal):
    return math.sin(signal)

uSin = np.frompyfunc(sin, 1, 1)

def tanh(signal):
    return math.tanh(signal)

uTanh = np.frompyfunc(tanh, 1, 1)

def sign(signal):
    if signal < 0.0: return -1.0
    elif signal == 0.0: return 0.0
    else: return 1.0

uSign = np.frompyfunc(sign, 1, 1)

def reLu(signal):
    if signal > 0.0: return signal
    else: return 0.0

uReLu = np.frompyfunc(reLu, 1, 1)

def leakyReLu(signal):
    if signal > 0.0: return signal
    else: return 0.01 * signal

uLeakyReLu = np.frompyfunc(leakyReLu, 1, 1)

def activationFunctionDerivative(activationFunction, signal):
    if activationFunction == uHeaviside:
        return 1.0
    elif activationFunction == uSigmoid:
        fX = sigmoid(signal)
        return fX * (1.0 - fX)
    elif activationFunction == uSin:
        return math.cos(signal)
    elif activationFunction == uTanh:
        fX = math.tanh(signal)
        return 1.0 - (fX * fX)
    elif activationFunction == uSign:
        return 1.0
    elif activationFunction == uReLu:
        if signal > 0.0: return 1.0
        else: return 0.01
    elif activationFunction == uLeakyReLu:
        if signal > 0.0: return 1.0
        else: return 0.01

uActivationFunctionDerivative = np.frompyfunc(activationFunctionDerivative, 2, 1)
