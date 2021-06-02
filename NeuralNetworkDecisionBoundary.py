import matplotlib as mpl
import seaborn as sns
import tkinter as tk
import numpy as np
import pandas as pd
from tkinter import Tk, Frame, Button, Scale, Label, Spinbox, Radiobutton, Scrollbar, Listbox, IntVar
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg as FigureCanvas
from ActivationFunctions import uHeaviside, uSigmoid, uSin, uTanh, uSign, uReLu, uLeakyReLu
from NeuralNetwork import NeuralNetwork
from Data import Class



### APPLICATION CONSTANTS ###
APPLICATION_GEOMETRY = '900x1000'
APPLICATION_WIDTH, APPLICATION_HEIGHT = 900, 1000



OFFSET = 10

PLOT_FRAME_X, PLOT_FRAME_Y = OFFSET, OFFSET
PLOT_FRAME_WIDTH, PLOT_FRAME_HEIGHT = APPLICATION_WIDTH - 2 * OFFSET, 600 - 2 * OFFSET

OPTIONS_FRAME_X, OPTIONS_FRAME_Y = OFFSET, PLOT_FRAME_HEIGHT + 2 * OFFSET
OPTIONS_FRAME_WIDTH, OPTIONS_FRAME_HEIGHT = PLOT_FRAME_WIDTH, APPLICATION_HEIGHT - OPTIONS_FRAME_Y - OFFSET
OPTION_FRAME_HEIGHT = (OPTIONS_FRAME_HEIGHT - 2 * OFFSET) / 3

DATA_OPTIONS_FRAME_X, DATA_OPTIONS_FRAME_Y = 0, 0
DATA_OPTIONS_FRAME_WIDTH, DATA_OPTIONS_FRAME_HEIGHT = PLOT_FRAME_WIDTH, OPTION_FRAME_HEIGHT



BUTTON_WIDTH, BUTTON_HEIGHT = 125, 75
BUTTON_BORDER = 2
SCALE_HEIGHT = 50

GENERATE_BUTTON_X, GENERATE_BUTTON_Y = DATA_OPTIONS_FRAME_WIDTH - BUTTON_WIDTH - 4 * OFFSET, (DATA_OPTIONS_FRAME_HEIGHT - BUTTON_HEIGHT) / 2

MODES_SCALE_X, MODES_SCALE_Y = 4 * OFFSET, (DATA_OPTIONS_FRAME_HEIGHT - SCALE_HEIGHT) / 2
MODES_SCALE_WIDTH = 225
MODES_LABEL_X, MODES_LABEL_Y = MODES_SCALE_X, MODES_SCALE_Y - 3 * OFFSET

SAMPLES_SPINBOX_WIDTH, SAMPLES_SPINBOX_HEIGHT = 75, 40
SAMPLES_SPINBOX_X, SAMPLES_SPINBOX_Y = GENERATE_BUTTON_X - SAMPLES_SPINBOX_WIDTH - 2 * OFFSET, (DATA_OPTIONS_FRAME_HEIGHT - SAMPLES_SPINBOX_HEIGHT) / 2

SAMPLES_SCALE_X, SAMPLES_SCALE_Y = MODES_SCALE_X + MODES_SCALE_WIDTH + 2 * OFFSET, SCALE_HEIGHT + OFFSET
SAMPLES_SCALE_WIDTH, SAMPLES_SCALE_HEIGHT = SAMPLES_SPINBOX_X - SAMPLES_SCALE_X - 2 * OFFSET, 75
SAMPLES_LABEL_X, SAMPLES_LABEL_Y = SAMPLES_SCALE_X, MODES_LABEL_Y
SAMPLES_LABEL_WIDTH = SAMPLES_SCALE_WIDTH + SAMPLES_SPINBOX_WIDTH + 2 * OFFSET


NEURAL_NETWORK_OPTIONS_FRAME_X, NEURAL_NETWORK_OPTIONS_FRAME_Y = 0, DATA_OPTIONS_FRAME_HEIGHT + OFFSET
NEURAL_NETWORK_OPTIONS_FRAME_WIDTH, NEURAL_NETWORK_OPTIONS_FRAME_HEIGHT = PLOT_FRAME_WIDTH, OPTION_FRAME_HEIGHT

TRAIN_BUTTON_X, TRAIN_BUTTON_Y = NEURAL_NETWORK_OPTIONS_FRAME_WIDTH - BUTTON_WIDTH - 4 * OFFSET, (NEURAL_NETWORK_OPTIONS_FRAME_HEIGHT - BUTTON_HEIGHT) / 2

ERROR_LABEL_WIDTH, ERROR_LABEL_HEIGHT = 150, 50
ERROR_LABEL_X, ERROR_LABEL_Y = TRAIN_BUTTON_X - 2 * OFFSET - ERROR_LABEL_WIDTH, (NEURAL_NETWORK_OPTIONS_FRAME_HEIGHT - ERROR_LABEL_HEIGHT) / 2

ERROR_BUTTON_WIDTH, ERROR_BUTTON_HEIGHT = 75, 50
ERROR_BUTTON_X = ERROR_LABEL_X - 2 * OFFSET - ERROR_BUTTON_WIDTH
ERROR_ONE_BUTTON_Y, ERROR_TWO_BUTTON_Y = (NEURAL_NETWORK_OPTIONS_FRAME_HEIGHT / 4) - (ERROR_BUTTON_HEIGHT / 2), (3 * NEURAL_NETWORK_OPTIONS_FRAME_HEIGHT / 4) - (ERROR_BUTTON_HEIGHT / 2)

ACTIVATION_FUNCTION_SCROLLBAR_WIDTH, ACTIVATION_FUNCTION_SCROLLBAR_HEIGHT = ERROR_BUTTON_X - 6 * OFFSET, NEURAL_NETWORK_OPTIONS_FRAME_HEIGHT - 2 * OFFSET
ACTIVATION_FUNCTION_SCROLLBAR_X, ACTIVATION_FUNCTION_SCROLLBAR_Y = 4 * OFFSET, (NEURAL_NETWORK_OPTIONS_FRAME_HEIGHT - ACTIVATION_FUNCTION_SCROLLBAR_HEIGHT) / 2


HIDDEN_LAYER_OPTIONS_FRAME_X, HIDDEN_LAYER_OPTIONS_FRAME_Y = 0, OPTIONS_FRAME_HEIGHT - OPTION_FRAME_HEIGHT
HIDDEN_LAYER_OPTIONS_FRAME_WIDTH, HIDDEN_LAYER_OPTIONS_FRAME_HEIGHT = PLOT_FRAME_WIDTH, OPTION_FRAME_HEIGHT

NEURAL_NETWORK_SCALE_WIDTH = (HIDDEN_LAYER_OPTIONS_FRAME_WIDTH - 12 * OFFSET) / 2

NEURAL_NETWORK_WIDTH_SCALE_X, NEURAL_NETWORK_WIDTH_SCALE_Y = 4 * OFFSET, (HIDDEN_LAYER_OPTIONS_FRAME_HEIGHT - SCALE_HEIGHT) / 2
NEURAL_NETWORK_WIDTH_SCALE_WIDTH = NEURAL_NETWORK_SCALE_WIDTH
NEURAL_NETWORK_WIDTH_LABEL_X, NEURAL_NETWORK_WIDTH_LABEL_Y = NEURAL_NETWORK_WIDTH_SCALE_X, NEURAL_NETWORK_WIDTH_SCALE_Y - 3 * OFFSET
NEURAL_NETWORK_WIDTH_LABEL_WIDTH = NEURAL_NETWORK_SCALE_WIDTH

NEURAL_NETWORK_DEPTH_SCALE_X, NEURAL_NETWORK_DEPTH_SCALE_Y = NEURAL_NETWORK_SCALE_WIDTH + 8 * OFFSET, NEURAL_NETWORK_WIDTH_SCALE_Y
NEURAL_NETWORK_DEPTH_SCALE_WIDTH = NEURAL_NETWORK_SCALE_WIDTH
NEURAL_NETWORK_DEPTH_LABEL_X, NEURAL_NETWORK_DEPTH_LABEL_Y = NEURAL_NETWORK_DEPTH_SCALE_X, NEURAL_NETWORK_WIDTH_LABEL_Y
NEURAL_NETWORK_DEPTH_LABEL_WIDTH = NEURAL_NETWORK_SCALE_WIDTH



### NEURAL NETWORK CONSTANTS ###
MEAN_SQUARED_ERROR = 0
MEAN_ABSOLUTE_ERROR = 1

MIN_MEAN, MAX_MEAN = 0.0, 1.0
MIN_PLOT_VALUE, MAX_PLOT_VALUE = MIN_MEAN - 0.2, MAX_MEAN + 0.2
MIN_SCALE, MAX_SCALE = 0.01, 0.025

MIN_NUM_OF_MODES, MAX_NUM_OF_MODES = 1, 5
MIN_NUM_OF_SAMPLES, MAX_NUM_OF_SAMPLES = 10, 100

MIN_NEURAL_NETWORK_WIDTH, MAX_NEURAL_NETWORK_WIDTH = 3, 5
MIN_NEURAL_NETWORK_DEPTH, MAX_NEURAL_NETWORK_DEPTH = 1, 3

BATCH_SIZE = 16
INITIAL_LEARNING_RATE = 0.01
NUM_OF_EPOCHS = 100



### FONTS AND COLORS ###
BUTTON_FONT = ('Verdana', 15)
OPTIONS_FONT = ('Verdana', 13)

BACKGROUND_COLOR = '#77a8a8'
PLOT_BACKGROUND_COLOR = '#d5f4e6'

paletteDictionary = {1.0: 'green', 0.0: 'red'}



randomGenerator = np.random.default_rng()

def meanAbsoluteError(labels, outputs):
    return np.mean(np.mean(np.abs(labels - outputs), axis=0))

def meanSquaredError(labels, outputs):
    return np.mean(np.mean((labels - outputs) * (labels - outputs), axis=0))

def generateData():
    application.dataOptionsFrame.checkSpinboxValue()
    application.plotFrame.updateDataSampleVariables()
    application.plotFrame.updateNeuralNetworkVariables()
    application.plotFrame.generateData()
    application.plotFrame.drawPlot()

def trainNeuron():
    if application.plotFrame.classes is not None:
        application.plotFrame.trainNeuron()
        application.plotFrame.evaluateNeuron()
        application.plotFrame.drawPlot()

def updateActivationFunction(event):
    listbox = application.neuralNetworkOptionsFrame.activationFunctionListbox
    activationFunctionText = str(listbox.get(listbox.curselection()))
    if activationFunctionText == "Heaviside": application.plotFrame.activationFunction = uHeaviside
    elif activationFunctionText == "Sigmoid": application.plotFrame.activationFunction = uSigmoid
    elif activationFunctionText == "Sin": application.plotFrame.activationFunction = uSin
    elif activationFunctionText == "Tanh": application.plotFrame.activationFunction = uTanh
    elif activationFunctionText == "Sign": application.plotFrame.activationFunction = uSign
    elif activationFunctionText == "ReLu": application.plotFrame.activationFunction = uReLu
    elif activationFunctionText == "Leaky ReLu": application.plotFrame.activationFunction = uLeakyReLu

def updateErrorFunction():
    chosenErrorFunction = application.neuralNetworkOptionsFrame.errorFunction.get()
    if (chosenErrorFunction == MEAN_ABSOLUTE_ERROR): application.plotFrame.errorFunction = meanAbsoluteError
    else: application.plotFrame.errorFunction = meanSquaredError

class Application(Tk):
    def __init__(self):
        Tk.__init__(self)
        
        Tk.wm_title(self, 'Neural Network Decision Boundary')
        self.geometry(APPLICATION_GEOMETRY)
        self.config(background=BACKGROUND_COLOR)
        
        
        self.plotFrame = PlotFrame(self)
        self.plotFrame.place(x=PLOT_FRAME_X,
                             y=PLOT_FRAME_Y,
                             width=PLOT_FRAME_WIDTH,
                             height=PLOT_FRAME_HEIGHT)
        
        
        self.optionsFrame = Frame(self, background=PLOT_BACKGROUND_COLOR)
        self.optionsFrame.place(x=OPTIONS_FRAME_X,
                                y=OPTIONS_FRAME_Y,
                                width=OPTIONS_FRAME_WIDTH,
                                height=OPTIONS_FRAME_HEIGHT)
        
        self.dataOptionsFrame = DataOptionsFrame(self.optionsFrame)
        self.dataOptionsFrame.place(x=DATA_OPTIONS_FRAME_X,
                                    y=DATA_OPTIONS_FRAME_Y,
                                    width=DATA_OPTIONS_FRAME_WIDTH,
                                    height=DATA_OPTIONS_FRAME_HEIGHT)
        
        self.neuralNetworkOptionsFrame = NeuralNetworkOptionsFrame(self.optionsFrame)
        self.neuralNetworkOptionsFrame.place(x=NEURAL_NETWORK_OPTIONS_FRAME_X,
                                             y=NEURAL_NETWORK_OPTIONS_FRAME_Y,
                                             width=NEURAL_NETWORK_OPTIONS_FRAME_WIDTH,
                                             height=NEURAL_NETWORK_OPTIONS_FRAME_HEIGHT)
        
        self.hiddenLayerOptionsFrame = HiddenLayerOptionsFrame(self.optionsFrame)
        self.hiddenLayerOptionsFrame.place(x=HIDDEN_LAYER_OPTIONS_FRAME_X,
                                           y=HIDDEN_LAYER_OPTIONS_FRAME_Y,
                                           width=HIDDEN_LAYER_OPTIONS_FRAME_WIDTH,
                                           height=HIDDEN_LAYER_OPTIONS_FRAME_HEIGHT)

def splitDataTrainingEvaluation(data):
    """
    Splits the input data into training (80%) and evaluation (20%) parts
    """
    height, width = data.shape
    indexes = np.linspace(0, height-1, num=height, dtype=int)
    trainingIndexes = np.random.choice(indexes, size=int(0.8 * height), replace=False)
    evaluationIndexes = indexes[list(set(range(height)) - set(trainingIndexes))]
    
    trainingData, evaluationData = np.empty(shape=(0, 3)), np.empty(shape=(0, 3))
    for index in trainingIndexes: trainingData = np.vstack((trainingData, data[index]))
    for index in evaluationIndexes: evaluationData = np.vstack((evaluationData, data[index]))
    
    return trainingData, evaluationData

def splitDataSamplesLabels(data):
    """
    Splits the input data into samples and labels
    """
    samples = data[:, :-1]
    twoDimLabels = data[:, -1:]
    
    labels = []
    for label in twoDimLabels: labels = np.hstack((labels, label))
    
    return samples, labels

class PlotFrame(Frame):
    def __init__(self, parent):
        Frame.__init__(self, parent)
        
        self.numOfModes = MIN_NUM_OF_MODES
        self.numOfSamples = MIN_NUM_OF_SAMPLES
        self.classes = None
        
        self.hiddenLayerSize = (3, 1)
        self.neuralNetwork = NeuralNetwork(self.hiddenLayerSize, BATCH_SIZE,
                                           INITIAL_LEARNING_RATE, NUM_OF_EPOCHS)
        
        self.activationFunction = uHeaviside
        self.errorFunction = meanAbsoluteError
        self.trainingData, self.evaluationData = None, None
        self.evaluationError = None
        
        self.figure = Figure(facecolor=PLOT_BACKGROUND_COLOR)
        self.axis = self.figure.add_subplot()
        self.plot = sns.scatterplot(ax=self.axis)
        self.plot.set(xlim=(MIN_PLOT_VALUE, MAX_PLOT_VALUE),
                      ylim=(MIN_PLOT_VALUE, MAX_PLOT_VALUE),
                      xlabel='x', ylabel='y')
        
        self.canvas = FigureCanvas(self.figure, self)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas.draw()
    
    def updateDataSampleVariables(self):
        self.numOfModes = application.dataOptionsFrame.numOfModesScale.get()
        self.numOfSamples = application.dataOptionsFrame.numOfSamplesScale.get()
    
    def updateNeuralNetworkVariables(self):
        self.hiddenLayerSize = (application.hiddenLayerOptionsFrame.neuralNetworkWidthScale.get(),
                                application.hiddenLayerOptionsFrame.neuralNetworkDepthScale.get())
    
    def generateData(self):
        """
        Generates sample data, splitting it into training and evaluation and shuffling it
        """
        self.classes = [Class((MIN_MEAN, MAX_MEAN), (MIN_SCALE, MAX_SCALE),
                              1.0, self.numOfSamples, self.numOfModes),
                        Class((MIN_MEAN, MAX_MEAN), (MIN_SCALE, MAX_SCALE),
                              0.0, self.numOfSamples, self.numOfModes)]
        self.neuralNetwork = NeuralNetwork(self.hiddenLayerSize, BATCH_SIZE,
                                           INITIAL_LEARNING_RATE, NUM_OF_EPOCHS)
        
        trainingData, evaluationData = np.empty(shape=(0, 3)), np.empty(shape=(0, 3))
        for tempClass in self.classes:
            tempTrainingData, tempEvaluationData = splitDataTrainingEvaluation(tempClass.getData())
            trainingData = np.vstack((trainingData, tempTrainingData))
            evaluationData = np.vstack((evaluationData, tempEvaluationData))
        
        randomGenerator.shuffle(trainingData)
        randomGenerator.shuffle(evaluationData)
        self.trainingData = trainingData
        self.evaluationData = evaluationData
        
        self.evaluateNeuron()
    
    def createDataFrame(self) -> pd.DataFrame:
        """
        Creates a data frame from the sample data for the plot
        """
        data = np.empty(shape=(0, 3))
        for tempClass in self.classes: data = np.vstack((data, tempClass.getData()))
        return pd.DataFrame(data, columns=['x', 'y', 'label']) 
    
    def drawData(self):
        """
        Draws a plot of the generated sample data
        """
        self.plot = sns.scatterplot(data=self.createDataFrame(),
                                    x='x', y='y', hue='label',
                                    palette=paletteDictionary,
                                    legend=False, ax=self.axis)
        self.plot.set(xlim=(MIN_PLOT_VALUE, MAX_PLOT_VALUE),
                      ylim=(MIN_PLOT_VALUE, MAX_PLOT_VALUE),
                      xlabel='x', ylabel='y')
    
    def drawDecisionBoundary(self):
        """
        Draws the decision boundary on the plot
        """
        x, y = np.linspace(MIN_PLOT_VALUE, MAX_PLOT_VALUE), np.linspace(MIN_PLOT_VALUE, MAX_PLOT_VALUE)
        xx, yy = np.meshgrid(x, y)
        
        samples = np.empty(shape=(0, 2))
        for i in range(len(xx)):
            for j in range(len(yy)):
                samples = np.vstack((samples, np.array([[xx[i, j], yy[i, j]]])))
        z = self.neuralNetwork.forward(samples, self.activationFunction)
        z = np.transpose(z)
        classOneZ = z[0]
        classTwoZ = z[1]
        classOneZ = np.reshape(classOneZ, xx.shape)
        classTwoZ = np.reshape(classTwoZ, xx.shape)
                
        self.axis.contourf(xx, yy, classOneZ, cmap=mpl.cm.Greens, alpha=0.5)
        self.axis.contourf(xx, yy, classTwoZ, cmap=mpl.cm.Reds, alpha=0.5)
        
    def drawPlot(self):
        self.axis.clear()
        self.drawData()
        self.drawDecisionBoundary()
        self.canvas.draw()
    
    def trainNeuron(self):
        samples, labels = splitDataSamplesLabels(self.trainingData)
        self.neuralNetwork.train([samples, labels], self.activationFunction)
        
    def evaluateNeuron(self):
        samples, labels = splitDataSamplesLabels(self.evaluationData)
        self.evaluationError = round(self.neuralNetwork.evaluate([samples, labels], self.activationFunction, self.errorFunction), 2)
        application.neuralNetworkOptionsFrame.evaluationErrorLabel.config(text=f"Error: {self.evaluationError}")
    
class DataOptionsFrame(Frame):
    def __init__(self, parent):
        Frame.__init__(self, parent, background=PLOT_BACKGROUND_COLOR)
        
        self.numOfModesLabel = Label(self, text='Number of Modes',
                                     background=PLOT_BACKGROUND_COLOR,
                                     font=OPTIONS_FONT)
        self.numOfModesLabel.place(x=MODES_LABEL_X, y=MODES_LABEL_Y,
                                   width=MODES_SCALE_WIDTH)
        self.numOfModesScale = Scale(self)
        self.numOfModesScale.place(x=MODES_SCALE_X, y=MODES_SCALE_Y,
                                   width=MODES_SCALE_WIDTH)
        self.numOfModesScale.config(from_=MIN_NUM_OF_MODES, to=MAX_NUM_OF_MODES,
                                    tickinterval=1, orient=tk.HORIZONTAL,
                                    showvalue=True, background=BACKGROUND_COLOR,
                                    highlightthickness=0, troughcolor=PLOT_BACKGROUND_COLOR,
                                    font=OPTIONS_FONT)
        
        self.numOfSamplesLabel = Label(self, text='Number of Samples', background=PLOT_BACKGROUND_COLOR,
                                       font=OPTIONS_FONT)
        self.numOfSamplesLabel.place(x=SAMPLES_LABEL_X, y=SAMPLES_LABEL_Y,
                                     width=SAMPLES_LABEL_WIDTH)
        self.numOfSamplesScale = Scale(self)
        self.numOfSamplesScale.place(x=SAMPLES_SCALE_X, y=SAMPLES_SCALE_Y,
                                     width=SAMPLES_SCALE_WIDTH)
        self.numOfSamplesScale.config(from_=MIN_NUM_OF_SAMPLES, to=MAX_NUM_OF_SAMPLES,
                                      orient=tk.HORIZONTAL, showvalue=False,
                                      background=BACKGROUND_COLOR, highlightthickness=0,
                                      troughcolor=PLOT_BACKGROUND_COLOR, font=OPTIONS_FONT,
                                      command=self.updateSamplesSpinbox)
        self.numOfSamplesSpinbox = Spinbox(self)
        self.numOfSamplesSpinbox.place(x=SAMPLES_SPINBOX_X, y=SAMPLES_SPINBOX_Y,
                                       width=SAMPLES_SPINBOX_WIDTH, height=SAMPLES_SPINBOX_HEIGHT)
        self.numOfSamplesSpinbox.config(from_=MIN_NUM_OF_SAMPLES, to=MAX_NUM_OF_SAMPLES,
                                        font=BUTTON_FONT, background=PLOT_BACKGROUND_COLOR,
                                        command=self.updateSamplesScale)
        
        self.generateButton = Button(self)
        self.generateButton.place(x=GENERATE_BUTTON_X, y=GENERATE_BUTTON_Y,
                                  width=BUTTON_WIDTH, height=BUTTON_HEIGHT)
        self.generateButton.config(text='Generate', font=BUTTON_FONT,
                                   background=PLOT_BACKGROUND_COLOR,
                                   border=BUTTON_BORDER, command=generateData)
        
    def updateSamplesScale(self):
        self.numOfSamplesScale.set(int(self.numOfSamplesSpinbox.get()))
    
    def updateSamplesSpinbox(self, numOfSamples):
        self.numOfSamplesSpinbox.delete(0, 'end')
        self.numOfSamplesSpinbox.insert(0, numOfSamples)
    
    def checkSpinboxValue(self):
        numOfSamples = int(self.numOfSamplesSpinbox.get())
        
        if numOfSamples < MIN_NUM_OF_SAMPLES:
            self.numOfSamplesSpinbox.delete(0, 'end')
            self.numOfSamplesSpinbox.insert(0, MIN_NUM_OF_SAMPLES)
        elif numOfSamples > MAX_NUM_OF_SAMPLES:
            self.numOfSamplesSpinbox.delete(0, 'end')
            self.numOfSamplesSpinbox.insert(0, MAX_NUM_OF_SAMPLES)

class NeuralNetworkOptionsFrame(Frame):
    def __init__(self, parent):
        Frame.__init__(self, parent, background=PLOT_BACKGROUND_COLOR)
        
        self.activationFunctionFrame = Frame(self)
        self.activationFunctionFrame.place(x=ACTIVATION_FUNCTION_SCROLLBAR_X,
                                           y=ACTIVATION_FUNCTION_SCROLLBAR_Y,
                                           width=ACTIVATION_FUNCTION_SCROLLBAR_WIDTH,
                                           height=ACTIVATION_FUNCTION_SCROLLBAR_HEIGHT)
        self.activationFunctionScrollbar = Scrollbar(self.activationFunctionFrame,
                                                     background=PLOT_BACKGROUND_COLOR)
        self.activationFunctionScrollbar.pack(side=tk.LEFT, fill=tk.BOTH)
        self.activationFunctionListbox = Listbox(self.activationFunctionFrame,
                                                 yscrollcommand=self.activationFunctionScrollbar.set,
                                                 font=OPTIONS_FONT,
                                                 background=PLOT_BACKGROUND_COLOR)
        self.activationFunctionListbox.insert(tk.END, "Heaviside")
        self.activationFunctionListbox.insert(tk.END, "Sigmoid")
        self.activationFunctionListbox.insert(tk.END, "Sin")
        self.activationFunctionListbox.insert(tk.END, "Tanh")
        self.activationFunctionListbox.insert(tk.END, "Sign")
        self.activationFunctionListbox.insert(tk.END, "ReLu")
        self.activationFunctionListbox.insert(tk.END, "Leaky ReLu")
        self.activationFunctionListbox.pack(side=tk.TOP, fill=tk.BOTH)
        self.activationFunctionListbox.bind('<<ListboxSelect>>', updateActivationFunction)
        self.activationFunctionScrollbar.config(command=self.activationFunctionListbox.yview)
        
        self.errorFunction = IntVar()
        self.errorFunctionOneButton = Radiobutton(self, text="MSE", variable=self.errorFunction,
                                                  value=MEAN_SQUARED_ERROR, command=updateErrorFunction,
                                                  background=PLOT_BACKGROUND_COLOR, font=OPTIONS_FONT)
        self.errorFunctionOneButton.place(x=ERROR_BUTTON_X, y=ERROR_ONE_BUTTON_Y,
                                          width=ERROR_BUTTON_WIDTH, height=ERROR_BUTTON_HEIGHT)
        self.errorFunctionTwoButton = Radiobutton(self, text="MAE", variable=self.errorFunction,
                                                  value=MEAN_ABSOLUTE_ERROR, command=updateErrorFunction,
                                                  background=PLOT_BACKGROUND_COLOR, font=OPTIONS_FONT)
        self.errorFunctionTwoButton.place(x=ERROR_BUTTON_X, y=ERROR_TWO_BUTTON_Y,
                                          width=ERROR_BUTTON_WIDTH, height=ERROR_BUTTON_HEIGHT)
        
        self.evaluationErrorLabel = Label(self, text=f"Error: {None}", background=PLOT_BACKGROUND_COLOR, font=BUTTON_FONT)
        self.evaluationErrorLabel.place(x=ERROR_LABEL_X, y=ERROR_LABEL_Y,
                                        width=ERROR_LABEL_WIDTH, height=ERROR_LABEL_HEIGHT)
        
        self.trainButton = Button(self)
        self.trainButton.place(x=TRAIN_BUTTON_X, y=TRAIN_BUTTON_Y,
                               width=BUTTON_WIDTH, height=BUTTON_HEIGHT)
        self.trainButton.config(text='Train', font=BUTTON_FONT, background=PLOT_BACKGROUND_COLOR,
                                border=BUTTON_BORDER, command=trainNeuron)

class HiddenLayerOptionsFrame(Frame):
    def __init__(self, parent):
        Frame.__init__(self, parent, background=PLOT_BACKGROUND_COLOR)
        
        self.neuralNetworkWidthLabel = Label(self, text='Hidden Layer Width',
                                             background=PLOT_BACKGROUND_COLOR,
                                             font=OPTIONS_FONT)
        self.neuralNetworkWidthLabel.place(x=NEURAL_NETWORK_WIDTH_LABEL_X, y=NEURAL_NETWORK_WIDTH_LABEL_Y,
                                           width=NEURAL_NETWORK_WIDTH_LABEL_WIDTH)
        self.neuralNetworkWidthScale = Scale(self)
        self.neuralNetworkWidthScale.place(x=NEURAL_NETWORK_WIDTH_SCALE_X, y=NEURAL_NETWORK_WIDTH_SCALE_Y,
                                           width=NEURAL_NETWORK_WIDTH_SCALE_WIDTH)
        self.neuralNetworkWidthScale.config(from_=MIN_NEURAL_NETWORK_WIDTH, to=MAX_NEURAL_NETWORK_WIDTH,
                                            tickinterval=1, orient=tk.HORIZONTAL,
                                            showvalue=True, background=BACKGROUND_COLOR,
                                            highlightthickness=0, troughcolor=PLOT_BACKGROUND_COLOR,
                                            font=OPTIONS_FONT)
        
        self.neuralNetworkDepthLabel = Label(self, text='Hidden Layer Depth',
                                             background=PLOT_BACKGROUND_COLOR,
                                             font=OPTIONS_FONT)
        self.neuralNetworkDepthLabel.place(x=NEURAL_NETWORK_DEPTH_LABEL_X, y=NEURAL_NETWORK_DEPTH_LABEL_Y,
                                           width=NEURAL_NETWORK_DEPTH_LABEL_WIDTH)
        self.neuralNetworkDepthScale = Scale(self)
        self.neuralNetworkDepthScale.place(x=NEURAL_NETWORK_DEPTH_SCALE_X, y=NEURAL_NETWORK_DEPTH_SCALE_Y,
                                           width=NEURAL_NETWORK_DEPTH_SCALE_WIDTH)
        self.neuralNetworkDepthScale.config(from_=MIN_NEURAL_NETWORK_DEPTH, to=MAX_NEURAL_NETWORK_DEPTH,
                                            tickinterval=1, orient=tk.HORIZONTAL,
                                            showvalue=True, background=BACKGROUND_COLOR,
                                            highlightthickness=0, troughcolor=PLOT_BACKGROUND_COLOR,
                                            font=OPTIONS_FONT)

"""
Configures the style variables of the plot
"""
mpl.rcdefaults()
sns.set(rc={'axes.facecolor': 'white', 'axes.edgecolor': 'black', 'axes.grid': True,
            'grid.color': 'black', 'grid.linewidth': 0.4, 'grid.alpha': 0.5,
            'axes.grid.which': 'both', 'xtick.bottom': True, 'xtick.minor.visible': True,
            'ytick.left': True, 'ytick.minor.visible': True, 'xtick.minor.width': 0.6,
            'ytick.minor.width': 0.6, 'font.family': 'fantasy', 'font.style': 'normal',
            'lines.markersize': 4})

application = Application()
application.neuralNetworkOptionsFrame.errorFunctionOneButton.invoke()
application.mainloop()