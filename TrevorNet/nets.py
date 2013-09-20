from neurons import InputNeuron, HiddenNeuron, OutputNeuron
import itertools
import random

class FeedForwardNet(object):
    """A backpropagating feed forward neural network."""

    def __init__(self, neuroncounts, learningrate, bias, weightfunc=random.random):
        """
        Params:
            neuroncounts: The neuron count for each layer. For example to make a
                network with 2 input neurons, 2 hidden neurons, and 1 output
                neuron you would pass (2, 2, 1). A four layer network could
                be constructed with (1, 3, 3, 7).
            learningrate: A small value, .1 or .2, that determines how quickly
                the net will react to training. Higher value is faster but has
                a risk of overshooting.
            bias: The initial bias (usually set to 1). Shifts the sigmoid curve
                making it possible to represent more functions.
            weightfunc: The function that determines the initial weight between
                any two neurons.
        """
        if len(neuroncounts) < 2:
            raise ValueError("Need at least two layers")

        #create layers of neurons (which can also be accessed via properties)
        inputcount = neuroncounts[0]
        hiddencounts = [i for i in neuroncounts[1:-1]]
        outputcount = neuroncounts[-1]

        inputlayer = [InputNeuron() for i in range(inputcount)]
        hiddenlayers = [[HiddenNeuron(learningrate, bias) for i in range(k)] for k in hiddencounts]
        outputlayer = [OutputNeuron(learningrate, bias) for i in range(outputcount)]

        self._layers = [inputlayer] + hiddenlayers + [outputlayer]

        #connect layers
        for i in range(len(self._layers)-1):
            for pair in itertools.product(self._layers[i], self._layers[i+1]):
                pair[0].connect_forward(pair[1], weightfunc())

    def train(self, data, targets):
        """Train the neural net with the given input data and targets.

        Params:
            data: Some sequence of input data to map to input neurons
            targets: Some sequence of output data to compare against output neurons
        """

        if len(data) != len(self.inputlayer):
            raise ValueError("Data must have same number of elements as input layer")
        elif len(targets) != len(self.outputlayer):
            raise ValueError("Targets must have same number of elements as output layer")

        self._propagate(data)
        self._backpropagate(targets)

    def predict(self, data):
        """Predict the target for the data.
        
        Params:
            data: Some sequence of input data to map to input neurons

        Return:
            a sequence of numerical signals from the output neurons.
        """
        if len(data) != len(self.inputlayer):
            raise ValueError("Data must have same number of elements as input layer")

        self._propagate(data)
        out = [neuron.signal for neuron in self.outputlayer]
        return out

    def _propagate(self, data):
        for idx, val in enumerate(data):
            self.inputlayer[idx] = val
        for layer in self._layers[1:]:
            for neuron in layer:
                neuron.propagate()
    
    def _backpropagate(self, targets):
        for idx, val in enumerate(targets):
            self.outputlayer[idx].backpropagate1(val)
        for layer in self.hiddenlayers[::-1]:
            for neuron in layer:
                neuron.backpropagate2()

    @property
    def inputlayer(self):
        return self._layers[0]

    @property
    def hiddenlayers(self):
        return self._layers[1:-1]

    @property
    def outputlayer(self):
        return self._layers[-1]

