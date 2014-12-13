import itertools
import random
import math


class PyFeedForwardNet(object):
    """A backpropagating feed forward neural network."""

    def __init__(self, layers):
        self._layers = layers

    @classmethod
    def fromfuncs(cls, neuroncounts, learningrate,
                   biasfunc=lambda: random.uniform(-1, 1),
                   weightfunc=lambda: random.uniform(-1, 1)):
        """
        Params:
            neuroncounts: The neuron count for each layer. For example to make
                a network with 2 input neurons, 2 hidden neurons, and 1 output
                neuron you would pass (2, 2, 1). A four layer network could
                be constructed with (1, 3, 3, 7).
            learningrate: A small value, .1 or .2, that determines how quickly
                the net will react to training. Higher value is faster but has
                a risk of overshooting.
            biasfunc: The initial bias. Shifts the sigmoid curve making it
                possible to represent more functions.
            weightfunc: The function that determines the initial weight between
                any two neurons.
        """
        if len(neuroncounts) < 2:
            raise ValueError("Need at least two layers")

        # create layers of neurons (which can also be accessed via properties)
        inputcount = neuroncounts[0]
        hiddencounts = [i for i in neuroncounts[1:-1]]
        outputcount = neuroncounts[-1]

        inputlayer = [InputNeuron() for i in range(inputcount)]
        hiddenlayers = [[HiddenNeuron(learningrate, biasfunc())
                        for i in range(k)]
                        for k in hiddencounts]
        outputlayer = [OutputNeuron(learningrate, biasfunc())
                       for i in range(outputcount)]

        layersresult = [inputlayer] + hiddenlayers + [outputlayer]

        # connect layers
        for i in range(len(layersresult)-1):
            for pair in itertools.product(layersresult[i], layersresult[i+1]):
                pair[0].connect_forward(pair[1], weightfunc())

        return cls(layersresult)

    @classmethod
    def fromlist(cls, layers):
        '''Takes an input in the form [inputcount, x, x, ...,  x]
           where each x is a list of
           [learningrate, bias, weights_from_previous_layer]
        '''
        inputcount = layers[0]

        inputlayer = [InputNeuron() for i in range(inputcount)]
        hiddenlayers = [[HiddenNeuron(n[0], n[1]) for n in layer]
                        for layer in layers[1:-1]]
        outputlayer = [OutputNeuron(n[0], n[1]) for n in layers[-1]]

        layersresult = [inputlayer] + hiddenlayers + [outputlayer]

        for i in range(len(layersresult)-1):
            leftlayer = layersresult[i]
            rightlayer = layersresult[i+1]
            for rneuronidx, rneuron in enumerate(rightlayer):
                weights = layers[i+1][rneuronidx][2]
                for lneuronidx, lneuron in enumerate(leftlayer):
                    weight = weights[lneuronidx]
                    lneuron.connect_forward(rneuron, weight)

        return cls(layersresult)

    @classmethod
    def fromfile(cls, fpath):
        with open(fpath) as f:
            thelist = eval(f.read())
        return cls.fromlist(thelist)

    def tolist(self):
        layers_rep = []
        layers_rep.append(len(self._layers[0]))
        for layer in self._layers[1:]:
            layer_rep = []
            for neuron in layer:
                a = neuron._learningrate
                b = neuron._bias
                c = [conn._weight for conn in neuron._back]
                layer_rep.append([a, b, c])
            layers_rep.append(layer_rep)
        return layers_rep

    def tofile(self, fname):
        savednet = str(self._net.tolist())
        with open(fname, 'w') as f:
            f.write(savednet)

    def train(self, data, targets):
        """Train the neural net with the given input data and targets.

        Params:
            data: Some sequence of input data to map to input neurons
            targets: Some sequence of output data to compare against output
                     neurons
        """

        if len(data) != len(self.inputlayer):
            msg = "Data must have same number of elements as input layer"
            raise ValueError(msg)
        elif len(targets) != len(self.outputlayer):
            msg = "Targets must have same number of elements as output layer"
            raise ValueError(msg)

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
            msg = "Data must have same number of elements as input layer"
            raise ValueError(msg)

        self._propagate(data)
        out = [neuron.signal for neuron in self.outputlayer]
        return out

    def _propagate(self, data):
        for idx, val in enumerate(data):
            self.inputlayer[idx].signal = val
        for layer in self._layers[1:]:
            for neuron in layer:
                neuron.propagate()

    def _backpropagate(self, targets):
        for idx, val in enumerate(targets):
            self.outputlayer[idx].backpropagate1(val)
        for layer in self.hiddenlayers[::-1]:
            for neuron in layer:
                neuron.backpropagate1()

        for neuron in self.outputlayer:
            neuron.backpropagate2()
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


def sigmoid(x):
    try:
        val = 1/(1 + math.exp(-x))
    except OverflowError:
        val = 0.
    return val


class PropagatingNeuron(object):
    def __init__(self, learningrate, bias):
        self._forward = []
        self._back = []
        self._signal = 0
        self._errsignal = 0
        self._learningrate = learningrate
        self._bias = bias

    def connect_forward(self, other, weight):
        conn = Connection(self, other, weight)
        self._forward.append(conn)
        other._back.append(conn)

    @property
    def signal(self):
        return self._signal

    @property
    def errsignal(self):
        return self._errsignal

    def propagate(self):
        thesum = sum((b.weight * b.signal for b in self._back))
        self._signal = sigmoid(thesum + self._bias)

    def backpropagate1(self):
        raise NotImplementedError()

    def backpropagate2(self):
        for b in self._back:
            b.weight -= self._learningrate * b.signal * self._errsignal
        self._bias -= self._learningrate * self._errsignal


class OutputNeuron(PropagatingNeuron):
    def backpropagate1(self, target):
        s = self.signal
        self._errsignal = s * (1 - s) * (s - target)


class HiddenNeuron(PropagatingNeuron):
    def backpropagate1(self):
        errsum = sum((f.weight * f.errsignal for f in self._forward))
        s = self.signal
        self._errsignal = s * (1 - s) * errsum


class InputNeuron(object):
    def __init__(self):
        self._forward = []
        self._signal = 0

    @property
    def signal(self):
        return self._signal

    @signal.setter
    def signal(self, value):
        self._signal = value

    def connect_forward(self, other, weight):
        conn = Connection(self, other, weight)
        self._forward.append(conn)
        other._back.append(conn)


class Connection(object):
    def __init__(self, back, forward, weight):
        self._forward = forward
        self._back = back
        self._weight = weight

    @property
    def weight(self):
        return self._weight

    @weight.setter
    def weight(self, value):
        self._weight = value

    @property
    def signal(self):
        return self._back.signal

    @property
    def errsignal(self):
        return self._forward.errsignal
