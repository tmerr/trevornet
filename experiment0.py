from collections import namedtuple
import math
import aimath


class FeedForwardNet:
    def __init__(self):
        pass

class InputLayer:
    def __init__(self, num_neurons):
        self._layer = Layer(num_neurons)
        self._neurons = [InputNeuron() for x in range(num_neurons)]

    @property
    def neurons(self):
        return self._neurons
    
    def connect_layer(self, other):
        for n in self._neurons:
            for m in other.neurons:
                _connect_neurons(n, m)

class Layer:
    def __init__(self, num_neurons, neuron_type):
        self._neurons = [neuron_type for x in range(num_neurons)]

    @property
    def neurons(self):
        return self._neurons

    def connect_layer(self, other):
        for n in self._neurons:
            for m in other.neurons:
                _connect_neurons(n, m)

    def _connect_neurons(self, neuron1, neuron2, weight=1):
        connection = Connection(neuron1, neuron2, weight)
        neuron1.attach_forward(connection)
        neuron2.attach_back(connection)

    def propagate(self):
        for n in self._neurons:
            n.propagate()

    def backpropagate1(self):
        for n in self._neurons:
            n.backpropagate1()

    def backpropagate2(self):
        for n in self._neurons:
            n.backpropagate2()

class InputNeuron:
    def __init__(self):
        self._forward = []
        self._back = []
        self._signal = 0

    @property
    def signal(self):
        return self._signal

    @signal.setter
    def signal(self, value):
        self._signal = value

    def attach_forward(self, connection):
        self._forward.append(connection)

class OutputNeuron:
    def __init__(self, errfunction):
        '''errfunction params: signal. returns error'''
        self._forward = []
        self._back = []
        self._signal = 0
        self._errsignal = 0
        self._learningrate = 1

    @property
    def signal(self):
        return self._signal

    @property
    def errsignal(self):
        return self._errsignal

    def attach_back(self, connection):
        self.back.append(connection)

    def propagate(self):
        thesum = 0
        for b in self._back:
            thesum += b.weight * b.signal
        self._signal = aimath.sigmoid(thesum)

    def backpropagate1(self):
        self._errsignal = self.errfunction(signal)

    def backpropagate2(self):
        a = self._learningrate
        b = self._errsignal
        c = aimath.sigmoidprime(self._signal)
        dweight = a * c

        for b in back:
            b.weight += dweight

class Neuron:
    def __init__(self):
        self._forward = []
        self._back = []
        self._signal = 0
        self._errsignal = 0
        self._learningrate = 1

    @property
    def signal(self):
        return self._signal

    @property
    def errsignal(self):
        return self._errsignal

    def attach_forward(self, connection):
        self._forward.append(connection)

    def attach_back(self, connection):
        self._back.append(connection)

    def propagate(self):
        thesum = 0
        for b in self._back:
            thesum += b.weight * b.signal
        self._signal = aimath.sigmoid(thesum)

    def backpropagate1(self):
        thesum = 0
        for f in self._forward:
            thesum += f.weight * f.errsignal
        self._errsignal = thesum

    def backpropogate2(self):
        a = self._learningrate
        b = self._errsignal
        c = aimath.sigmoidprime(self._signal)
        dweight = a * b * c

        for b in back:
            b.weight += dweight

class Connection:
    def __init__(self, forward, back, weight):
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
        return back.signal

    @property
    def errsignal(self):
        return forward.errsignal
