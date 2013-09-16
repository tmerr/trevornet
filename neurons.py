import math
import aimath

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
    def __init__(self):
        self._neuron = Neuron()

    @property
    def signal(self):
        return self._neuron._signal

    @property
    def errsignal(self):
        return self._neuron._errsignal

    def attach_back(self, connection):
        self._neuron.attach_back(connection)

    def propagate(self):
        self._neuron.propagate()

    def backpropagate1(self, target):
        self._neuron._errsignal = target - self.signal
    
    def backpropagate2(self):
        self._neuron.backpropagate2

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

        for b in self._back:
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
        return self._back.signal

    @property
    def errsignal(self):
        return forward.errsignal
