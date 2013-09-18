import math
import aimath
import random

class InputNeuron(object):
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

class OutputNeuron(object):
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
        '''Update error signal based on the given target value'''
        s = self.signal
        self._neuron._errsignal = aimath.sigmoidprime(s) * (s - target)
 
    def backpropagate2(self):
        self._neuron.backpropagate2()

class Neuron(object):
    def __init__(self):
        self._forward = []
        self._back = []
        self._signal = 0
        self._errsignal = 0
        self._learningrate = .2
        self._bias = 1

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
        thesum = sum([b.weight * b.signal for b in self._back])
        self._signal = aimath.sigmoid(thesum + self._bias)

    def backpropagate1(self):
        '''Update error signal'''
        errsum = sum([f.weight * f.errsignal for f in self._forward])
        self._errsignal = aimath.sigmoidprime(self._signal) * errsum

    def backpropagate2(self):
        '''Adjust weights and bias'''
        z = self._learningrate * self._errsignal
        for b in self._back:
            tmp = b.weight
            b.weight -= z * b.signal

        self._bias -= self._learningrate * self._errsignal

class Connection(object):
    def __init__(self, back, forward, weight):
        self._forward = forward
        self._back = back
        self._weight = random.random()

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
