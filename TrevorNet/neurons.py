import math
import aimath
import random

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
        thesum = sum([b.weight * b.signal for b in self._back])
        self._signal = aimath.sigmoid(thesum + self._bias)

    def backpropagate1(self):
        raise NotImplementedError()

    def backpropagate2(self):
        for b in self._back:
            b.weight -= self._learningrate * b.signal * self._errsignal
        self._bias -= self._learningrate * self._errsignal

class OutputNeuron(PropagatingNeuron):
    def backpropagate1(self, target):
        s = self.signal
        self._errsignal = aimath.sigmoidprime(s) * (s - target)

class HiddenNeuron(PropagatingNeuron):
    def backpropagate1(self):
        errsum = sum([f.weight * f.errsignal for f in self._forward])
        self._errsignal = aimath.sigmoidprime(self._signal) * errsum

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
