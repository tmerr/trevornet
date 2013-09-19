"""
These classes represents layers of a neural net. They serve as container for
neurons, supporting iteration, indexing, and len. To set them up call
connect_forward between layers.

The backpropagation algorithm can then be performed on these neurons through
propagate, backpropagate1, and backpropagate2.
"""

import neurons
import random

class _Layer(object):
    def __init__(self, neurons):
        """Initialize

        Params:
            neurons: The neurons that make up the layer.
        """
        self._neurons = neurons

    def connect_forward(self, other, weight_funct=None):
        """Connect every neuron of this layer forward to the other layer. The
        direction does matter.

        Params:
            other: The layer to connect to.
            weight_funct: An optional function that chooses the initial weight
                of each connection. Defaults to a random value from -1 to 1.
        """

        if weight_funct is None:
            weight_funct = lambda x: random.random()*2 - 1

        for n in self._neurons:
            for m in other.neurons:
                n.connect_forward(m, weight_funct())

    def __len__(self):
        return len(neurons)

    def __getitem__(self, key):
        return __getitem__(neurons, key)

    def __iter__(self):
        return __iter__(neurons)

class _PropagatingLayer(_Layer):
    """An abstract layer. When inheriting you must call this __init__."""

    def propagate(self):
        """Update this layer's signals using the signals from the layer behind."""
        for n in self._neurons:
            n.propagate()

    def backpropagate1(self):
        """Update this layer's error signals using error signals from the layer
        in front."""
        raise NotImplementedError()

    def backpropagate2(self):
        """Use the error signals from backpropagate1 to update the weights from
        this layer to the one behind it.
        """
        for n in self._neurons:
            n.backpropagate2()

class InputLayer(_Layer):
    def __init__(self, num_neurons):
        n = [neurons.InputNeuron() for _ in range(num_neurons)]
        super(InputLayer, self).__init__(n)

    @property
    def inputs(self):
        """The input signals"""
        return [neuron.signal for neuron in self]

    @inputs.setter
    def inputs(self, values):
        """
        Params:
            values: A sequence of floats that is mapped to each neuron.
        """
        for neuron, value in zip(self, values):
            neuron.signal= value

class OutputLayer(_PropagatingLayer):
    def __init__(self, num_neurons): 
        n = [neurons.OutputNeuron() for _ in range(num_neurons)]
        super(OutputLayer, self).__init__(n)

    @property
    def outputs(self):
        return [neuron.signal for neuron in self]

    def backpropagate1(self, targets):
        """
        Params:
        targets: A sequence of target values that correspond to each neuron in
               the output layer.
        """
        if len(targets) != len(self):
            raise ValueError("num floats must be same as num neurons")
        for neuron, target in zip(self, targets):
            n.backpropagate1(target)

class HiddenLayer(_PropagatingLayer):
    def __init__(self, num_neurons, learningrate):
        n = [neurons.HiddenNeuron() for _ in range(num_neurons)]
        super(HiddenLayer, self).__init__(neurons)

    def backpropagate1(self):
        for n in self._neurons:
            n.backpropagate1()
