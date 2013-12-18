import math
import sys

def sigmoid(x):
    try:
        val = 1/(1 + math.exp(-x))
    except OverflowError:
        val = 0.
    return val

def sigmoidprime(x):
    return (1 - sigmoid(x))*sigmoid(x)
