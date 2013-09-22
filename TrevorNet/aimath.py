import math
import sys

def sigmoid(x):
    try:
        val = 1/(1 + math.exp(-x))
    except OverflowError:
        val = sys.float_info.max
    return val

def sigmoidprime(x):
    return (1 - sigmoid(x))*sigmoid(x)
