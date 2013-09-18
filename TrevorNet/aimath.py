import math

def sigmoid(x):
    return 1/(1 + math.exp(-x))

def sigmoidprime(x):
    return (1 - sigmoid(x))*sigmoid(x)
