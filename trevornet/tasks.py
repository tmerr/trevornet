#! /usr/bin/env python3

"""
Some example tasks for neural nets. They aren't fit to be unit tests, because
they're slow, and I have only vague expectations of their behavior.
"""

from trevornet.nets.pynet.feedforward import PyFeedForwardNet
from trevornet import idx
import random
import math
import time
import os

def XOR():
    """Exclusive or"""
    net = PyFeedForwardNet((2, 2, 1), .5)

    domain = ((1,1), (1,0), (0,1), (0,0))
    rng = ((0,), (1,), (1,), (0,))
    for i in range(10000):
        r = random.randrange(4)
        net.train(domain[r], rng[r])

    for d in domain:
        print('{0} => {1}'.format(d, net.predict(d)))

def sin():
    """A normalized sin: f(x) = .5*sin(x)+.5"""
    net = PyFeedForwardNet((1, 50, 1), .2)

    for i in range(10000):
        if i%1000 == 0:
            print('progress: {0}%'.format(i/100))
        x = random.random()*2*math.pi
        y = .5*math.sin(x)+.5
        net.train((x,), (y,))

    for i in range(20):
        x = .05*i*2*math.pi
        print('{0} => {1}'.format((x,), net.predict((x,))))

def OCR(maxtime = None):
    """Optical character recognition

    Params:
        maxtime: The maximum time to train in seconds
    """
    traindatapath = os.path.join('trevornet', 'dataset', 'train-images.idx3-ubyte')
    trainlabelpath = os.path.join('trevornet', 'dataset', 'train-labels.idx1-ubyte')

    print("Parsing training data...")
    with open(traindatapath, 'rb') as f:
        traindata = idx.idx_to_list(f.read())

    print("Parsing training labels..")
    with open(trainlabelpath, 'rb') as f:
        trainlabels = idx.idx_to_list(f.read())

    print("Creating 784, 200, 50, 10 net")
    net = PyFeedForwardNet((28*28, 25, 25, 10), .02)

    print("Training...")
    start_time = time.time()

    numimages = len(traindata)
    count = 0
    for image, targetint in zip(traindata, trainlabels):
        count += 1

        pixels = [row for col in image for row in col]
        pixels = [float(x)/256 for x in pixels]
        targetlist = [0 for i in range(10)]
        targetlist[targetint] = 1

        assert len(pixels) == 28*28
        assert len(targetlist) == 10
        net.train(pixels, targetlist)

        print("Training image {0} of {1}".format(count, numimages))

        if maxtime and (time.time() - start_time > maxtime):
            print("Training stopped at {0} of {1} images".format(count, numimages))
            break

    del traindata
    del trainlabels

    testdatapath = os.path.join('trevornet', 'dataset', 't10k-images.idx3-ubyte')
    testlabelpath = os.path.join('trevornet',' dataset', 't10k-labels.idx1-ubyte')

    print("Parsing test data...")
    with open(testdatapath, 'rb') as f:
        testdata = idx.idx_to_list(f.read())

    print("Parsing test labels...")
    with open(testlabelpath, 'rb') as f:
        testlabels = idx.idx_to_list(f.read())

    successes, failures = 0, 0
    for image, targetint in zip(testdata, testlabels):
        pixels = [row for col in image for row in col]
        pixels = [float(x)/256 for x in pixels]
        targetlist = [0 for i in range(10)]
        targetlist[targetint] = 1

        outputs = net.predict(pixels)
        outputint = max(range(len(outputs)), key=outputs.__getitem__)

        if outputint == targetint:
            successes += 1
        else:
            failures += 1

        total = successes + failures
        thestr = "Predictions: {0}, Success rate: {1}%".format(
            total, (successes/total)*100
        )
        print(thestr)#, end='\r')

    print("Prediction done!")
    thestr = "Predictions: {0}, Success rate: {1}%".format(
        total, (successes/total)*100
    )
    print(thestr)

if __name__ == '__main__':
    #XOR()
    #sin()
    OCR()
