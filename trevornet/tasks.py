#! /usr/bin/env python3

"""
Some example tasks for neural nets. They aren't fit to be unit tests, because
they're slow, and I have only vague expectations of their behavior.
"""

from trevornet.nets.feedforward import PyFeedForwardNet
from trevornet import idx
import random
import math
import time
import os
import argparse

def xor():
    """Exclusive or"""
    net = PyFeedForwardNet.fromfuncs((2, 2, 1), .5)

    domain = ((1, 1), (1, 0), (0, 1), (0, 0))
    rng = ((0,), (1,), (1,), (0,))
    for i in range(10000):
        r = random.randrange(4)
        net.train(domain[r], rng[r])

    for d in domain:
        print('{0} => {1}'.format(d, net.predict(d)))


def sin():
    """A normalized sin: f(x) = .5*sin(x)+.5"""
    net = PyFeedForwardNet.fromfuncs((1, 20, 1), .25)

    for i in range(50000):
        if i % 5000 == 0:
            print('progress: {0}%'.format(i/500))
        x = random.random()*4*math.pi
        y = .5*math.sin(x)+.5
        net.train((x,), (y,))

    print('\ninput\toutput\tanswer\terror')
    sumoferrors = 0
    for i in range(20):
        x = .05*i*4*math.pi
        prediction = net.predict((x,))[0]
        answer = .5*math.sin(x)+.5
        error = abs(answer-prediction)*100
        print('{0:.4f}\t{1:.4f}\t{2:.4f}\t{3:.4f}%'.format(x, prediction, answer, error))
        sumoferrors += error
    avgerror = sumoferrors / 20
    print('\naverage error: {0}%\n'.format(avgerror))


def ocr(maxtime=None):
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
    net = PyFeedForwardNet.fromfuncs((28*28, 25, 25, 10), .02)

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
        print(thestr)

    print("Prediction done!")
    thestr = "Predictions: {0}, Success rate: {1}%".format(
        total, (successes/total)*100
    )
    print(thestr)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test some example neural net tasks.')
    parser.add_argument('task', type=str,
                        help='the task to run: xor, sin or ocr')
    args = parser.parse_args()
    if args.task == 'xor':
        xor()
    elif args.task == 'sin':
        sin()
    elif args.task == 'ocr':
        ocr()
    else:
        print("Invalid task: {0}".format(args.task))
