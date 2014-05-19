#! /usr/bin/env python3

"""
Some example tasks for neural nets. They aren't fit to be unit tests, because
they're slow, and I have only vague expectations of their behavior.
"""

from trevornet.nets.feedforward import PyFeedForwardNet
from trevornet.ocr import Ocr
import random
import math
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
        c = Ocr.fromrandom()
        c.train()
        c.tofile('theocrnet.dat')
        print('Saved trained net to theocrnet.dat!')
        c.test()
    else:
        print("Invalid task: {0}".format(args.task))
