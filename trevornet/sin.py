#!python3

from trevornet.feedforward import FeedForwardNet
import random
import math

def sin():
    """A normalized sin: f(x) = .5*sin(x)+.5"""
    net = FeedForwardNet.fromfuncs((1, 20, 1), .25)

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
    sin()
