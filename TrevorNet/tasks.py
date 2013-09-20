import nets
import random
import math

def XOR():
    '''Exclusive or'''
    net = nets.FeedForwardNet((2, 2, 1), .5)
    
    domain = ((1,1), (1,0), (0,1), (0,0))
    rng = ((0,), (1,), (1,), (0,))
    for i in range(10000):
        r = random.randrange(4)
        net.train(domain[r], rng[r])

    for d in domain:
        print('{0} => {1}'.format(d, net.predict(d)))

def sin():
    '''A normalized sin: f(x) = .5*sin(x)+.5'''
    net = nets.FeedForwardNet((1, 50, 1), .2)

    for i in range(10000):
        verbose = False
        if i%1000 == 0:
            print('progress: {0}%'.format(i/100))
            verbose = True
        x = random.random()*2*math.pi
        y = .5*math.sin(x)+.5
        net.train((x,), (y,))

    for i in range(20):
        x = .05*i*2*math.pi
        print('{0} => {1}'.format((x,), net.predict((x,))))

if __name__ == '__main__':
    XOR()
    sin()
