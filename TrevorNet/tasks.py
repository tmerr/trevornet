import nets
import random
import math

def XOR():
    '''Exclusive or'''
    net = nets.FeedForwardNet2(2, 3, 1)
    
    domain = ((1,1), (1,-1), (-1,1), (-1,-1))
    rng = ((0,), (1,), (1,), (0,))
    for i in range(100000):
        r = random.randrange(4)
        net.train(domain[r], rng[r])

    for d in domain:
        print('{0} => {1}'.format(d, net.predict(d)))

def sin():
    '''A normalized sin: f(x) = .5*sin(x)+.5'''
    net = nets.FeedForwardNet2(1, 50, 1)

    for i in range(300000):
        verbose = False
        if i%10000 == 0:
            print('progress: {0}%'.format(i/3000))
            verbose = True
        x = random.random()*2*math.pi
        y = .5*math.sin(x)+.5
        net.train((x,), (y,), verbose)

    for i in range(20):
        x = .05*i*2*math.pi
        print('{0} => {1}'.format((x,), net.predict((x,))))

if __name__ == '__main__':
    XOR()
    sin()
