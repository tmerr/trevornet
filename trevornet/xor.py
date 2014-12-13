from trevornet.feedforward import FeedForwardNet
import random

def xor():
    """Exclusive or"""
    net = FeedForwardNet.fromfuncs((2, 2, 1), .5)

    domain = ((1, 1), (1, 0), (0, 1), (0, 0))
    rng = ((0,), (1,), (1,), (0,))
    for i in range(10000):
        r = random.randrange(4)
        net.train(domain[r], rng[r])

    for d in domain:
        print('{0} => {1}'.format(d, net.predict(d)))

if __name__ == '__main__':
    xor()
