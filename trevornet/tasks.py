import nets
import random
import math
import idx
import time

def XOR():
    """Exclusive or"""
    net = nets.FeedForwardNet((2, 2, 1), .5)
    
    domain = ((1,1), (1,0), (0,1), (0,0))
    rng = ((0,), (1,), (1,), (0,))
    for i in range(10000):
        r = random.randrange(4)
        net.train(domain[r], rng[r])

    for d in domain:
        print('{0} => {1}'.format(d, net.predict(d)))

def sin():
    """A normalized sin: f(x) = .5*sin(x)+.5"""
    net = nets.FeedForwardNet((1, 50, 1), .2)

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
    traindatapath = 'dataset\\train-images.idx3-ubyte'
    trainlabelpath = 'dataset\\train-labels.idx1-ubyte'

    print("Parsing training data...")
    with open(traindatapath, 'rb') as f:
        traindata = idx.idx_to_list(f.read())
    
    print("Parsing training labels..")
    with open(trainlabelpath, 'rb') as f:
        trainlabels = idx.idx_to_list(f.read())

    print("Creating 784, 200, 50, 10 net")
    net = nets.FeedForwardNet((28*28, 25, 25, 10), 10)

    print("Training...")
    start_time = time.time()

    numimages = len(traindata)
    count = 0
    for image, targetint in zip(traindata, trainlabels):
        count += 1

        pixels = [row for col in image for row in col]
        targetlist = [0 for i in range(10)]
        targetlist[targetint] = 1

        assert len(pixels) == 28*28
        assert len(targetlist) == 10
        net.train(pixels, targetlist)

        if time.time() - start_time > maxtime:
            print("Training stopped at {0} of {1} images".format(count, numimages))
            break

    del traindata
    del trainlabels
    
    testdatapath = 'dataset\\t10k-images.idx3-ubyte'
    testlabelpath = 'dataset\\t10k-labels.idx1-ubyte'

    print("Parsing test data...")
    with open(testdatapath, 'rb') as f:
        testdata = idx.idx_to_list(f.read())

    print("Parsing test labels...")
    with open(testlabelpath, 'rb') as f:
        testlabels = idx.idx_to_list(f.read())

    successes, failures = 0, 0
    for image, targetint in zip(testdata, testlabels):
        pixels = [row for col in image for row in col]
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
        print(thestr, end='\r')

    print("Prediction done!")
    thestr = "Predictions: {0}, Success rate: {1}%".format(
        total, (successes/total)*100
    )
    print(thestr)
    

if __name__ == '__main__':
    XOR()
    #sin()
    OCR(120)
