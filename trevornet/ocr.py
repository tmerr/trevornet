from trevornet import idx
from trevornet.nets.feedforward import PyFeedForwardNet
import os
from tkinter import *
from tkinter import filedialog
import multiprocessing
import queue
import math
import copy


DEBUG = False


def read_point_bilinear(x, y, image):
    '''
    Read a point in the image of [[number]] using bilinear interpolation.
    Algorithm is from the wikipedia article, translated to python.

    Params:
        x: The real number x coordinate
        y: The real number y coordinate
        image: A collection of real numbers that is indexable [y][x]

    Return:
        A real number that is the interpolated value of the pixel at the
        given x and y.
    '''

    if len(image) == 1:
        y = math.floor(y)

    if len(image[0]) == 1:
        x = math.floor(x)

    if not (0 <= y <= len(image)-1):
        raise IndexError('index y={0} not within height {1}'.format(y, len(image)))

    if not (0 <= x <= len(image[0])-1):
        raise IndexError('index x={0} not within width {1}'.format(x, len(image[0])))

    if float(x).is_integer() and float(y).is_integer():
        return image[int(y)][int(x)]

    x1 = math.floor(x)
    x2 = math.ceil(x)
    y1 = math.floor(y)
    y2 = math.ceil(y)

    out11 = read_point_bilinear(x1, y1, image)
    out12 = read_point_bilinear(x1, y2, image)
    out21 = read_point_bilinear(x2, y1, image)
    out22 = read_point_bilinear(x2, y2, image)

    out = \
        out11 * (1-x) * (1-y) + \
        out21 * x * (1-y) + \
        out12 * (1-x) * y + \
        out22 * x*y

    return out


def resize(width, height, image):
    '''
    Resize an image of [[number]] to the given size using bilinear
    interpolation. The initial width and height are inferred by using
    len on the image.

    Params:
        width: The new width in pixels (a natural number)
        height: The new height in pixels (a natural number)
        image: A collection of real numbers that is indexable [y][x]

    Return:
        A new image resized to the given width and height.
    '''

    if width < 1 or not float(width).is_integer():
        raise ValueError('width={0} must be a natural number.'.format(height))

    if height < 1 or not float(height).is_integer():
        raise ValueError('height={0} must be a natural number.'.format(height))

    heighti = len(image)
    widthi = len(image[0])

    if (width == widthi and height == heighti):
        return copy.deepcopy(image)

    xinterval = widthi / width
    yinterval = heighti / height

    xstart = xinterval/2
    ystart = yinterval/2

    squishx = (widthi-1)/widthi
    squishy = (heighti-1)/heighti

    xs = [(xstart + x*xinterval)*squishx for x in range(width)]
    ys = [(ystart + y*yinterval)*squishy for y in range(height)]

    if (DEBUG):
        thestr = 'Resizing size {0}, {1} to size {2}, {3}, xinterval {4}, yinterval {5}, xstart {6}, ystart {7}'
        print(thestr.format(widthi, heighti, width, height, xinterval, yinterval, xstart, ystart))
        print(xs)
        print(ys)

    return [[read_point_bilinear(x, y, image) for x in xs] for y in ys]


def get_occupied_region(image):
    '''
    Returns the region of the image as that is occupied by content. The goal is
    to distinguish what is whitespace and what is not.

    Return:
        The region is a tuple (x, y, width, height) within the image.
    '''

    x1 = len(image[0]) - 1
    y1 = len(image) - 1
    x2 = 0
    y2 = 0

    for y_idx, line in enumerate(image):
        for x_idx, px in enumerate(line):
            if px > 0:
                x1 = min(x1, x_idx)
                y1 = min(y1, y_idx)
                x2 = max(x2, x_idx)
                y2 = max(y2, y_idx)

    xcount = max(x2 - x1 + 1, 0)
    ycount = max(y2 - y1 + 1, 0)

    return (x1, y1, xcount, ycount)


def crop(region, image):
    '''
    Crops the image to the given region.

    Params:
        region: The region as a tuple (x, y, width, height) to crop with.
        image: The image to crop.

    Return:
        The cropped image.
    '''

    (x, y, width, height) = region

    tooleft = x < 0
    tootop = y < 0
    tooright = x + width > len(image[0])
    toobottom = y + height > len(image)
    if tooleft or tootop or tooright or toobottom:
        errorstr = 'Crop region {0} must be within the image\'s width={1}, height={2}'
        raise IndexError(errorstr.format(region, len(image[0]), len(image)))

    if width == 0 or height == 0:
        return [[0]]

    return [[image[y+m][x+n] for n in range(width)] for m in range(height)]


def paste(source, onto, position):
    ontocopy = copy.deepcopy(onto)
    (startx, starty) = position
    for yi, line in enumerate(source):
        for xi, val in enumerate(line):
            ontocopy[starty + yi][startx + xi] = val
    return ontocopy


def upscale_to_aspect(size, aspect):
    '''
    Scale up a size to match an aspact ratio.

    Params:
        The size (width, height)
        The aspect ratio x/y

    Return:
        The scaled up size (width, height) where width and heights are integral.
    '''
    (width, height) = size
    size_aspect = width / height

    if size_aspect == aspect:
        return size
    elif size_aspect < aspect:
        return (round(height*aspect), height)
    elif size_aspect > aspect:
        return (width, round(width/aspect))


def bound(lower, upper, value):
    return max(lower, min(upper, value))


def size_normalize_contents(image):
    height = len(image)
    width = len(image[0])

    if (width == 0 or height == 0):
        return image

    occx, occy, occwidth, occheight = get_occupied_region(image)
    if (occwidth == 0 or occheight == 0):
        return image

    upscaledw, upscaledh = upscale_to_aspect((occwidth, occheight), width/height)
    diffw = upscaledw - occwidth
    diffh = upscaledh - occheight
    newx = int(bound(0, 27, occx - diffw/2))
    newy = int(bound(0, 27, occy - diffh/2))

    if newx + upscaledw > width:
        upscaledw = width - upscaledw

    if newy + upscaledh > height:
        upscaledh = height - upscaledh

    occupied2 = (newx, newy, upscaledw, upscaledh)

    _20by20 = resize(20, 20, crop(occupied2, image))
    whitespace = [[0 for x in range(28)] for y in range(28)]
    _28by28 = paste(_20by20, whitespace, (4, 4))
    return _28by28


class OcrPresentation(object):
    '''
    Lets the user modify self.pixels via a GUI by dragging left click.
    Points are automatically sent to the neural net which updates a label.
    '''
    def __init__(self):
        self.root = Tk()
        self.root.title("Character recognition")
        self.root.resizable(0, 0)

        frame = Frame(self.root)
        frame.pack()
        loadbtn = Button(frame, text='Load existing net', command=self.load)
        loadbtn.pack(side=LEFT)
        trainbtn = Button(frame, text='Create and train new net', command=self.trainnew)
        trainbtn.pack(side=LEFT)

        self.canvas = Canvas(self.root, bg="white", width=28*8, height=28*8)
        self.canvas.configure(cursor="crosshair")
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self.mouseheld)

        frame2 = Frame(self.root)
        frame2.pack()
        clearbtn = Button(frame2, text='Clear', command=self.clearpressed)
        clearbtn.pack(side=LEFT)
        self.label = Label(self.root, text='-1')
        self.label.pack()
        exit = Button(frame2, text='Exit', command=self.exitpressed)
        exit.pack(side=LEFT)

        self.pixels = [[0 for x in range(28)] for y in range(28)]

        self.worker = None

        self.root.after(0, self.update())
        self.root.mainloop()

    def startworker(self, fname):
        if not self.worker is None:
            self.worker.terminate()
        self.output_queue = multiprocessing.Queue(1)
        self.input_queue = multiprocessing.Queue(1)
        self.worker = multiprocessing.Process(target=OcrWorker,
                                              args=(fname, self.output_queue, self.input_queue),
                                              daemon=True)
        self.worker.start()
        self.output_queue.put_nowait(self.pixels)

    def trainnew(self):
        fname = filedialog.asksaveasfilename(parent=self.root, title='Choose where to save the trained net to')
        net = Ocr.fromrandom()
        net.train()
        net.tofile(fname)
        self.startworker(fname)

    def load(self):
        fname = filedialog.askopenfilename(parent=self.root, title='Choose a file to neural net file to use')
        self.startworker(fname)

    def exitpressed(self):
        self.root.quit()
        if self.worker is not None:
            self.worker.terminate()

    def clearpressed(self):
        self.pixels = [[0 for x in range(28)] for y in range(28)]
        self.canvas.create_rectangle(0, 0, 28*8, 28*8, fill='white', outline='white')

    def mouseheld(self, event):
        gridx, gridy = event.x//8, event.y//8
        if 0 < gridx > 27 or 0 < gridy > 27:
            return
        darkness = min(self.pixels[gridy][gridx] + 1, 1)
        self.pixels[gridy][gridx] = darkness

    def draw_image(self, image, fmt):
        hexdigits = '0123456789abcdef'
        for gridy, line in enumerate(image):
            for gridx, val in enumerate(line):
                screenx, screeny = gridx*8, gridy*8
                digit = 15-int(val*15)
                bounded_digit = bound(0, 15, digit)
                if bounded_digit != 15:
                    fillcolor = fmt.format(hexdigits[bounded_digit])
                    self.canvas.create_rectangle(screenx, screeny, screenx+7, screeny+7,
                                                fill=fillcolor, outline=fillcolor)

    def draw(self):
        self.canvas.delete('all')
        normalized = size_normalize_contents(self.pixels)
        self.draw_image(normalized, '#{0}{0}f')
        self.draw_image(self.pixels, '#{0}{0}{0}')

    def update(self):
        # Try to push an image out to the net, and check for any results that
        # came back.
        self.draw()
        if self.worker is not None:
            try:
                outputlayer = self.input_queue.get_nowait()
                highest = max(outputlayer)
                normalizedoutputs = [x/highest for x in outputlayer]
                outputbars = ['=' * int(10*x) for x in normalizedoutputs]

                labeltext = []
                labeltext.append('Output layer')
                for idx, bar in enumerate(outputbars):
                    labeltext.append('{0}: {1}'.format(idx, bar))

                labeltext.append('')
                prediction = max(range(len(outputlayer)), key=outputlayer.__getitem__)
                labeltext.append('Prediction: {0}'.format(prediction))

                self.label.config(text='\n'.join(labeltext), justify=LEFT)
                self.output_queue.put_nowait(self.pixels)
            except queue.Empty:
                pass
            except queue.Full:
                print("Queue full, this shouldn't happen.")
        self.root.after(20, self.update)


class OcrWorker(object):
    def __init__(self, fname, input_queue, output_queue):
        self.net = PyFeedForwardNet.fromfile(fname)
        if len(self.net.inputlayer) != 28*28:
            raise ValueError('Expected a 28*28 input layer')
        if len(self.net.outputlayer) != 10:
            raise ValueError('Expected a size 10 output layer')

        self.input_queue = input_queue
        self.output_queue = output_queue
        self.loop()

    def loop(self):
        while True:
            image = self.input_queue.get()
            normalized = size_normalize_contents(image)
            flatpixels = [row for col in normalized for row in col]
            outputs = self.net.predict(flatpixels)
            self.output_queue.put(outputs)


class Ocr(object):
    '''
    Neural net with a 28*28 input layer and size 10 output layer with methods
    for training and testing using the MNIST database.
    '''

    def __init__(self, net):
        self._net = net

    @classmethod
    def fromfile(cls, fpath):
        net = PyFeedForwardNet.fromfile(fpath)
        if len(net.inputlayer) != 28*28:
            raise ValueError('Ocr net expected a 28*28 input layer')
        if len(net.outputlayer) != 10:
            raise ValueError('Ocr net expected a size 10 output layer')
        return cls(net)

    @classmethod
    def fromrandom(cls):
        net = PyFeedForwardNet.fromfuncs((28*28, 25, 25, 10), .05)
        return cls(net)

    def tofile(self, fname):
        savednet = str(self._net.tolist())
        with open(fname, 'w') as f:
            f.write(savednet)

    def train(self):
        traindatapath = os.path.join('trevornet', 'dataset', 'train-images.idx3-ubyte')
        trainlabelpath = os.path.join('trevornet', 'dataset', 'train-labels.idx1-ubyte')

        print("Parsing training data...")
        with open(traindatapath, 'rb') as f:
            traindata = idx.idx_to_list(f.read())

        print("Parsing training labels..")
        with open(trainlabelpath, 'rb') as f:
            trainlabels = idx.idx_to_list(f.read())

        print("Training...")

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
            self._net.train(pixels, targetlist)

            print("Training image {0} of {1}".format(count, numimages))

        del traindata
        del trainlabels

    def test(self):
        testdatapath = os.path.join('trevornet', 'dataset', 't10k-images.idx3-ubyte')
        testlabelpath = os.path.join('trevornet','dataset', 't10k-labels.idx1-ubyte')

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

            outputs = self._net.predict(pixels)
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
    OcrPresentation()
