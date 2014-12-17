#!python3

'''
An implementation of a markov chain used for text generation.

Just pass a file name as an argument and it should load it up, build a markov
chain with a state for each word(s), and start walking through the chain,
writing incoherent text to the terminal.
'''

import string
import random
import time
import sys
import argparse


asciiset = set(string.ascii_letters)
asciiset.add(' ')
asciiset.add('.')
def strip2ascii(txt):
    return ''.join([ch for ch in txt if ch in asciiset])


def tokenize(fname):
    '''
    Generate tokens defined by
    - Sequences of characters that aren't spaces
    - Periods

    For example, 'This is a test. Ok.' => ('This', 'is', 'a', 'test', '.', 'Ok, '.')
    '''
    with open(fname, 'r') as f:
        for line in f:
            stripped = strip2ascii(line)
            for word in stripped.split():
                
                if word[-1] == '.':
                    yield word[:-1]
                    yield '.'
                else:
                    yield word


def buildtransitionmap(tokens, order):
    dct = {}
    prev = ('',)*order

    for token in tokens:
        if prev in dct:
            dct[prev].append(token)
        else:
            dct[prev] = [token]

        prev = prev[1:]+(token,)
        
    return dct


def walk(transmap, prev=None):
    if prev == None:
        prev = random.choice(list(transmap.keys()))

    while True:
        if not prev in transmap:
            prev = random.choice(list(transmap.keys()))

        word = random.choice(transmap[prev])
        yield word
        prev = prev[1:]+(word,)


def finiteramble(fname, order, n):
    '''Write n words to the terminal by walking through the markov chain'''
    transmap = buildtransitionmap(tokenize(fname), order)
    for i, word in enumerate(walk(transmap)):
        if i == n:
            break
        print(word, end=' ')


def eternalramble(fname, order):
    '''Endlessly write words to the terminal by walking through the markov chain'''
    transmap = buildtransitionmap(tokenize(fname), order)
    for word in walk(transmap):
        print(word, end=' ')
        sys.stdout.flush()
        time.sleep(0.25)


def launch():
    desctxt = 'a markov chain based text generator.'
    parser = argparse.ArgumentParser(description=desctxt)
    parser.add_argument('filename', type=str,
            help= 'the filename of the text to base the markov chain on')
    parser.add_argument('order', type=int,
            help= 'how many consecutive words make up each state (2 works well)')
    parser.add_argument('n', type=int, nargs='?', default=None,
            help= 'the number of words to output. if omitted there will be no limit.')

    args = parser.parse_args()

    if (args.n is None):
        eternalramble(args.filename, args.order)
    else:
        finiteramble(args.filename, args.order, args.n)


if __name__ == '__main__':
    launch()
