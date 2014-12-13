#!python3

import string
import random
import time
import sys


'''
This is an implementation of a markov chain used for text generation.
Just pass a file name as an argument and it should load it up, build a markov
chain with a state for each word(s), and start walking through the chain, writing
incoherent text to the terminal.
'''


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


def transition(word, transmap):
    return random.choice(transmap[word])


def eternalramble(fname, order):
    '''
    Walk through the markov chain printing out words to the terminal one at a time
    '''
    transmap = buildtransitionmap(tokenize(fname), order)
    prev = random.choice(list(transmap.keys()))
    while True:
        word = transition(prev, transmap)
        print(word, end=' ')
        prev  = prev[1:]+(word,)
        sys.stdout.flush()
        time.sleep(0.25)


def printusage():
    print('Usage: markov filename order')
    print('  filename: the filename of the text to base the markov chain on.')
    print('  order: how many consecutive words make up each state (2 works well)')


def launch():
    if len(sys.argv) != 3:
        printusage()
        return

    try:
        order = int(sys.argv[2])
    except:
        printusage()
        return

    eternalramble(sys.argv[1], order)


if __name__ == '__main__':
    launch()
