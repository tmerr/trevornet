#! /usr/bin/env python3
"""
Parse an IDX file like the one used in the MNIST handwritten digit database.
A description of the format is on this page: http://yann.lecun.com/exdb/mnist/
"""

import struct

def _is_sequence(seq):
    return hasattr(seq, '__getitem__') and not hasattr(seq, 'strip')

def find_dimensions(seq):
    """Find the dimensions sizes of the sequence by recursively finding deeper
    sequences. A sequence with maximum indexes seq[3][5] uses dimension
    sizes (3, 5). The length of this tuple is the number of dimensions.

    Params:
        lst: The sequence to find the dimensions of.

    Return:
        A list of dimension sizes
    """
    sizes = []
    if _is_sequence(seq):
        sizes.append(len(seq))
        sizes += find_dimensions(seq[0])
    return sizes

def _build_magic_number(seq, typestr):
    typebyte = {}
    typebyte['B'] = b'\x08'
    typebyte['b'] = b'\x09'
    typebyte['h'] = b'\x0B'
    typebyte['i'] = b'\x0C'
    typebyte['f'] = b'\x0D'
    typebyte['d'] = b'\x0E'
    dimension_sizes = find_dimensions(seq)
    num_dimensions = len(dimension_sizes)
    dimensionbyte = num_dimensions.to_bytes(1, 'big')

    header = b'\x00\x00' + typebyte[typestr] + dimensionbyte
    return header

def _build_dimension_sizes(seq):
    bytez = bytearray()
    
    dims = find_dimensions(seq)
    for size in dims:
        bytez += (struct.pack('>i', size))
    return bytez

def _build_data(seq, typecode):
    if not _is_sequence(seq):
        formatstring = '>{0}'.format(typecode)
        return struct.pack(formatstring, seq)

    data = bytearray()
    if _is_sequence(seq):
        for s in seq:
            data.extend(_build_data(s, typecode))

    return data

def list_to_idx(lst, typecode):
    """Convert an n dimensional list into IDX bytes.

    Params:
        lst: The n dimensional list to convert and write.
        typecode: The C type the data should be stored as.
            B: unsigned byte
            b: signed byte
            h: short (2 bytes)
            i: int (4 bytes)
            f: float (4 bytes)
            d: double (8 bytes)
    """
    magicnumber = _build_magic_number(lst, typecode)
    dimension_sizes = _build_dimension_sizes(lst)
    data = _build_data(lst, typecode)

    return magicnumber + dimension_sizes + data

def idx_to_list(thebytes):
    #TODO: Complete this
    """Convert the IDX bytes to an n dimensional list.
    
    Params:
        fpath: The file path to read from.
    """
    if not thebytes[0] == '\x00' and thebytes[1] == '\x00':
        raise IOError("IDX file should start with two 0 bytes")
