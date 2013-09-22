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

def idx_to_list(bytez):
    """Convert the IDX bytes to nested lists
    
    Params:
        bytez: The IDX file bytes.
    """
    # byte 0: 0
    # byte 1: 0
    if not (bytez[0] == 0 and bytez[1] == 0):
        raise IOError("IDX file should start with two 0 bytes")

    # byte 2: The number of dimensions
    # byte 3: The type code
    typebyte = bytez[2]
    numdims = bytez[3]

    # 4 bytes for each dimension: size of dimensions
    fmtstring = '>' + 'i'*numdims
    dimension_sizes = struct.unpack(fmtstring, bytez[4:4+4*numdims])
    
    # Rest of the data starts here
    startoffset = 4 + 4*numdims

    typedata = {}
    typedata[8] = 'B', 1
    typedata[9] = 'b', 1
    typedata[11] = 'h', 2
    typedata[12] = 'i', 4
    typedata[13] = 'f', 4
    typedata[14] = 'd', 8

    typecode = typedata[typebyte][0]
    elementlength = typedata[typebyte][1]
    num_elements = (len(bytez)-startoffset)//elementlength

    formatstr = ''.join(('>', typecode*num_elements))
    flatlist = struct.unpack_from(formatstr, bytez, startoffset)

    def _recursive(inputlst, dimsizes):
        """Recursively split the flat list into chunks and merge them back into a
        nested list structure."""
        if len(dimsizes) == 1:
            return list(inputlst)

        outerlist = []

        chunksize = len(inputlst)//dimsizes[0]
        for i in range(0, len(inputlst), chunksize):
            chunk = inputlst[i:i+chunksize]
            innerlist = _recursive(chunk, dimsizes[1:])
            outerlist.append(innerlist)
        
        return outerlist

    return _recursive(flatlist, dimension_sizes)
