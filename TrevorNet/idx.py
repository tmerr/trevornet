"""
Parse an IDX file like the one used in the MNIST handwritten digit database.
A description of the format is on this page: http://yann.lecun.com/exdb/mnist/
"""

def _find_dimensions(seq, curdepth=0):
    """Find the number of dimensions of the sequence by recursively finding
    deeper sequences. seq[0] uses 1 dimension, seq[0][0] uses 2 dimensions.

    Params:
        lst: The sequence to find the dimensions of.

    Return:
        The number of dimensions.
    """
    if hasattr(seq, "__getitem__") and not hasattr(seq, "strip"):
        curdepth = _find_dimensions(seq[0], curdepth + 1)
    return curdepth

def list_to_idx(lst, fname, typecode):
    """Convert an n dimensional list into an IDX file and write it at fname.
    
    Params:
        lst: The n dimensional list to convert and write.
        fname: The filename to write to.
        typecode: The type of the data.
            B: unsigned byte
            b: signed byte
            h: short (2 bytes)
            i: int (4 bytes
            f: float (4 bytes)
            d: double (8 bytes)
    """
    

def idx_to_list(fname):
    """Convert the IDX file to an n dimensional list.
    
    Params:
        fname: The filename to read from.
    """
    with open(fname, 'rb') as f:
        pass
