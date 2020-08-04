import os
import sys
from collections import OrderedDict
from os.path import join,getsize

def readable_size(size):
    for unit in ['K','M']:
        if abs(size) < 1024.0:
            return "%.1f%sB" % (size,unit)
        size /= 1024.0
    size /= 1024.0
    return "%.1f%s" % (size,'GB')


a = 0
a = a + 32* 256 * 256 * 16
a = a + 64* 256 * 256 * 16 * 8
a = a + 64* 512 * 512 * 32 
a = a + 32* 512 * 512 * 32 * 8
a = a + 14 * 512 * 512 * 32
a = a + 14 * 512 * 512 * 32

b = (a * 1 + 2649) * 8

print(readable_size(b))
