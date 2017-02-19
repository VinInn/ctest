
# coding: utf-8

# In[1]:

import numpy
import ctypes
import subprocess
print "bha"
print subprocess.check_output("echo here am",stderr=subprocess.STDOUT,shell=True)
print subprocess.check_output("c++ -v; exit 0",stderr=subprocess.STDOUT,shell=True)
print subprocess.check_output("ls -l;c++ fatlib.cc -fPIC -shared -o fatlib.so; exit 0",stderr=subprocess.STDOUT,shell=True)
print "bho"
lib = numpy.ctypeslib.load_library('fatlib','.')
hello = lib.fathello
hello.restype = ctypes.c_char_p
print hello()
print "bhe"
fma = lib.myfma
fma.restype = ctypes.c_float
fma.argtypes = [ctypes.c_float,ctypes.c_float,ctypes.c_float]
print fma(1,2,3)


# In[ ]:



