import numpy
lib = numpy.ctypeslib.load_library('hello','.')
hello = lib.hello
hello.restype = None
hello("ciao")

