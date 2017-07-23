import multiprocessing
a = [1,2,3,4]
def d(x) : 
	return 2*x

p = multiprocessing.Pool(4)
print d(a[0])
b = p.map(d,a)
print b
a = [-1,-2,-3,-4]
b = p.map(d,a)
print b

