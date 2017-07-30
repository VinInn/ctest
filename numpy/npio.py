import numpy as np

n = ['1','2']

a = np.array([[1,2,3,4],[4,5,6,8]])
b = np.array([[1,2,3,4,3,3],[4,5,6,8,2,2]])

c = np.array([[1,2,3,4],[4,5,6,8],[4,5,6,8]])
d = np.array([[4,5,6,8,2,2],[1,2,3,4,3,3],[4,5,6,8,2,2]])

k1 = (a,b)
k2 = (c,d)
v1 = [a,b]
v2 = [c,d]

def q(*a) :
  for i in a: print i

q(n,*(k1+k2))

np.savez_compressed('/tmp/bha',n,*(k1+k2))

ret = np.load('/tmp/bha.npz')
print len(ret['arr_0'])

r1 = [ret['arr_'+str(i+1)] for i in range(0,len(ret['arr_0']))}
r2 = [ret['arr_'+str(i+1+len(ret['arr_0']))] for i in range(0,len(ret['arr_0']))]
map(np.array_equal,r1,v1)
map(np.array_equal,r2,v2)

