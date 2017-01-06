from theano import *
import theano.tensor as T
import sys
a = T.dscalar('x')
b = T.dscalar('y')
f = a+b
add = function([In(a,value=10),In(b,value=5)],f)

if len(sys.argv) == 3:
    n1 = int(sys.argv[1])
    n2 = int(sys.argv[2])
    print add(n1,n2)
else:
    print add()
