import theano.tensor as T
import theano
import sys
from theano import pp
a = T.scalar('a')
b = T.scalar('b')
f = a+b
print pp(f)
print type(a)
print a.type
add = theano.function([a,b],f)
if len(sys.argv) == 3:
    arguments = map(int,sys.argv[1:])
    print add(arguments[0],arguments[1])
