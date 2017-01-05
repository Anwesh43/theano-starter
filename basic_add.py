import theano.tensor as T
import theano
import sys
from theano import pp
a = T.scalar('a')
b = T.scalar('b')
f = a+b
f2 = a+2*b
print pp(f)
print type(a)
print a.type
add = theano.function([a,b],f)
if len(sys.argv) == 3:
    arguments = map(int,sys.argv[1:])
    print add(arguments[0],arguments[1])
    print f2.eval({a:arguments[0],b:arguments[1]})
