import theano.tensor as T
import theano
import sys
a = T.scalar('a')
b = T.scalar('b')
f = a+b
add = theano.function([a,b],f)
if len(sys.argv) == 3:
    arguments = map(int,sys.argv[1:])
    print add(arguments[0],arguments[1])
