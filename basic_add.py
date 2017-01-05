import theano.tensor as T
import theano
a = T.scalar('a')
b = T.scalar('b')
f = a+b
add = theano.function([a,b],f)
print add(2,3)
print add(5,10)
