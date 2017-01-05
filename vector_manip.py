import theano.tensor as T
import sys
from theano import function
a = T.vector('a')
s = a**10+a
if len(sys.argv) > 1:
    arguments = map(int,sys.argv[1:])
    print s.eval({a:arguments})
