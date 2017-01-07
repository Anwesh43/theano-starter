import theano.tensor as T
from theano import *
from theano.tensor.shared_randomstreams import RandomStreams
srng = RandomStreams(seed=1000)
r_uv = srng.uniform((2,2))
r_nd = srng.normal((2,2))
rng_val =  r_uv.rng.get_value(borrow=True)
rng_val.seed(345)
r_uv.rng.set_value(rng_val,borrow=True)
r_uniform = function([],r_uv)
r_normal = function([],r_nd,no_default_updates=True)

print r_uniform()
print r_normal()
print r_normal()


#printing seeds of random variables
