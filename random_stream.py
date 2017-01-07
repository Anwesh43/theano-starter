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

rnd_val = r_uv.rng.get_value(borrow=True)
state = rnd_val.get_state()

v1 =  r_uniform()
v2 =  r_uniform()

rnd_val = r_uv.rng.get_value(borrow=True)
rnd_val.set_state(state)
r_nd.rng.set_value(rnd_val)
v3 = r_uniform()
print v1
print v2
print v3
print v1 == v3
