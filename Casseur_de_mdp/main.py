import hashlib
from brute_force import bf
import gpu
from numba import cuda
import numpy
import time
# import tensorflow as tf


HASH = 'a9993e364706816aba3e25717850c26c9cd0d89d'
mdp = numpy.empty(shape=0)
all = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','0','1','2','3','4','5','6','7','8','9','@','[',']','^','_','!','"','#','$','%','&','(',')','*','+',',','-','.','/',':',';','{','}','<','>','=','|','~','?']
start = time.perf_counter()
bf('sha1', HASH)
end = time.perf_counter()
print("elapsed time before cuda implementation = {}s".format((end - start)))

d_HASH = cuda.to_device(HASH)
d_mdp = cuda.to_device(mdp)
start = time.perf_counter()
gpu.kernel_combinaisons[5,2](d_mdp, d_HASH)
cuda.syncthreads()
end = time.perf_counter()
mdp = numpy.empty(shape=d_mdp.shape, dtype=d_mdp.dtype)
d_mdp.copy_to_host(mdp)
HASH = numpy.empty(shape=d_HASH.shape, dtype=d_HASH.dtype)
d_HASH.copy_to_host(HASH)
print(mdp)
print(HASH)
print("elapsed time before cuda implementation= {}s".format((end - start)))