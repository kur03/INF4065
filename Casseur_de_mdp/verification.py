import hashlib
from numba import cuda
def verification_sha1 (hash, mdp) :
  if(hashlib.sha1(mdp.encode()).hexdigest() == hash) :
    print("Le mot de passe est : ", mdp)
    return mdp
  return False

#@cuda.jit 
def verification_MD5 (hash, mdp) :
  if(hashlib.md5(mdp.encode()).hexdigest() == hash) :
    print("Le mot de passe est : ", mdp)
    return mdp
  return False