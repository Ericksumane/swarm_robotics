"""Analytic checks distinguish termination, truncation and trace boundaries."""
import numpy as np
from swarm_core.learning.mappo_adapter import gae

def main():
    r=np.array([[[1.]],[[2.]]]); v=np.zeros_like(r); nv=np.full_like(r,10.)
    term=np.array([[[0.]],[[1.]]]); end=np.array([[[1.]],[[1.]]])
    a,ret=gae(r,v,nv,term,end,0.9,0.95)
    np.testing.assert_allclose(a[:,0,0],[10.,2.])
    a,_=gae(r,v,nv,np.ones_like(term),end,0.9,0.95)
    np.testing.assert_allclose(a,r)
    print('Terminal zero bootstrap, truncation bootstrap and reset trace isolation: PASSED')

if __name__=='__main__': main()
