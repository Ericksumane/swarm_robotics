"""Check local-only entry, latch retention, mission reset and phase guards."""
import numpy as np
from swarm_core.learning.approach_router import LocalApproachRouter

def main():
 router=LocalApproachRouter();raw=np.zeros(36);raw[10]=1.;raw[12]=.5
 assert router.select(raw)=='cruise'
 raw[12]=.5-2e-5;assert router.select(raw)=='approach'
 raw[12]=.5;assert router.select(raw)=='approach'
 router.reset();assert router.select(raw)=='cruise'
 raw[10]=0;raw[11]=1;assert router.select(raw)=='approach'
 router.reset();raw[11]=0;raw[9]=1
 try:router.select(raw)
 except ValueError:pass
 else:raise AssertionError('Unsupported FORM accepted')
 print('Local approach entry, retention and reset: PASSED')
 print('Unsupported formation acquisition rejected: PASSED')

if __name__=='__main__':main()
