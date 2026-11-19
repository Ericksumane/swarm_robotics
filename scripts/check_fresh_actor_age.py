"""An available packet is not necessarily young enough for the raw expert."""
import numpy as np
from swarm_core.learning.formation_expert_router import LocalFormationExpertRouter
history=np.zeros((2,11));history[:,:2]=[[1.,0.],[.5,np.sqrt(3)/2]]
rows=np.zeros((2,9));rows[:,6]=.2
o=dict(neighbor_mask=np.ones(2),neighbor_features=rows,formation_history=dict(neighbor_features=history,neighbor_mask=np.ones(2)))
legacy=LocalFormationExpertRouter();qualified=LocalFormationExpertRouter(.1)
for _ in range(4):assert legacy.select(o)=='formation_fresh';mode=qualified.select(o)
assert mode=='formation'
o['neighbor_features'][:,6]=0.
for _ in range(14):assert qualified.select(o)=='formation'
assert qualified.select(o)=='formation_fresh'
print('Valid-but-aged packet routing and legacy expert parity: PASSED')
