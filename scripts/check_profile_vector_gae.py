"""Compare vector-profile GAE with independent per-environment recursions."""
import json
import numpy as np
from swarm_core.learning.mappo_adapter import gae
from swarm_core.motion_profile import scaled_linear_speed
rng=np.random.default_rng(105033);shape=(9,12,1)
r,v,n=[rng.normal(size=shape).astype(np.float32) for _ in range(3)]
t=np.zeros(shape,dtype=np.float32);e=t.copy();t[3,:3]=1.;e[3,:3]=1.;e[5,6:9]=1.
a,ret=gae(r,v,n,t,e,.999,.95)
for i in range(4):
 sl=slice(i*3,(i+1)*3);expected=gae(r[:,sl],v[:,sl],n[:,sl],t[:,sl],e[:,sl],.999,.95)
 np.testing.assert_array_equal(a[:,sl],expected[0]);np.testing.assert_array_equal(ret[:,sl],expected[1])
p=json.load(open('configs/training/speed033_campaign.json'));assert sum(p['allocation_per_profile'].values())==p['total_environment_steps_per_profile']==200000
base=dict(robot_defaults=dict(max_forward_speed_mps=.3,max_reverse_speed_mps=.15),peer_history=dict(max_speed_mps=.3),formation_peer_history=dict(max_speed_mps=.3))
new=scaled_linear_speed(base,1.1);assert base['peer_history']['max_speed_mps']==.3
assert np.isclose(new['peer_history']['max_speed_mps'],.33) and np.isclose(new['formation_peer_history']['max_speed_mps'],.33)
print('Independent profile GAE boundaries, exact budget and coherent history speed limits: PASSED')
