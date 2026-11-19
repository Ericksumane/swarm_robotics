"""Check reproducible profile sampling and copy isolation."""
import json
from swarm_core.communication.profile_sampler import ProfileSampler
settings=json.load(open('configs/training/arrival_communication_mixture.json'));a=ProfileSampler(settings);b=ProfileSampler(settings)
assert [a.sample(i) for i in range(100)]==[b.sample(i) for i in range(100)]
assert {a.sample(i)['profile_name'] for i in range(100)}=={'nominal','good','moderate','severe'}
sample=a.sample(1);sample['delay_s']=999
assert a.sample(1)['delay_s']!=999
for probabilities in [[1.],[-1.,1.,.5,.5],[0.,0.,0.,0.]]:
 try:ProfileSampler(dict(settings,probabilities=probabilities))
 except ValueError:pass
 else:raise AssertionError('Invalid mixture accepted.')
print('Episode network sampling, coverage and isolation: PASSED')
