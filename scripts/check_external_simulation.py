"""Require fresh feedback and prove no Python dynamics integration occurs."""
import json
import numpy as np
import swarm_core.simulation as python_simulation
from swarm_core.external_simulation import ExternalSimulation
task=json.load(open('configs/tasks/dispersed_full_aligned.json'));comm=json.load(open('configs/communication/nominal.json'));sim=ExternalSimulation(task,comm);sim.reset(42)
initial=[vars(s).copy() for s in sim.states]
try:sim.advance(np.zeros((3,2)))
except RuntimeError:pass
else:raise AssertionError('Advanced without feedback.')
sim.initialize_physics(initial,.5,42);measured=[dict(s) for s in initial];measured[0]['x']+=.37
old=python_simulation.step
def forbidden(*args,**kwargs):raise AssertionError('Python physics integration was called.')
python_simulation.step=forbidden
try:
 sim.offer(measured,.6);sim.advance(np.ones((3,2)))
finally:python_simulation.step=old
assert sim.states[0].x==measured[0]['x'] and sim.time_s==.1
message,_,valid=sim.network.read(1,0,.1);assert valid and message.x==measured[0]['x']
for bad_clock in [.6,.8,float('nan')]:
 try:sim.offer(measured,bad_clock)
 except ValueError:pass
 else:raise AssertionError('Stale/skipped clock accepted.')
sim.offer(measured,.7)
try:sim.offer(measured,.7)
except RuntimeError:pass
else:raise AssertionError('Duplicate pending sample accepted.')
print('External feedback freshness, clock discipline and no Python integration: PASSED')
