"""Verify measurement separation, reproducibility and consistent broadcasts."""
import json
from swarm_core.simulation import Simulation
def main():
    task=json.load(open('configs/tasks/formation_noisy.json'))
    comm=json.load(open('configs/communication/nominal.json'))
    sim=Simulation(task,comm)
    assert sim.estimates!=sim.states
    initial=list(sim.estimates)
    message,_,valid=sim.network.read(1,0,0.)
    assert valid and message.x==sim.estimates[0].x
    assert message.x!=sim.states[0].x
    sim.reset()
    assert sim.estimates==initial
    print('Measured-state separation, consistent broadcast and seeded reset: PASSED')
if __name__=='__main__': main()
