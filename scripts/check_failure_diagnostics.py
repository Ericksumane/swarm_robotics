"""Distinguish acquisition, stopping and interrupted hold without changing outcomes."""
import json, math
from copy import deepcopy
from dataclasses import replace
from swarm_core.termination import FormationTermination
from swarm_core.dynamics import RobotState

def main():
    task=json.load(open('configs/tasks/formation_v1.json'))
    task['simulation']['episode_time_limit_s']=.3
    triangle=[RobotState(x=0),RobotState(x=1),RobotState(x=.5,y=math.sqrt(3)/2)]
    expanded=[replace(s,x=2*s.x,y=2*s.y) for s in triangle]
    moving=[replace(s,speed=.1) for s in triangle]
    def run(sequence,expected):
        monitor=FormationTermination(task)
        for states in sequence:info=monitor.update(states)
        assert info['reason']=='time_limit' and info['truncated'] and not info['terminated']
        assert info['diagnostics']['failure_stage']==expected
        return monitor,info['diagnostics']
    _,never=run([expanded]*3,'never_formed')
    assert never['first_geometry_time_s'] is None
    _,stopping=run([moving]*3,'formed_never_stopped')
    assert math.isclose(stopping['geometry_ready_time_s'],.3)
    assert stopping['max_hold_time_s']==0
    assert math.isclose(stopping['mean_abs_speed_mps'],.1)
    assert math.isclose(stopping['elapsed_time_s'],.3)
    monitor,interrupted=run([triangle,moving,triangle],'hold_not_sustained')
    assert math.isclose(interrupted['qualifying_time_s'],.2)
    assert math.isclose(interrupted['max_hold_time_s'],.1)
    monitor.reset()
    assert monitor.geometry_steps==monitor.max_hold_steps==0
    assert monitor.best_pair_error is None
    print('Acquisition / stopping / sustained-hold failure classification: PASSED')
    print('Diagnostic reset and unchanged time-limit outcomes: PASSED')

if __name__=='__main__':main()
