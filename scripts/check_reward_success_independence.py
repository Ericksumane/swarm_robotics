"""Success bonus must not depend on progress shaping being enabled."""
import json, math
import numpy as np
from swarm_core.formation_env import FormationEnv
from swarm_core.dynamics import RobotState

def main():
    load = lambda path: json.load(open(path))
    task = load('configs/tasks/formation_relative_critic.json')
    comm = load('configs/communication/nominal.json')
    weights = load('configs/training/formation_reward_proximity.json')
    for progress in (0., 20.):
        config = dict(weights, progress_weight=progress)
        env = FormationEnv(task, comm, config)
        env.reset(42)
        env.simulation.states = [RobotState(x=0,y=0),RobotState(x=1,y=0),
            RobotState(x=.5,y=math.sqrt(3)/2)]
        for index in range(env.monitor.required_steps):
            _, reward, terminated, truncated, info = env.step(np.zeros((3,2)))
            assert info['reward_terms']['success'] == (weights['success_bonus'] if terminated else 0.)
            np.testing.assert_allclose(reward, sum(info['reward_terms'].values()), atol=1e-6)
        assert terminated and not truncated and info['reason'] == 'success'
    print('Success bonus independent of progress shaping: PASSED')

if __name__ == '__main__': main()
