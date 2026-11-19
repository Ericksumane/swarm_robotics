"""Check mastery promotion, rehearsal isolation and fixed active goals."""
import json
import numpy as np
from swarm_core.distance_curriculum import DistanceCurriculum
from swarm_core.mission_env import MissionEnv

def main():
 task=json.load(open('configs/tasks/translate_stop_distance_curriculum.json'));settings=task['mission_distance_curriculum']
 env=MissionEnv(task,json.load(open('configs/communication/good.json')),json.load(open('configs/training/translation_reward_local_goal.json')))
 curriculum=DistanceCurriculum(settings,91);curriculum.apply(env);env.reset(88)
 goals=np.array(env.goals);np.testing.assert_allclose(np.array(env.goals)-np.array(env.starts),np.tile([.5,0.],(3,1)))
 for _ in range(8):curriculum.record(True)
 assert curriculum.stage==1
 # Rehearsal outcomes must never promote an untested frontier.
 curriculum.sampled_stage=0
 for _ in range(20):curriculum.record(True)
 assert curriculum.stage==1 and len(curriculum.history)==0
 np.testing.assert_array_equal(env.goals,goals)
 for _ in range(25):
  curriculum.sampled_stage=curriculum.stage
  curriculum.record(True)
 assert curriculum.stage==4
 print('Distance mastery progression and rehearsal isolation: PASSED')
 print('Reset-only curriculum and fixed episode targets: PASSED')

if __name__=='__main__':main()
