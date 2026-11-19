"""One robot's fixed-weight mission service; ROS is handled by the parent."""
import argparse,hashlib,json,sys,uuid
from pathlib import Path
import torch
from swarm_core.dynamics import RobotState
from swarm_core.mission_spec import completion_task
from swarm_core.learning.staged_policy import BatchedMissionPolicy
from swarm_core.deployment.local_runtime import LocalMissionRuntime

def emit(value):print(json.dumps(value,allow_nan=False),flush=True)
def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',required=True);p.add_argument('--expected-sha256',required=True);p.add_argument('--robot-id',required=True,type=int,choices=[0,1,2]);p.add_argument('--mission-id',required=True);p.add_argument('--frame-id',required=True);p.add_argument('--profile',choices=['nominal','good','moderate','severe'],default='nominal');a=p.parse_args()
    path=Path(a.bundle);digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if digest!=a.expected_sha256:raise ValueError('Bundle checksum mismatch.')
    torch.set_num_threads(1);b=torch.load(path,map_location='cpu',weights_only=True)
    policy=BatchedMissionPolicy(b['translation'],b['formation'],alignment=b['alignment'],acquisition_routing=b['acquisition_routing'],parking=b.get('parking'),motion_gate=b.get('motion_gate'))
    task=completion_task(b['task'],a.profile)
    runtime=LocalMissionRuntime(task,a.robot_id,a.mission_id,uuid.uuid4().hex,a.frame_id,policy)
    emit(dict(kind='ready',protocol_version=1,robot_id=a.robot_id,mission_id=a.mission_id,bundle_sha256=digest,task=task,shadow_only=True))
    for line in sys.stdin:
        try:
            r=json.loads(line)
            if r.get('protocol_version')!=1:raise ValueError('Unsupported service protocol.')
            if r.get('op')=='peer':emit(dict(kind='peer',accepted=runtime.accept_peer(r['peer'],r['now_s'])));continue
            if r.get('op')!='tick':raise ValueError('Unknown operation.')
            own=RobotState(**r['own_state']);result=runtime.tick(own,r['now_s'],r['odometry_stamp_s'])
            emit(dict(kind='tick',**result))
        except Exception as error:emit(dict(kind='error',error=str(error)));return 2
    return 0
if __name__=='__main__':sys.exit(main())
