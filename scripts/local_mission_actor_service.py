"""One-robot actor-only JSON-lines service for the full fixed mission bundle.

This service accepts a locally constructed observation. It does not construct
mission phases, receive ROS messages or publish actuator commands. The ROS
mission adapter must supply those functions before hardware deployment.
"""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
from swarm_core.learning.staged_policy import BatchedMissionPolicy

def emit(value):print(json.dumps(value,allow_nan=False),flush=True)

def decode_observation(value):
    if not isinstance(value,dict):raise ValueError('Observation must be an object.')
    result=dict(value)
    for key in ('local_features','neighbor_features','neighbor_mask'):
        result[key]=np.asarray(value[key],dtype=np.float32)
    if value.get('formation_history') is not None:
        history=dict(value['formation_history'])
        for key in ('neighbor_features','neighbor_mask'):
            history[key]=np.asarray(history[key],dtype=np.float32)
        result['formation_history']=history
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',required=True);p.add_argument('--robot-id',required=True,type=int,choices=[0,1,2]);p.add_argument('--expected-sha256',required=True);a=p.parse_args()
    path=Path(a.bundle);digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if digest!=a.expected_sha256:raise ValueError('Bundle checksum differs from deployment manifest.')
    torch.set_num_threads(1)
    bundle=torch.load(path,map_location='cpu',weights_only=True)
    policy=BatchedMissionPolicy(bundle['translation'],bundle['formation'],alignment=bundle['alignment'],acquisition_routing=bundle['acquisition_routing'],parking=bundle.get('parking'),motion_gate=bundle.get('motion_gate'))
    emit(dict(kind='ready',protocol_version=1,robot_id=a.robot_id,bundle_sha256=digest,architecture=bundle['architecture'],training_enabled=False))
    for line in sys.stdin:
        try:
            request=json.loads(line)
            if request.get('protocol_version')!=1:raise ValueError('Unsupported inference protocol version.')
            if request.get('op')=='reset':policy.reset();emit(dict(kind='reset'));continue
            if request.get('op')!='act':raise ValueError('Unknown inference operation.')
            started=time.perf_counter()
            action,mode=policy.act_local(decode_observation(request['observation']),a.robot_id)
            emit(dict(kind='action',action=action.tolist(),mode=mode,inference_ms=1000*(time.perf_counter()-started)))
        except Exception as error:
            policy.reset()
            emit(dict(kind='error',error=str(error)))
            # A bad request requires a supervisor restart; do not continue with
            # potentially mismatched local mission/routing state.
            return 2
    return 0

if __name__=='__main__':sys.exit(main())
