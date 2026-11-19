"""Check trusted artifact loading and one local action on the installed CPU stack."""
from pathlib import Path
import hashlib,json
import numpy as np
import torch
from swarm_core.learning.staged_policy import BatchedMissionPolicy
from swarm_core.mission_env import MissionEnv
from swarm_core.mission_spec import completion_task

def main():
    root=Path(__file__).resolve().parents[1];d=json.loads((root/'configs/deployment/delivery_candidate.json').read_text());path=root/d['actor_bundle']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==d['actor_bundle_sha256'],'Policy checksum mismatch.'
    torch.set_num_threads(1);b=torch.load(path,map_location='cpu',weights_only=True)
    model=BatchedMissionPolicy(b['translation'],b['formation'],alignment=b['alignment'],acquisition_routing=b['acquisition_routing'],parking=b.get('parking'),motion_gate=b.get('motion_gate'))
    task=completion_task(b['task'],'nominal');comm=json.loads((root/'configs/communication/nominal.json').read_text())
    obs,_=MissionEnv(task,comm,{}).reset(42)
    action,mode=model.act_local(obs[0],0)
    assert action.shape==(2,) and np.isfinite(action).all()
    print(json.dumps(dict(result='passed',torch=torch.__version__,numpy=np.__version__,device='cpu',mode=mode,bundle_sha256=d['actor_bundle_sha256'])))
if __name__=='__main__':main()
