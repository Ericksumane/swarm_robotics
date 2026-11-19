"""Verify full mission action parity with three independent local instances."""
from pathlib import Path
import argparse,copy,hashlib,json,subprocess,sys
import numpy as np
import torch
from swarm_core.mission_env import MissionEnv
from swarm_core.mission_spec import completion_task
from swarm_core.learning.staged_policy import BatchedMissionPolicy

def main():
    torch.set_num_threads(1)
    parser=argparse.ArgumentParser();parser.add_argument('--candidate');parser.add_argument('--output',default='experiments/runs/delivery_preparation_001');cli=parser.parse_args()
    registry=json.load(open(cli.candidate or 'experiments/selected_history_startup_candidate.json'))
    path=Path(registry['actor_bundle']);digest=hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest==registry['sha256']
    bundle=torch.load(path,map_location='cpu',weights_only=True)
    def model():return BatchedMissionPolicy(bundle['translation'],bundle['formation'],alignment=bundle['alignment'],acquisition_routing=bundle['acquisition_routing'],parking=bundle.get('parking'),motion_gate=bundle.get('motion_gate'))
    results={};service_observation=None
    for profile in ('nominal','severe'):
        task=completion_task(bundle['task'],profile)
        env=MissionEnv(task,json.load(open('configs/communication/'+profile+'.json')),json.load(open('configs/training/geometry_control.json')))
        observations,_=env.reset(42);batched=model();locals_=[model() for _ in range(3)]
        maximum=0.;modes_seen=set();ticks=0
        if service_observation is None:service_observation=copy.deepcopy(observations[0])
        while True:
            expected,modes=batched.act(observations)
            for i in range(3):
                actual,mode=locals_[i].act_local(observations[i],i)
                assert mode==modes[i],(profile,i,mode,modes[i])
                np.testing.assert_allclose(actual,expected[i],rtol=0.,atol=2e-6)
                maximum=max(maximum,float(np.max(np.abs(actual-expected[i]))));modes_seen.add(mode)
            observations,_,terminated,truncated,info=env.step(expected);ticks+=1
            if terminated or truncated:break
        assert info['reason']=='success',(profile,info)
        results[profile]=dict(ticks=ticks,max_action_difference=maximum,modes=sorted(modes_seen))
    o=service_observation
    payload={k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in o.items() if k!='formation_history'}
    payload['formation_history']={k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in o['formation_history'].items() if k in ('neighbor_ids','neighbor_features','neighbor_mask','neighbor_sequence')}
    requests=[dict(protocol_version=1,op='reset'),dict(protocol_version=1,op='act',observation=payload)]
    command=[sys.executable,'scripts/local_mission_actor_service.py','--bundle',str(path),'--robot-id','0','--expected-sha256',digest]
    process=subprocess.run(command,input=''.join(json.dumps(r,allow_nan=False)+'\n' for r in requests),capture_output=True,text=True,check=True)
    messages=[json.loads(line) for line in process.stdout.splitlines()]
    assert [m['kind'] for m in messages]==['ready','reset','action'],messages
    expected,mode=model().act_local(o,0)
    np.testing.assert_allclose(messages[-1]['action'],expected,rtol=0.,atol=2e-6)
    assert messages[-1]['mode']==mode and messages[0]['training_enabled'] is False
    bad=subprocess.run(command[:-1]+['0'*64],input='',capture_output=True,text=True)
    assert bad.returncode!=0 and 'checksum' in bad.stderr
    out=Path(cli.output);out.mkdir(exist_ok=True)
    (out/'local_inference_parity.json').write_text(json.dumps(dict(bundle_sha256=digest,profiles=results,service_protocol='passed',checksum_rejection='passed'),indent=2))
    print('Full-mission batched versus independent local inference: PASSED')
    print('Versioned local service and checksum rejection: PASSED')
    print(json.dumps(results))

if __name__=='__main__':main()
