from pathlib import Path
import subprocess
import torch
from swarm_core.learning.mappo_adapter import make_policy

root=Path(__file__).resolve().parents[2]
torch.manual_seed(444)
_,policy,_=make_policy(3,local_dim=16,critic_dim=39,neighbor_dim=9,geometry_summary=True)
with torch.no_grad():
 policy.actor.act.action_out.fc_mean.weight.zero_()
 policy.actor.act.action_out.fc_mean.bias.copy_(torch.tensor([.15,0.]))
prior=root/'experiments/initializations/translation_slow_forward_prior.pt'
if prior.exists():raise FileExistsError(prior)
torch.save(dict(contract='mission_relational_v2',actor=policy.actor.state_dict(),initialization='trainable mean head: zero weights, bias .15 forward and zero yaw'),prior)
args=[str(root/'.venv/bin/python'),'scripts/train_formation.py','--steps','120000','--rollout','256','--epochs','5','--seed','444','--gamma','.99','--gae-lambda','.95','--initial-std','.02','--final-std','.02','--entropy-coef','0','--snapshot-interval','30000','--initial-actor',str(prior),'--task','configs/tasks/translate_stop_half_meter_relative.json','--reward','configs/training/translation_reward_goal.json','--communication','configs/communication/good.json','--output','experiments/runs/translation_short_relative_001']
p=subprocess.Popen(args,cwd=root,stdout=open(root/'experiments/runs/translation_short_relative_001.log','w'),stderr=subprocess.STDOUT,start_new_session=True)
print(p.pid)
