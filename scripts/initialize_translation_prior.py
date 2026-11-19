"""Create a transparent constant-forward neural initialization, not a controller."""
from pathlib import Path
import torch
from swarm_core.learning.mappo_adapter import make_policy

def main():
 torch.manual_seed(443)
 _,policy,_=make_policy(3,local_dim=16,critic_dim=38,neighbor_dim=9,geometry_summary=True)
 with torch.no_grad():
  policy.actor.act.action_out.fc_mean.weight.zero_()
  policy.actor.act.action_out.fc_mean.bias.copy_(torch.tensor([.4,0.]))
 out=Path('experiments/initializations/translation_forward_prior.pt')
 out.parent.mkdir(parents=True,exist_ok=True)
 if out.exists():raise FileExistsError(out)
 torch.save(dict(contract='mission_relational_v2',actor=policy.actor.state_dict(),
  initialization='constant mean normalized forward .4 and yaw 0; all parameters trainable'),out)
 print(out)

if __name__=='__main__':main()
