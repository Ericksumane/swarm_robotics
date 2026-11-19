"""Check phase std, preserved means, PPO likelihoods and actor gradients."""
import numpy as np
import torch
from swarm_core.learning.mappo_adapter import make_policy,hidden_inputs

def main():
 torch.set_num_threads(1);torch.manual_seed(81)
 args,plain,_=make_policy(3,local_dim=16,critic_dim=39,neighbor_dim=9,geometry_summary=True)
 _,phase,_=make_policy(3,local_dim=16,critic_dim=39,neighbor_dim=9,geometry_summary=True,hold_std_scale=.1)
 phase.actor.load_state_dict(plain.actor.state_dict())
 raw=torch.randn(4,36);raw[:,9:12]=torch.tensor([[0.,1.,0.],[0.,0.,1.],[0.,1.,0.],[0.,0.,1.]])
 raw[:,-2:]=1.
 normal=plain.actor.act.action_out(plain.actor.base(raw))
 scaled=phase.actor.act.action_out(phase.actor.base(raw))
 torch.testing.assert_close(normal.mean,scaled.mean)
 torch.testing.assert_close(scaled.stddev[[0,2]],normal.stddev[[0,2]])
 torch.testing.assert_close(scaled.stddev[[1,3]],normal.stddev[[1,3]]*.1)
 h,m=hidden_inputs(4,args)
 actions,old_logp,_=phase.actor(raw,h,m)
 new_logp,entropy=phase.actor.evaluate_actions(raw,h,actions.detach(),m)
 torch.testing.assert_close(old_logp,new_logp)
 loss=-new_logp.mean();loss.backward()
 assert all(p.grad is None or torch.isfinite(p.grad).all() for p in phase.actor.parameters())
 assert phase.actor.act.action_out.fc_mean.weight.grad.abs().sum()>0
 print('HOLD variance scaling preserves learned action means: PASSED')
 print('Phase-conditioned PPO likelihood and finite gradients: PASSED')

if __name__=='__main__':main()
