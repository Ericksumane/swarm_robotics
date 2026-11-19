"""Actor warm starts must respect an explicit initial exploration scale."""
import json, subprocess, sys, tempfile
from pathlib import Path
import torch
from swarm_core.learning.mappo_adapter import make_policy

def main():
    torch.set_num_threads(1)
    _, policy, _ = make_policy(3, critic_dim=23, neighbor_dim=9)
    with torch.no_grad():
        policy.actor.act.action_out.logstd._bias.fill_(0.)
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        actor = directory/'initial.pt'
        output = directory/'run'
        torch.save({'actor': policy.actor.state_dict()}, actor)
        subprocess.run([sys.executable, 'scripts/train_formation.py',
            '--steps','16','--rollout','16','--epochs','1',
            '--initial-actor',str(actor),'--initial-std','.02',
            '--task','configs/tasks/formation_relative_critic.json',
            '--reward','configs/training/formation_reward_proximity.json',
            '--output',str(output)], check=True, stdout=subprocess.DEVNULL)
        manifest=json.loads((output/'manifest.json').read_text())
        assert all(abs(value-.02)<1e-6 for value in manifest['starting_action_std'])
        update=json.loads((output/'updates.jsonl').read_text().splitlines()[0])
        assert all(key in update for key in ('approx_kl','ppo_clip_fraction','actor_grad_norm'))
    print('Loaded actor respects requested initial exploration: PASSED')

if __name__ == '__main__': main()
