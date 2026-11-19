"""Project actor encoder integrated with the unchanged upstream PPO trainer."""
import numpy as np
import torch
from torch import nn
from gymnasium.spaces import Box
from onpolicy.config import get_config
from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy
from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO
from swarm_core.learning.neighbor_encoder import NeighborEncoder

CONTRACT = "formation_hold_raw_v1"

def pack(observations):
    return np.stack([np.concatenate([
        o['local_features'], o['neighbor_features'].reshape(-1),
        o['neighbor_mask']]) for o in observations]).astype(np.float32)

class SetBase(nn.Module):
    """Parse raw rows inside the actor graph; preserve encoder gradients."""
    def __init__(self, hidden_size, local_dim=9, neighbor_dim=7, geometry_summary=False, append_hold=False, local_only=False):
        super().__init__()
        if type(local_only) is not bool:raise ValueError("local_only must be boolean.")
        self.local_only=local_only
        self.append_hold=append_hold
        if append_hold and local_dim!=16:raise ValueError("Phase exploration requires the local16 mission contract.")
        self.local_dim = local_dim
        self.neighbor_dim = neighbor_dim
        self.geometry_summary = geometry_summary
        if geometry_summary and neighbor_dim not in (9,11,13):
            raise ValueError("Relational summary requires nine geometry features with optional history metadata.")
        self.neighbors = NeighborEncoder(feature_dim=neighbor_dim)
        self.head = nn.Sequential(nn.Linear(local_dim + 34 + (4 if geometry_summary else 0), hidden_size), nn.Tanh(),
                                  nn.Linear(hidden_size, hidden_size), nn.Tanh())

    def forward(self, raw):
        count = (raw.shape[-1] - self.local_dim) // (self.neighbor_dim + 1)
        if raw.shape[-1] != self.local_dim + (self.neighbor_dim + 1) * count:
            raise ValueError('Invalid raw observation width.')
        if self.local_only:
            constant=raw.new_zeros((len(raw),34+(4 if self.geometry_summary else 0)))
            encoded=self.head(torch.cat([raw[:,:self.local_dim],constant],dim=-1))
            return torch.cat([encoded,raw[:,11:12]],dim=-1) if self.append_hold else encoded
        rows = raw[:, self.local_dim:self.local_dim + self.neighbor_dim * count].reshape(raw.shape[0], count, self.neighbor_dim)
        mask = raw[:, self.local_dim + self.neighbor_dim * count:]
        encoded = self.neighbors(rows, mask)
        inputs=[raw[:, :self.local_dim], encoded]
        if self.geometry_summary:
            weights=mask.unsqueeze(-1)
            count_valid=weights.sum(dim=1).clamp_min(1.)
            clean=torch.where(weights.bool(),rows,torch.zeros_like(rows))
            unit=clean[:,:,:2]/clean[:,:,:2].norm(dim=-1,keepdim=True).clamp_min(1e-8)
            error=clean[:,:,8:9]
            vector=(unit*error*weights).sum(dim=1)/count_valid
            signed=(error*weights).sum(dim=1)/count_valid
            largest=(error.abs()*weights).amax(dim=1) if count else raw.new_zeros((len(raw),1))
            inputs.append(torch.cat([vector,signed,largest],dim=-1))
        encoded=self.head(torch.cat(inputs, dim=-1))
        return torch.cat([encoded,raw[:,11:12]],dim=-1) if self.append_hold else encoded

def make_policy(num_agents=3, device='cpu', hidden_size=64, local_dim=9, critic_dim=None, neighbor_dim=7, geometry_summary=False, hold_std_scale=None, local_only=False):
    args = get_config().parse_args([])
    args.algorithm_name = 'mappo'
    args.use_recurrent_policy = False
    args.use_naive_recurrent_policy = False
    args.use_popart = False
    args.use_valuenorm = False
    args.hidden_size = hidden_size
    args.max_grad_norm = 0.5
    raw_dim = local_dim + (neighbor_dim + 1) * (num_agents - 1)
    # Critic uses physical states, critic-only readiness time and team ID.
    critic_dim = num_agents * 5 + 2 if critic_dim is None else critic_dim
    policy = R_MAPPOPolicy(args,
        Box(-np.inf, np.inf, (raw_dim,), dtype=np.float32),
        Box(-np.inf, np.inf, (critic_dim,), dtype=np.float32),
        Box(-1, 1, (2,), dtype=np.float32), device=torch.device(device))
    policy.actor.base = SetBase(hidden_size, local_dim, neighbor_dim, geometry_summary,append_hold=hold_std_scale is not None,local_only=local_only).to(device)
    if hold_std_scale is not None:
        from swarm_core.learning.phase_exploration import PhaseScaledGaussian
        previous=policy.actor.act.action_out
        replacement=PhaseScaledGaussian(hidden_size,2,args.use_orthogonal,args.gain,hold_std_scale).to(device)
        replacement.load_state_dict(previous.state_dict())
        policy.actor.act.action_out=replacement
    policy.actor_optimizer = torch.optim.Adam(policy.actor.parameters(),
        lr=args.lr, eps=args.opti_eps, weight_decay=args.weight_decay)
    return args, policy, R_MAPPO(args, policy, device=torch.device(device))

def critic_input(env):
    states = env.simulation.states
    if env.task.get('critic_contract') == 'mission_relative_v1':
        desired=env.task['team']['desired_pair_distance_m']
        goal_scale=env.task['mission']['translation_distance_m']
        limits=env.task['robot_defaults']
        positions=np.asarray([(s.x,s.y) for s in states])
        pairs=[np.linalg.norm(positions[i]-positions[j])/desired for i in range(len(states)) for j in range(i)]
        rows=[]
        for own in states:
            c,s=np.cos(own.heading),np.sin(own.heading)
            rotation=np.array([[c,s],[-s,c]])
            row=[]
            for i,peer in enumerate(states):
                relative=rotation@np.array([peer.x-own.x,peer.y-own.y])/desired
                angle=peer.heading-own.heading
                goal=np.zeros(2) if env.goals[i] is None else rotation@(env.goals[i]-[peer.x,peer.y])/goal_scale
                row.extend([*relative,np.sin(angle),np.cos(angle),peer.speed/limits['max_forward_speed_mps'],peer.yaw_rate/limits['max_yaw_rate_radps'],*np.eye(3)[env.phases[i]],*goal])
            row.extend(pairs)
            row.extend([env.monitor.hold_steps*env.simulation.dt/env.task['mission']['success_hold_duration_s'],*(rotation@env.direction)])
            rows.append(row)
        return np.asarray(rows,dtype=np.float32)
    if env.task.get('observation_contract') in ('formation_geometry_critic_v3','formation_relational_v4','formation_history_v5','formation_history_risk_v6'):
        positions=np.asarray([(s.x,s.y) for s in states])
        desired=env.task['team']['desired_pair_distance_m']
        pairs=[np.linalg.norm(positions[i]-positions[j])/desired for i in range(len(states)) for j in range(i)]
        limits=env.task['robot_defaults']
        speed_scale=max(limits['max_forward_speed_mps'],limits['max_reverse_speed_mps'])
        rows=[]
        for own in states:
            c,s=np.cos(own.heading),np.sin(own.heading)
            row=[]
            for peer in states:
                dx,dy=peer.x-own.x,peer.y-own.y
                angle=peer.heading-own.heading
                row.extend([(c*dx+s*dy)/desired,(-s*dx+c*dy)/desired,
                    np.sin(angle),np.cos(angle),peer.speed/speed_scale,
                    peer.yaw_rate/limits['max_yaw_rate_radps']])
            row.extend(pairs)
            row.extend([env.monitor.hold_steps*env.simulation.dt/env.task['mission']['success_hold_duration_s'],float(np.sqrt(np.mean((np.asarray(pairs)-1.)**2)))])
            rows.append(row)
        return np.asarray(rows,dtype=np.float32)
    flat = [v for s in states for v in (s.x, s.y, s.heading, s.speed, s.yaw_rate)]
    if hasattr(env, 'phases'):
        flat = []
        for i, state in enumerate(states):
            goal = env.goals[i] if env.goals[i] is not None else [0.,0.]
            flat.extend([state.x,state.y,state.heading,state.speed,state.yaw_rate,
                *np.eye(3)[env.phases[i]],*goal,
                float(env.simulation.formation_ready[i]),float(env.ready_steps[i])])
    return np.asarray([flat + [env.monitor.hold_steps * env.simulation.dt, i]
                       for i in range(len(states))], dtype=np.float32)

def hidden_inputs(num_agents, args):
    return (np.zeros((num_agents, args.recurrent_N, args.hidden_size), np.float32),
            np.ones((num_agents, 1), np.float32))

def gae(rewards, values, next_values, terminated, episode_end, gamma, lam):
    """Bootstrap truncations, stop traces at every reset and terminal."""
    advantages = np.zeros_like(rewards)
    running = np.zeros_like(rewards[0])
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * (1 - terminated[t]) * next_values[t] - values[t]
        running = delta + gamma * lam * (1 - episode_end[t]) * running
        advantages[t] = running
    return advantages, advantages + values
