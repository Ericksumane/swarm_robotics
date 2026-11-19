"""Balanced four-profile MAPPO adaptation of selected existing mission experts.

One vector tick advances four independent three-robot environments. It counts
as ONE environment step PER PROFILE, not twelve steps. Critics are centralized
only in training. CPU rollout replicas and optional CUDA PPO have identical
weights at every rollout boundary. No vendor/hardware actuation is involved.
"""
import argparse,copy,hashlib,json,math,subprocess,time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from swarm_core.formation_env import FormationEnv
from swarm_core.alignment_env import AlignmentEnv
from swarm_core.arrival_training import ArrivalTrainingEnv
from swarm_core.formation_safety_training import FormationSafetyTrainingEnv
from swarm_core.mission_env import MissionEnv
from swarm_core.motion_profile import scaled_linear_speed
from swarm_core.learning.mappo_adapter import make_policy,hidden_inputs,pack,critic_input,gae

ROOT=Path(__file__).resolve().parents[2]
STAGES={
 'formation_history':('configs/tasks/history_collision_training.json','configs/training/history_collision_reward.json'),
 'formation_fresh':('configs/tasks/dispersed_acquisition_training.json','configs/training/dispersed_acquisition_reward.json'),
 'alignment':('configs/tasks/alignment_stationary_v3.json','configs/training/alignment_reward.json'),
 'cruise':('configs/tasks/translate_stop_hold_hysteresis.json','configs/training/geometry_control.json'),
 'cruise_degraded':('configs/tasks/translate_stop_hold_hysteresis.json','configs/training/geometry_control.json'),
 'approach':('experiments/runs/approach_recovery_phase_noise_001/training_task.json','configs/training/approach_reward.json')}

def read(path):return json.loads((ROOT/path).read_text())
def atomic_json(path,data):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)
def cpu_state(module):return {k:v.detach().cpu().clone() for k,v in module.state_dict().items()}
def component(bundle,name):
 if name=='formation_history':return bundle['formation']
 if name=='formation_fresh':return bundle['formation']['fresh_actor']
 if name=='alignment':return bundle['alignment']
 return bundle['translation'][name]

def task_for(name,seed,plan=None):
 plan=plan or {}
 task=read(plan.get('task_overrides',{}).get(name,STAGES[name][0]));task=scaled_linear_speed(task,plan.get('task_linear_speed_scale',1.1));task['seed']=seed
 if plan.get('mission_profile'):
  profile=plan['mission_profile']
  task['robot_defaults']['max_forward_speed_mps']=profile['max_forward_speed_mps']
  task['team']['desired_pair_distance_m']=profile['desired_pair_distance_m']
  task['mission']['final_heading_tolerance_rad']=profile['final_heading_tolerance_rad']
  for key in ('formation_peer_history','peer_history'):
   if key in task:task[key]['max_speed_mps']=profile['max_forward_speed_mps']
  if 'arrival_training' in task:
   task['arrival_training']['parking']['heading_tolerance_rad']=profile['final_heading_tolerance_rad']
   task['arrival_training']['final_heading_weight']=profile['arrival_heading_weight']
 if name in ('formation_history','formation_fresh') and not plan.get('preserve_task_initialization',False):
  task['simulation']['episode_time_limit_s']=180.
  # User-discovered failures are DEVELOPMENT training cases, never held out.
  bank=task['initialization'].setdefault('hard_examples',dict(probability=.3,position_jitter_m=.04,heading_jitter_rad=.15,minimum_pair_error_m=.12,entries=[]))
  for run in ['gazebo_live_20261105_010906_33245','gazebo_live_20261105_011258_34765','gazebo_live_20261105_011855_37033']:
   m=read('experiments/runs/'+run+'/manifest.json');bank['entries'].append(dict(source=run,states=m['hello']['initial_states']))
  task['initialization'].pop('near_formation',None)
 return task

def main():
 p=argparse.ArgumentParser();p.add_argument('--plan',default='configs/training/speed033_campaign.json');p.add_argument('--output',required=True);p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--smoke',action='store_true');p.add_argument('--resume',action='store_true');a=p.parse_args()
 torch.set_num_threads(1);plan=read(a.plan);profiles=plan['profiles'];assert profiles in (['nominal'],['nominal','good','moderate','severe']);batch_size=3*len(profiles)
 assert sum(plan['allocation_per_profile'].values())==plan['total_environment_steps_per_profile'] and plan['total_environment_steps_per_profile']>0
 if 'total_environment_steps' in plan:assert plan['total_environment_steps']==len(profiles)*plan['total_environment_steps_per_profile']
 if a.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable.')
 torch.manual_seed(plan['seed']);rng=np.random.default_rng(plan['seed']);out=Path(a.output);out=out if out.is_absolute() else ROOT/out
 if out.exists() and not a.resume:raise FileExistsError(out)
 out.mkdir(parents=True,exist_ok=a.resume)
 source=ROOT/plan['source_bundle'];assert hashlib.sha256(source.read_bytes()).hexdigest()==plan['source_sha256']
 manifest=out/'manifest.json'
 if not manifest.exists():atomic_json(manifest,dict(plan=plan,device=a.device,smoke=a.smoke,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),source_sha256={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for folder in ['src/swarm_core','learning/training'] for f in sorted((ROOT/folder).rglob('*.py'))}))
 else:
  old=json.loads(manifest.read_text());assert old['plan']==plan and old['smoke']==a.smoke
 bundle=torch.load(out/'candidate_bundle.pt' if (out/'candidate_bundle.pt').exists() else source,map_location='cpu',weights_only=True)
 overall=json.loads((out/'progress.json').read_text()) if (out/'progress.json').exists() else dict(status='training',smoke=a.smoke,stages={},steps_per_profile={v:0 for v in profiles})
 all_started=time.monotonic()
 for stage_index,(name,normal_budget) in enumerate(plan['allocation_per_profile'].items()):
  if overall['stages'].get(name,{}).get('complete'):continue
  budget=max(64,(plan['critic_only_warmup_updates']+2)*16) if a.smoke else normal_budget;rollout_length=16 if a.smoke else plan['rollout'];epochs=1 if a.smoke else plan['epochs'];stage=out/name;stage.mkdir(exist_ok=True)
  item=component(bundle,name);task=task_for(name,plan['seed']+stage_index*100000,plan);weights=read(plan.get('reward_overrides',{}).get(name,STAGES[name][1]));cls=FormationSafetyTrainingEnv if task.get('environment')=='formation_safety' else ArrivalTrainingEnv if task.get('environment')=='arrival_refinement' else (AlignmentEnv if name=='alignment' else (FormationEnv if name.startswith('formation') else MissionEnv))
  atomic_json(stage/'task.json',task);atomic_json(stage/'reward.json',weights)
  envs=[cls(task,read('configs/communication/'+profile+'.json'),weights) for profile in profiles]
  local_dim=15 if task.get('formation_pose_goal') else 13 if task.get('formation_capabilities') else item['local_dim']
  options=dict(local_dim=local_dim,critic_dim=item['critic_dim'],neighbor_dim=item['neighbor_dim'],geometry_summary=item['geometry_summary'],hold_std_scale=item.get('hold_std_scale'),local_only=item.get('local_only',False))
  args,learner,trainer=make_policy(3,device=a.device,**options);_,rollout_policy,_=make_policy(3,device='cpu',**options)
  if task.get('formation_capabilities') and item['local_dim']==9:
   if name!='formation_fresh':raise ValueError('Capabilities are supported only for fresh formation.')
   actor=dict(item['actor'])
   weight=actor['base.head.0.weight']
   actor['base.head.0.weight']=torch.cat((weight[:,:9],weight.new_zeros((weight.shape[0],4)),weight[:,9:]),dim=1)
   item.update(actor=actor,local_dim=13,contract='formation_capabilities_v1')
  if task.get('formation_pose_goal') and item['local_dim']==13:
   actor=dict(item['actor'])
   weight=actor['base.head.0.weight']
   actor['base.head.0.weight']=torch.cat((weight[:,:9],weight.new_zeros((weight.shape[0],2)),weight[:,9:]),dim=1)
   item.update(actor=actor,local_dim=15,contract='formation_pose_v1')
  learner.actor.load_state_dict(item['actor']);actor_initial=cpu_state(learner.actor)
  args.entropy_coef=0.;trainer.entropy_coef=0.;trainer.clip_param=plan['clip_parameter']
  if task.get('environment')=='arrival_refinement':trainer._use_policy_active_masks=True;trainer._use_value_active_masks=False
  for group in learner.actor_optimizer.param_groups:group['lr']=plan['actor_lr']
  learner.actor.act.action_out.logstd._bias.requires_grad_(False)
  with torch.no_grad():learner.actor.act.action_out.logstd._bias.fill_(math.log(plan['exploration_std']))
  done=0;updates=0;episodes=[0]*len(profiles);outcomes=[Counter() for _ in profiles];elapsed_before=0.
  if a.resume and (stage/'checkpoint.pt').exists():
   ck=torch.load(stage/'checkpoint.pt',map_location=a.device,weights_only=True);learner.actor.load_state_dict(ck['actor']);learner.critic.load_state_dict(ck['critic']);learner.actor_optimizer.load_state_dict(ck['actor_optimizer']);learner.critic_optimizer.load_state_dict(ck['critic_optimizer']);done=ck['steps_per_profile'];updates=ck['updates'];episodes=[e+1 for e in ck['episodes']];elapsed_before=ck['elapsed_s'];torch.set_rng_state(ck['rng_state'].cpu())
  obs=[env.reset(task['seed']+episode*17+i)[0] for i,(env,episode) in enumerate(zip(envs,episodes))];hidden,masks=hidden_inputs(batch_size,args);started=time.monotonic()
  while done<budget:
   rollout_policy.actor.load_state_dict(cpu_state(learner.actor));rollout_policy.critic.load_state_dict(cpu_state(learner.critic));rollout_policy.actor.eval();rollout_policy.critic.eval();rows=[]
   for _ in range(min(rollout_length,budget-done)):
    raw=np.concatenate([pack(o) for o in obs]);central=np.concatenate([critic_input(env) for env in envs])
    assert raw.shape[0]==batch_size and central.shape==(batch_size,item['critic_dim'])
    with torch.no_grad():values,actions,logps,_,_=rollout_policy.get_actions(central,raw,hidden,hidden,masks)
    raw_actions=actions.numpy();next_obs=[];rewards=[];terms=[];ends=[];next_centers=[];actor_active=[]
    for i,env in enumerate(envs):
     o,reward,term,trunc,info=env.step(raw_actions[i*3:(i+1)*3]);rewards.append(reward[:,None]);terms.append(float(term));ends.append(float(term or trunc));next_centers.append(critic_input(env));actor_active.append(np.asarray(getattr(env,'last_actor_active_masks',np.ones((3,1),dtype=np.float32))).copy())
     if term or trunc:
      outcomes[i][info['reason']]+=1
      with (stage/'episodes.jsonl').open('a') as f:f.write(json.dumps(dict(profile=profiles[i],stage_step=done+1,episode=episodes[i],reason=info['reason'],time_s=info['time_s']))+'\n')
      episodes[i]+=1;o=env.reset(task['seed']+episodes[i]*17+i)[0]
     next_obs.append(o)
    with torch.no_grad():nv=rollout_policy.get_values(np.concatenate(next_centers),hidden,masks).numpy()
    rows.append((central,raw,raw_actions,values.numpy(),logps.numpy(),np.asarray(rewards).reshape(batch_size,1),nv,np.repeat(terms,3)[:,None],np.repeat(ends,3)[:,None],np.asarray(actor_active).reshape(batch_size,1)));obs=next_obs;done+=1
   central,raw,actions,values,logps,rewards,nv,terms,ends,actives=[np.asarray(v) for v in zip(*rows)]
   advantages,returns=gae(rewards,values,nv,terms,ends,plan['gamma'],plan['gae_lambda']);valid_advantages=advantages[actives>0]
   advantages=(advantages-valid_advantages.mean())/(valid_advantages.std()+1e-5) if len(valid_advantages) else np.zeros_like(advantages)
   central,raw,actions,values,logps,returns,advantages,actives=[x.reshape(-1,x.shape[-1]) for x in [central,raw,actions,values,logps,returns,advantages,actives]]
   trainer.prep_training();diagnostics=[];update_actor=updates>=plan['critic_only_warmup_updates']
   for epoch in range(epochs):
    for index in np.array_split(rng.permutation(len(raw)),4):
     h,m=hidden_inputs(len(index),args);active_mask=actives[index];can_update=bool(update_actor and active_mask.sum()>0)
     if active_mask.sum()==0:active_mask=np.ones_like(active_mask)
     sample=(central[index],raw[index],h,h.copy(),actions[index],values[index],returns[index],m,active_mask,logps[index],advantages[index],None)
     result=trainer.ppo_update(sample,update_actor=can_update)
     if not all(torch.isfinite(torch.as_tensor(v)).all().item() for v in result):raise RuntimeError('Nonfinite upstream PPO output.')
     ratio=result[5].detach().clamp_min(1e-12);diagnostics.append(float(((ratio-1)-ratio.log()).mean().cpu()))
    if np.mean(diagnostics)>.03:break
   updates+=1;elapsed=elapsed_before+time.monotonic()-started
   ck=dict(actor=cpu_state(learner.actor),critic=cpu_state(learner.critic),actor_optimizer=learner.actor_optimizer.state_dict(),critic_optimizer=learner.critic_optimizer.state_dict(),steps_per_profile=done,updates=updates,episodes=episodes,elapsed_s=elapsed,rng_state=torch.get_rng_state())
   torch.save(ck,stage/'checkpoint.tmp');(stage/'checkpoint.tmp').replace(stage/'checkpoint.pt')
   previous_stages=sum(v['steps_per_profile'] for k,v in overall['stages'].items() if k!=name and v.get('complete'))
   overall['stages'][name]=dict(complete=False,steps_per_profile=done,budget_per_profile=budget,elapsed_s=elapsed,outcomes={p:dict(o) for p,o in zip(profiles,outcomes)})
   overall['steps_per_profile']={p:previous_stages+done for p in profiles};overall['active_stage']=name
   atomic_json(out/'progress.json',overall)
   record=dict(stage=name,steps_per_profile=done,total_steps_per_profile=previous_stages+done,total_environment_steps=len(profiles)*(previous_stages+done),steps_per_second=len(profiles)*done/max(elapsed,1e-9),approx_kl=float(np.mean(diagnostics)),actor_updated=update_actor,outcomes={p:dict(o) for p,o in zip(profiles,outcomes)})
   with (out/'updates.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
   if updates==1 or updates%10==0 or done==budget:print(json.dumps(record),flush=True)
  final_actor=cpu_state(learner.actor);assert any(not torch.equal(actor_initial[k],v) for k,v in final_actor.items() if 'logstd' not in k)
  if task.get('formation_capabilities'):bundle['task']['formation_capabilities']=True
  if task.get('formation_pose_goal'):bundle['task']['formation_pose_goal']=True
  item['actor']=final_actor;overall['stages'][name]['complete']=True
  bundle['status']='speed033_trained_candidate_pending_evaluation';bundle['adaptation']=dict(type='Upstream MAPPO continuation across configured communication profiles',campaign=str(out.relative_to(ROOT)),steps_per_profile=overall['steps_per_profile'],scope=plan['scope'])
  torch.save(bundle,out/'candidate_bundle.tmp');(out/'candidate_bundle.tmp').replace(out/'candidate_bundle.pt');atomic_json(out/'progress.json',overall)
 if plan.get('mission_profile'):
  profile=plan['mission_profile']
  bundle['task']['robot_defaults']['max_forward_speed_mps']=profile['max_forward_speed_mps']
  bundle['task']['team']['desired_pair_distance_m']=profile['desired_pair_distance_m']
  bundle['task']['mission']['final_heading_tolerance_rad']=profile['final_heading_tolerance_rad']
  for key in ('formation_peer_history','peer_history'):
   if key in bundle['task']:bundle['task'][key]['max_speed_mps']=profile['max_forward_speed_mps']
  if bundle.get('parking'):bundle['parking']['heading_tolerance_rad']=profile['final_heading_tolerance_rad']
  torch.save(bundle,out/'candidate_bundle.tmp')
  (out/'candidate_bundle.tmp').replace(out/'candidate_bundle.pt')
 overall['status']='training_complete';overall['elapsed_this_invocation_s']=time.monotonic()-all_started;overall['candidate_sha256']=hashlib.sha256((out/'candidate_bundle.pt').read_bytes()).hexdigest();atomic_json(out/'progress.json',overall)
 expected=max(64,(plan['critic_only_warmup_updates']+2)*16)*len(plan['allocation_per_profile']) if a.smoke else plan['total_environment_steps_per_profile']
 assert all(v==expected for v in overall['steps_per_profile'].values())
 print(json.dumps(overall),flush=True)

if __name__=='__main__':main()
