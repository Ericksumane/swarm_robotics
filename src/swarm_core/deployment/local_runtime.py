"""One robot's complete mission state; no simulator or remote truth access."""
from copy import deepcopy
import math
import numpy as np
from swarm_core.observations import append_formation_geometry
from swarm_core.communication.receiver import Receiver
from swarm_core.communication.history import build_history_observation
from swarm_core.communication.message import StateMessage
from swarm_core.observations import build_local_features,build_neighbor_observation,formation_pose_features
from swarm_core.history_startup import HistoryStartupCoordinator
from swarm_core.goal_stop import limit_forward_for_arrival
from .protocol import PeerEnvelope,token

class LocalMissionRuntime:
    def __init__(self,task,robot_id,mission_id,session_id,frame_id,policy,peer_timeout_s=5.,odometry_timeout_s=.3):
        if type(robot_id) is not int or robot_id not in (0,1,2) or task['team']['num_agents']!=3:raise ValueError('Only the validated three-robot team is supported.')
        if not task.get('history_startup') or not task.get('formation_peer_history'):raise ValueError('History mission task required.')
        if not np.isfinite([peer_timeout_s,odometry_timeout_s]).all() or min(peer_timeout_s,odometry_timeout_s)<=0:raise ValueError('Invalid watchdog times.')
        self.task=deepcopy(task);self.robot_id=robot_id;self.mission_id=token(mission_id);self.session_id=token(session_id);self.frame_id=token(frame_id);self.policy=policy
        self.peer_timeout_s=peer_timeout_s;self.odometry_timeout_s=odometry_timeout_s;self.dt=task['simulation']['time_step_s']
        self.receiver=Receiver(robot_id,.5);self.peers=[i for i in range(3) if i!=robot_id]
        self.coordinator=HistoryStartupCoordinator(task['history_startup'],self.dt,
            desired_distance_m=task['team']['desired_pair_distance_m'])
        direction=np.asarray(task['mission']['direction_world'],dtype=float)
        if direction.shape!=(2,) or not np.isfinite(direction).all() or np.linalg.norm(direction)==0:raise ValueError('Invalid mission direction.')
        self.direction=direction/np.linalg.norm(direction);self.peer_sessions={};self.peer_seen={};self.sequence=0
        self.phase=0;self.goal=None;self.start=None;self.ready=False;self.proposal=None;self.committed=False
        self.fault=None;self.rejected={};self.last_tick=None;self.started_at=None;self.last_command=np.zeros(2);self.own_state=None;self.arrived_once=False;self.move_started_at=None

    def fail(self,reason):
        if self.fault is None:self.fault=reason
        self.ready=False;self.committed=False;self.proposal=None

    def accept_peer(self,value,now):
        try:
            envelope=PeerEnvelope.from_dict(value)
            if envelope.mission_id!=self.mission_id:raise ValueError('wrong_mission')
            if envelope.frame_id!=self.frame_id:raise ValueError('wrong_frame')
            message=envelope.state;sender=message.sender_id
            if sender==self.robot_id:raise ValueError('self_message')
            age=now-message.generated_at_s
            if not math.isfinite(now) or age<0 or age>self.peer_timeout_s:raise ValueError('invalid_message_time')
            previous=self.peer_sessions.get(sender)
            if previous is not None and previous!=envelope.session_id:
                self.fail('peer_session_changed');raise ValueError('peer_session_changed')
            if envelope.faulted:self.fail('peer_faulted')
            accepted=self.receiver.accept(message,now)
            if accepted:self.peer_sessions[sender]=envelope.session_id;self.peer_seen[sender]=now
            return accepted
        except (ValueError,TypeError,KeyError) as error:
            reason=str(error);self.rejected[reason]=self.rejected.get(reason,0)+1
            return False

    def observation(self,own,now):
        o=build_neighbor_observation(own,self.receiver,self.peers,now,predict_to_now=self.task.get('observation_contract')=='mission_local_goal_time_aligned_v4')
        c,s=math.cos(own.heading),math.sin(own.heading);rotation=np.array([[c,s],[-s,c]])
        goal=np.zeros(2) if self.goal is None else rotation@(self.goal-[own.x,own.y])
        o['goal_distance_m']=None if self.goal is None else float(np.linalg.norm(goal))
        horizon=self.task['goal_observation_horizon_m'];norm=float(np.linalg.norm(goal));goal*=min(1.,horizon/max(norm,1e-12))
        desired=self.task['team']['desired_pair_distance_m']
        o['neighbor_features']=append_formation_geometry(
            o['neighbor_features'],o['neighbor_mask'],desired)
        o['desired_pair_distance_m']=desired
        local=build_local_features(own,self.task['robot_defaults'],desired)
        o['local_features']=np.concatenate([local,np.eye(3)[self.phase],goal,rotation@self.direction]).astype(np.float32)
        if self.task.get('formation_pose_goal'):
            o['formation_pose']=formation_pose_features(own.heading,self.direction)
        if self.task.get('formation_capabilities'):
            from swarm_core.mixed_commands import FORMATION_CAPABILITIES
            o['formation_capabilities']=FORMATION_CAPABILITIES[self.robot_id]
        o['formation_history']=build_history_observation(own,self.receiver,self.peers,now,desired,self.task['formation_peer_history'])
        o['startup_committed']=self.committed;o['control_time_s']=now
        return o

    def tick(self,own,now,odometry_stamp):
        if not math.isfinite(now) or now<0:raise ValueError('Invalid control clock.')
        if self.started_at is None:self.started_at=now
        if self.last_tick is not None and (now<=self.last_tick or now-self.last_tick>.5):self.fail('control_clock_or_deadline')
        self.last_tick=now;self.own_state=own
        values=[own.x,own.y,own.heading,own.speed,own.yaw_rate,odometry_stamp]
        if not np.isfinite(values).all() or now-odometry_stamp<0 or now-odometry_stamp>self.odometry_timeout_s:self.fail('invalid_or_stale_odometry')
        if len(self.peer_sessions)==2 and any(now-seen>self.peer_timeout_s for seen in self.peer_seen.values()):self.fail('peer_timeout')
        if self.fault:return self._output(own,now,odometry_stamp,np.zeros(2),'fault_stop')
        o=self.observation(own,now)
        if self.phase==0:
            self.ready,self.proposal,start=self.coordinator.update(own,self.receiver,o['formation_history'],now,self.direction)
            self.committed=self.coordinator.committed
            if start:
                self.phase=1;self.move_started_at=now;self.start=np.array([own.x,own.y]);self.goal=self.start+self.task['mission']['translation_distance_m']*self.direction
        elif self.phase==1 and o['goal_distance_m']<=self.task['mission']['hold_entry_tolerance_m']:self.phase=2
        elif self.phase==2 and o['goal_distance_m']>self.task['mission']['hold_exit_tolerance_m']:self.phase=1
        o=self.observation(own,now)
        if len(self.peer_sessions)<2:
            action=np.zeros(2);mode='awaiting_peers'
        else:action,mode=self.policy.act_local(o,self.robot_id)
        action=limit_forward_for_arrival(
            action,own,self.goal,self.task['robot_defaults'],self.dt,
            self.task.get('braking_envelope_mps2'))
        if mode=='parked':self.arrived_once=True
        if now-self.started_at>=self.task['simulation']['episode_time_limit_s'] and not self.arrived_once:
            self.fail('mission_deadline');action=np.zeros(2);mode='fault_stop'
        return self._output(own,now,odometry_stamp,action,mode)

    def _output(self,own,now,stamp,action,mode):
        limits=self.task['robot_defaults'];a=np.asarray(action,dtype=float)
        if a.shape!=(2,) or not np.isfinite(a).all():self.fail('invalid_actor_action');a=np.zeros(2);mode='fault_stop'
        a=np.clip(a,-1.,1.);target=np.array([a[0]*(limits['max_forward_speed_mps'] if a[0]>=0 else limits['max_reverse_speed_mps']),a[1]*limits['max_yaw_rate_radps']])
        if self.fault:command=np.zeros(2)
        else:
            maximum=np.array([limits['max_linear_acceleration_mps2'],limits['max_angular_acceleration_radps2']])*self.dt
            command=self.last_command+np.clip(target-self.last_command,-maximum,maximum)
        self.last_command=command
        packet=StateMessage(sender_id=self.robot_id,sequence=self.sequence,generated_at_s=now if self.fault else float(stamp),x=float(own.x) if math.isfinite(own.x) else 0.,y=float(own.y) if math.isfinite(own.y) else 0.,heading=float(own.heading) if math.isfinite(own.heading) else 0.,speed=float(own.speed) if math.isfinite(own.speed) else 0.,yaw_rate=float(own.yaw_rate) if math.isfinite(own.yaw_rate) else 0.,payload_size_bytes=72,formation_ready=bool(self.ready),start_at_s=self.proposal,start_committed=bool(self.committed))
        self.sequence+=1;envelope=PeerEnvelope(self.mission_id,self.session_id,self.frame_id,packet,bool(self.fault))
        return dict(time_s=now,move_started_at_s=self.move_started_at,locally_arrived_once=self.arrived_once,command=command.tolist(),action=a.tolist(),mode=mode,phase=self.phase,fault=self.fault,peer=envelope.to_dict(),start_m=None if self.start is None else self.start.tolist(),goal_m=None if self.goal is None else self.goal.tolist(),goal_error_m=None if self.goal is None or not np.isfinite([own.x,own.y]).all() else float(np.linalg.norm(self.goal-[own.x,own.y])),rejected=dict(self.rejected))
