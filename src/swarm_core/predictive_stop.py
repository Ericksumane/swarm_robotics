"""Optional local translational stop gate using received peer estimates.

This gate does not steer or supply expert actions. It can veto translation
while leaving policy yaw unchanged. The finite-horizon prediction is a heuristic,
not a formal collision-avoidance guarantee, especially with missing information,
non-circular footprints or uncalibrated dynamics.
"""
import math
from types import SimpleNamespace
import numpy as np
from swarm_core.dynamics import RobotState,step
from swarm_core.communication.prediction import predict_received_pose

class PredictiveStopGate:
    def __init__(self,settings):
        self.veto_when_missing_form=settings.get('veto_when_missing_form',settings.get('veto_when_missing',True))
        self.veto_when_missing=settings.get('veto_when_missing',True);self.uncertainty_in_move=settings.get('uncertainty_in_move',True)
        if type(self.veto_when_missing) is not bool or type(self.uncertainty_in_move) is not bool:raise ValueError('Stop gate options must be boolean.')
        self.horizon=settings['horizon_s'];self.dt=settings['prediction_step_s'];self.margin=settings['clearance_margin_m'];self.bound_scale=settings['history_bound_scale_m']
        if not np.isfinite([self.horizon,self.dt,self.margin,self.bound_scale]).all() or min(self.horizon,self.dt,self.margin,self.bound_scale)<=0:raise ValueError('Invalid stop-gate settings.')
    def apply(self,action,observation):
        action=np.asarray(action,dtype=np.float32)
        if abs(action[0])<1e-9:return action,False
        history=observation['formation_history'];rows=np.asarray(history['neighbor_features']);mask=np.asarray(history['neighbor_mask'])
        def stop():
            result=action.copy();result[0]=0.;return result,True
        if rows.shape[0]!=2:raise ValueError('Exactly two peer rows required.')
        local=np.asarray(observation['local_features'])
        missing_veto=self.veto_when_missing_form if local[9]>.5 else self.veto_when_missing
        if not np.all(mask==1) and missing_veto:return stop()
        rows=rows[mask.astype(bool)]
        if len(rows)==0:return action,False
        local=np.asarray(observation['local_features']);limits=dict(max_forward_speed_mps=float(local[2]),max_reverse_speed_mps=float(local[3]),max_yaw_rate_radps=float(local[4]),max_linear_acceleration_mps2=float(local[5]),max_angular_acceleration_radps2=float(local[6]))
        if not -limits['max_reverse_speed_mps']<=local[0]<=limits['max_forward_speed_mps'] or abs(local[1])>limits['max_yaw_rate_radps']:
            return np.zeros(2,dtype=np.float32),True
        own=RobotState(speed=float(local[0]),yaw_rate=float(local[1]));diameter=2*float(local[7])
        peers=[SimpleNamespace(x=float(r[0]),y=float(r[1]),heading=math.atan2(r[2],r[3]),speed=float(r[4]),yaw_rate=float(r[5])) for r in rows]
        bounds=rows[:,10]*self.bound_scale
        if local[9]<.5 and not self.uncertainty_in_move:bounds=np.zeros_like(bounds)
        for index in range(math.ceil(self.horizon/self.dt)):
            own=step(own,action,limits,self.dt);t=(index+1)*self.dt
            for peer,bound in zip(peers,bounds):
                x,y,_=predict_received_pose(peer,t)
                if math.hypot(x-own.x,y-own.y)-diameter-float(bound)<self.margin:return stop()
        return action,False
