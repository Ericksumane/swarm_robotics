"""Bounded peer dead reckoning from accepted packets, never simulator truth."""
import math
import numpy as np
from .prediction import predict_received_pose

CONTRACT = "formation_history_v5"

def build_history_observation(own_state, receiver, neighbor_ids, now_s, desired_distance, settings):
    """Return eleven features and an estimate-availability mask per peer.

    First nine: body-relative predicted pose, reported velocities, age/fresh TTL,
    distance/desired and signed distance error/desired. Last two: fresh flag and
    position-error bound / configured maximum. Availability does NOT mean fresh.
    The acceleration bound assumes held last measured rates; it is not a
    calibrated covariance or a collision-avoidance guarantee.
    """
    horizon = settings["horizon_s"]
    maximum = settings["max_position_error_m"]
    acceleration = settings["max_linear_acceleration_mps2"]
    yaw_limit = settings["max_yaw_rate_radps"]
    speed_limit = settings["max_speed_mps"]
    initial_error = settings["initial_position_error_m"]
    values=[horizon, maximum, acceleration, yaw_limit, speed_limit, initial_error, desired_distance]
    if not np.isfinite(values).all() or min(values[:5]) <= 0 or initial_error < 0 or desired_distance <= 0:
        raise ValueError("Invalid history settings.")
    own=np.asarray([own_state.x,own_state.y,own_state.heading,own_state.speed,own_state.yaw_rate])
    if not np.isfinite(own).all(): raise ValueError("Own state must be finite.")
    ids=tuple(sorted(neighbor_ids))
    if len(set(ids))!=len(ids) or any(type(i) is not int or i<0 or i==receiver.robot_id for i in ids):
        raise ValueError("Invalid neighbor IDs.")
    relative_risk=settings.get("include_relative_risk",False)
    if type(relative_risk) is not bool:raise ValueError("Risk feature switch must be boolean.")
    features=np.zeros((len(ids),13 if relative_risk else 11),np.float32)
    mask=np.zeros(len(ids),np.float32)
    fresh_mask=np.zeros(len(ids),np.float32)
    sequences=np.full(len(ids),-1,dtype=np.int64)
    ages=np.full(len(ids),np.nan)
    bounds=np.full(len(ids),np.nan)
    c,s=math.cos(own_state.heading),math.sin(own_state.heading)
    for row,sender in enumerate(ids):
        message,age,fresh=receiver.read(sender,now_s)
        if message is None: continue
        # Integral of a speed-change bound plus a heading-change bound. Cap
        # each by the maximum discrepancy between two admissible velocities.
        linear=min(.5*acceleration*age**2,2*speed_limit*age)
        angular=min(speed_limit*yaw_limit*age**2,2*speed_limit*age)
        error=initial_error+linear+angular
        sequences[row]=message.sequence
        ages[row]=age;bounds[row]=error;fresh_mask[row]=float(fresh)
        if age>horizon or error>maximum: continue
        x,y,heading=predict_received_pose(message,age)
        dx,dy=x-own_state.x,y-own_state.y
        angle=heading-own_state.heading;distance=math.hypot(dx,dy)
        features[row,:11]=[c*dx+s*dy,-s*dx+c*dy,math.sin(angle),math.cos(angle),
            message.speed,message.yaw_rate,age/receiver.max_age_s,
            distance/desired_distance,distance/desired_distance-1.,float(fresh),error/maximum]
        if relative_risk:
            p=np.array([features[row,0],features[row,1]],dtype=float)
            v=np.array([message.speed*math.cos(angle)-own_state.speed,message.speed*math.sin(angle)])
            closing=-float(np.dot(p,v))/max(distance,1e-9)
            horizon=settings.get('relative_risk_horizon_s',2.)
            diameter=settings.get('footprint_diameter_m',.5)
            if not np.isfinite([horizon,diameter]).all() or min(horizon,diameter)<=0:raise ValueError('Invalid relative-risk settings.')
            closest_t=float(np.clip(-np.dot(p,v)/max(float(np.dot(v,v)),1e-12),0.,horizon))
            features[row,11:]=[closing,float(np.linalg.norm(p+closest_t*v))-diameter]
        mask[row]=1.
    return dict(neighbor_ids=ids,neighbor_features=features,neighbor_mask=mask,
        neighbor_fresh_mask=fresh_mask,neighbor_sequence=sequences,neighbor_age_s=ages,neighbor_position_error_bound_m=bounds)
