"""Explicit synthetic speed experiments; no implicit policy retraining."""
from copy import deepcopy
import math

def scaled_linear_speed(task, scale=1.0):
    if not math.isfinite(scale) or not 1.0 <= scale <= 1.15:
        raise ValueError("Experimental linear speed scale must be within [1.0, 1.15].")
    result=deepcopy(task)
    limits=result['robot_defaults']
    for key in ['max_forward_speed_mps','max_reverse_speed_mps']:
        limits[key]*=scale
    for key in ['formation_peer_history','peer_history']:
        if key in result:result[key]['max_speed_mps']=max(limits['max_forward_speed_mps'],limits['max_reverse_speed_mps'])
    result['experimental_linear_speed_scale']=scale
    return result
