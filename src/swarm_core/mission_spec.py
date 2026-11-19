"""Explicit profile-dependent completion criteria for experimental missions."""
from copy import deepcopy
import math

def completion_task(task,profile):
    result=deepcopy(task)
    overrides=result.get('completion_by_profile',{}).get(profile,{})
    if set(overrides)-{'formation_tolerance_m'}:raise ValueError('Unsupported completion override.')
    if overrides:
        value=overrides['formation_tolerance_m']
        if not math.isfinite(value) or not .1<=value<=.2:raise ValueError('Formation completion tolerance outside experimental range.')
        result['mission'].update(overrides)
    deadline=result.get('deadline_by_profile',{}).get(profile)
    if deadline is not None:
        if not math.isfinite(deadline) or not 120<=deadline<=600:raise ValueError('Invalid profile mission deadline.')
        result['simulation']['episode_time_limit_s']=deadline
    result['effective_completion_profile']=profile
    return result
