"""Critic/training-only constant-velocity closest-approach reward metric."""
import numpy as np

def closing_risk(states,radius,horizon,margin):
    if not np.isfinite([radius,horizon,margin]).all() or min(radius,horizon,margin)<=0:
        raise ValueError('Positive finite collision-risk settings required.')
    values=np.asarray([[s.x,s.y,s.heading,s.speed] for s in states])
    if len(values)<2 or not np.isfinite(values).all():raise ValueError('Finite team states required.')
    positions=values[:,:2];velocities=values[:,3,None]*np.column_stack([np.cos(values[:,2]),np.sin(values[:,2])])
    risks=[]
    for i in range(len(states)):
        for j in range(i):
            p=positions[i]-positions[j];v=velocities[i]-velocities[j]
            t=np.clip(-np.dot(p,v)/max(float(np.dot(v,v)),1e-12),0.,horizon)
            clearance=float(np.linalg.norm(p+t*v))-2*radius
            risks.append(max(0.,(margin-clearance)/margin)**2)
    return float(max(risks))
