"""Local hysteresis between frozen fresh-message and history MAPPO actors."""
import numpy as np

class LocalFormationExpertRouter:
    def __init__(self,max_normalized_age=None):
        if max_normalized_age is not None and (not np.isfinite(max_normalized_age) or not 0<=max_normalized_age<=1):raise ValueError('Invalid fresh actor age threshold.')
        self.max_normalized_age=max_normalized_age;self.reset()
    def reset(self):self.missing_steps=0;self.fresh_steps=0;self.history=False;self.close=False
    def select(self,observation):
        fresh=bool(np.all(observation['neighbor_mask']==1))
        if fresh and self.max_normalized_age is not None:fresh=bool(np.all(observation['neighbor_features'][:,6]<=self.max_normalized_age+1e-6))
        self.missing_steps=0 if fresh else self.missing_steps+1
        self.fresh_steps=self.fresh_steps+1 if fresh else 0
        history=observation['formation_history'];mask=history['neighbor_mask'].astype(bool)
        distances=np.linalg.norm(history['neighbor_features'][mask,:2],axis=1)
        if len(distances) and distances.min()<.85:self.close=True
        elif len(distances)==2 and distances.min()>1.05:self.close=False
        if self.missing_steps>=4:self.history=True
        elif self.fresh_steps>=15:self.history=False
        return 'formation' if self.history or self.close else 'formation_fresh'
