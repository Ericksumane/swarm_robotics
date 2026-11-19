"""Reproducible episode-level domain randomization of synthetic networks."""
from copy import deepcopy
import numpy as np

class ProfileSampler:
 def __init__(self,settings):
  self.profiles=deepcopy(settings['profiles']);self.probabilities=np.asarray(settings['probabilities'],dtype=float)
  if not self.profiles or self.probabilities.shape!=(len(self.profiles),) or not np.isfinite(self.probabilities).all() or np.any(self.probabilities<0) or not np.isclose(self.probabilities.sum(),1.):raise ValueError('Invalid communication mixture.')
 def sample(self,seed):
  rng=np.random.default_rng(np.random.SeedSequence([seed,73219]))
  index=int(rng.choice(len(self.profiles),p=self.probabilities))
  return deepcopy(self.profiles[index])
