"""Gaussian exploration scaled by observable mission phase, without mean control."""
import math
import torch
from onpolicy.algorithms.utils.distributions import DiagGaussian,FixedNormal

class PhaseScaledGaussian(DiagGaussian):
 def __init__(self,inputs_dim,action_dim,use_orthogonal,gain,hold_std_scale):
  if not math.isfinite(hold_std_scale) or not 0<hold_std_scale<=1:
   raise ValueError('HOLD exploration scale must lie in (0,1].')
  super().__init__(inputs_dim,action_dim,use_orthogonal,gain)
  self.hold_std_scale=hold_std_scale
 def forward(self,x):
  encoded,hold=x[:,:-1],x[:,-1:]
  mean=self.fc_mean(encoded)
  std=self.logstd(torch.zeros_like(mean)).exp()
  scale=torch.where(hold>.5,torch.full_like(hold,self.hold_std_scale),torch.ones_like(hold))
  return FixedNormal(mean,std*scale)
