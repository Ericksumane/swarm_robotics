"""Predict a received pose using only payload rates and original message age."""
import math

def predict_received_pose(message,age_s):
 if not math.isfinite(age_s) or age_s<0:raise ValueError('Message age must be finite and nonnegative.')
 if age_s==0:return message.x,message.y,message.heading
 turn=message.yaw_rate*age_s;half=turn/2
 sinc=1. if half==0 else math.sin(half)/half
 travel=message.speed*age_s*sinc;mid=message.heading+half
 return (message.x+travel*math.cos(mid),message.y+travel*math.sin(mid),(message.heading+turn+math.pi)%(2*math.pi)-math.pi)
