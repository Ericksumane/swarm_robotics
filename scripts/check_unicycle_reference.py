"""Independent unicycle-equation check, informed by the VMAS DiffDrive reference.

Reference inspected: https://github.com/proroklab/VectorizedMultiAgentSimulator/
blob/main/vmas/simulator/dynamics/diff_drive.py
No external engine code or controller is imported. Verify constant-command motion
against independently integrated dx=v*cos(theta), dy=v*sin(theta), dtheta=omega.
"""
import json,math
import numpy as np
from swarm_core.dynamics import RobotState,step

def main():
    limits=json.load(open('configs/tasks/formation_v1.json'))['robot_defaults']
    def derivative(p,v,w):return np.array([v*math.cos(p[2]),v*math.sin(p[2]),w])
    for v in (-.12,0.,.2):
        for w in (-.6,0.,.6):
            for heading in (-2.,0.,1.):
                state=RobotState(.3,-.7,heading,v,w)
                reference=np.array([state.x,state.y,state.heading])
                action=[v/(limits['max_forward_speed_mps'] if v>=0 else limits['max_reverse_speed_mps']),w/limits['max_yaw_rate_radps']]
                for _ in range(30):
                    state=step(state,action,limits,.1)
                    # Ten smaller RK4 intervals provide a separate numerical reference.
                    for _ in range(10):
                        h=.01;k1=derivative(reference,v,w)
                        k2=derivative(reference+h*k1/2,v,w)
                        k3=derivative(reference+h*k2/2,v,w)
                        k4=derivative(reference+h*k3,v,w)
                        reference+=h*(k1+2*k2+2*k3+k4)/6
                    np.testing.assert_allclose([state.x,state.y],reference[:2],atol=1e-9,rtol=0.)
                    angle=(state.heading-reference[2]+math.pi)%(2*math.pi)-math.pi
                    assert abs(angle)<1e-10
    print('Straight, turning and reversing unicycle motion versus independent RK4: PASSED')

if __name__=='__main__':main()
