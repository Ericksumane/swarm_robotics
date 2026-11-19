"""Report install readiness separately from unverified physical deployment."""
from pathlib import Path
import argparse,hashlib,json,platform,sys

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['shadow','hardware'],default='shadow');a=p.parse_args()
    root=Path(__file__).resolve().parents[1];path=root/'configs/deployment/delivery_candidate.json';checks={}
    try:
        d=json.loads(path.read_text());bundle=root/d['actor_bundle']
        checks['policy_checksum']=bundle.is_file() and hashlib.sha256(bundle.read_bytes()).hexdigest()==d['actor_bundle_sha256']
        checks['validated_team_size']=d['supported_team_size']==3
        checks['inference_python']=(root/'.venv/bin/python').is_file()
        checks['local_service']=(root/'scripts/robot_mission_service.py').is_file()
        checks['typed_ros_interface']=(root/'ros2_ws/src/swarm_interfaces/msg/PeerState.msg').is_file()
        checks['ros_runtime_built']=(root/'ros2_ws/install/swarm_runtime/lib/swarm_runtime/full_mission_node').is_file()
        checks['ros_jazzy_installed']=Path('/opt/ros/jazzy/setup.bash').is_file()
        checks['tested_cpu_architecture']=platform.machine()=='x86_64' and platform.system()=='Linux'
    except Exception as error:checks['manifest']=False;checks['error']=str(error)
    blockers=[]
    if a.mode=='hardware':
        blockers=['Exact robot profiles are unverified.','Shared localization and clock accuracy are unverified.','No platform actuator adapter is provided in this shadow release.','Independent physical stopping has not been validated.','Policy acceptance on calibrated heterogeneous dynamics is not established.']
    passed=all(value is True for value in checks.values()) and not blockers
    print(json.dumps(dict(mode=a.mode,passed=passed,checks=checks,hardware_blockers=blockers,scope='Software checks do not authorize physical actuation.'),indent=2))
    return 0 if passed else 2
if __name__=='__main__':sys.exit(main())
