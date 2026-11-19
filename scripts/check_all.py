"""Run project checks without shell expansion or optional network services."""
from pathlib import Path
import subprocess
import sys

def main():
    root=Path(__file__).resolve().parents[1]
    # These checks require ROS, external models, or explicit artifact arguments.
    manual_checks = {
        'check_actor_export.py',
        'check_go2_locomotion_contract.py',
        'check_mixed_actor_shadow.py',
        'check_wheeltec_waypoint_pilot.py',
    }
    checks = sorted(p for p in (root/'scripts').glob('check_*.py')
                    if p.name != 'check_all.py' and p.name not in manual_checks)
    for path in checks:
        print('CHECK:',path.name,flush=True)
        subprocess.run([sys.executable,str(path)],cwd=root,check=True)
    print(f'All {len(checks)} checks passed.',flush=True)

if __name__=='__main__': main()
