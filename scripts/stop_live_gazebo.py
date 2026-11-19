"""Stop only the verified project-owned live driver; it cleans up its children."""
from pathlib import Path
import json,os,signal
root=Path(__file__).resolve().parents[1];metadata=json.loads((root/'experiments/runs/gazebo_live/latest.json').read_text());pid=metadata['pid']
try:
 command=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode();cwd=Path(f'/proc/{pid}/cwd').resolve()
 if 'scripts/gazebo_live.py' not in command or cwd!=root:raise RuntimeError('Recorded PID no longer identifies this project live driver.')
 os.kill(pid,signal.SIGINT);print('Requested clean shutdown of live Gazebo driver',pid)
except FileNotFoundError:print('The recorded live simulation has already stopped.')
