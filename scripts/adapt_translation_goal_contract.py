"""Version an observation adapter explicitly; do not alter or retrain weights."""
from pathlib import Path
import hashlib,json,torch

def main():
 root=Path(__file__).resolve().parents[1]
 source=root/'experiments/runs/translation_promotion_001/translation_short_relative_001_checkpoint_0030208'
 out=root/'experiments/runs/translation_local_goal_001';out.mkdir(exist_ok=False)
 manifest=json.loads((source/'manifest.json').read_text())
 manifest['source_training_task']=manifest['task']
 manifest['task']=json.loads((root/'configs/tasks/translate_stop_local_goal.json').read_text())
 manifest['contract']='mission_local_goal_v3'
 manifest['adaptation']=dict(source=str(source.relative_to(root)),source_checkpoint_sha256=hashlib.sha256((source/'checkpoint.pt').read_bytes()).hexdigest(),weights_changed=False,description='Observation-only local goal horizon 0.5 m; actual target remains 5 m.')
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
 checkpoint=torch.load(source/'checkpoint.pt',map_location='cpu',weights_only=True)
 checkpoint['source_contract']=checkpoint['contract'];checkpoint['contract']='mission_local_goal_v3'
 torch.save(checkpoint,out/'checkpoint.pt')
 print(out)

if __name__=='__main__':main()
