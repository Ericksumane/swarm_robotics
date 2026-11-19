"""Create a relocatable source/model runtime package without training environments."""
from pathlib import Path
import argparse,hashlib,json,shutil,subprocess,tarfile

def main():
    p=argparse.ArgumentParser();p.add_argument('--candidate',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[1];record=json.loads(Path(a.candidate).read_text());artifact=root/record['actor_bundle']
    if hashlib.sha256(artifact.read_bytes()).hexdigest()!=record['sha256']:raise ValueError('Candidate checksum mismatch.')
    out=Path(a.output);out.mkdir(exist_ok=False);dest=out/'swarm_runtime_release';dest.mkdir()
    ignore=shutil.ignore_patterns('.git','__pycache__','*.pyc','*.egg-info')
    for name in ('src','requirements','ros2_ws/src','external/on-policy/onpolicy','configs/communication','docs/deployment'):
        target=dest/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copytree(root/name,target,ignore=ignore)
    for name in ('pyproject.toml','external/on-policy/setup.py','external/on-policy/README.md','docs/mappo_upstream.md','docs/mathematical_formulation.md','docs/full_mission.md','docs/communication_model.md','docs/observation_action_contract.md','docs/reward_design.md','docs/common_release_validation.md'):
        target=dest/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/name,target)
    for path in (root/'external/on-policy').glob('LICENSE*'):shutil.copy2(path,dest/'external/on-policy'/path.name)
    if record.get('report'):
        report=(root/record['report']).resolve()
        if not report.is_relative_to((root/'docs').resolve()):raise ValueError('Candidate report must be inside project docs.')
        target=dest/report.relative_to(root);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(report,target)
        for name in ('speed033_training.md','speed033_training_results.md','arrival_refinement_100k.md'):
            source=root/'docs'/name
            if source.exists():shutil.copy2(source,dest/'docs'/name)
    scripts=['robot_mission_service.py','local_mission_actor_service.py','build_ros.sh','run_full_mission_shadow.sh','install_runtime.sh','preflight_deployment.py','smoke_runtime_install.py']
    (dest/'scripts').mkdir()
    for name in scripts:shutil.copy2(root/'scripts'/name,dest/'scripts'/name)
    (dest/'policies').mkdir();shutil.copy2(artifact,dest/'policies/mission_bundle.pt')
    manifest=json.loads((root/'configs/deployment/delivery_candidate.json').read_text());manifest.update(actor_bundle='policies/mission_bundle.pt',actor_bundle_sha256=record['sha256'],status='software_candidate_shadow_only_hardware_gated')
    (dest/'configs/deployment').mkdir(parents=True);(dest/'configs/deployment/delivery_candidate.json').write_text(json.dumps(manifest,indent=2)+'\n')
    shutil.copy2(root/'configs/deployment/acceptance_proposal.json',dest/'configs/deployment/acceptance_proposal.json')
    shutil.copy2(root/'docs/deployment/README.md',dest/'README.md')
    release=dict(schema_version=1,scope='Common CPU inference and ROS shadow runtime; no hardware adapter',policy_sha256=record['sha256'],upstream_commit=subprocess.check_output(['git','-C',str(root/'external/on-policy'),'rev-parse','HEAD'],text=True).strip(),source_commit=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),source_dirty=bool(subprocess.check_output(['git','-C',str(root),'status','--porcelain'],text=True).strip()))
    (dest/'RELEASE.json').write_text(json.dumps(release,indent=2)+'\n')
    hashes=[]
    for path in sorted(dest.rglob('*')):
        if path.is_file():hashes.append(hashlib.sha256(path.read_bytes()).hexdigest()+'  '+str(path.relative_to(dest)))
    (dest/'SHA256SUMS').write_text('\n'.join(hashes)+'\n')
    archive=out/'swarm_runtime_release.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:tar.add(dest,arcname=dest.name)
    print(json.dumps(dict(archive=str(archive),archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),policy_sha256=record['sha256']),indent=2))
if __name__=='__main__':main()
