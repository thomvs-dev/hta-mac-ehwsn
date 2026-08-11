"""Generate the compute-bounded one-seed Step 3 probe notebook."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def cell(kind, source):
    result = {"cell_type": kind, "metadata": {}, "source": source.splitlines(True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cells = [
        cell("markdown", """# HTA-MAC Step 3 — bounded 100-episode decision probe

This replaces the withdrawn multi-thousand-episode notebook. It runs one frozen configuration, one optimizer seed, and 100 episodes. Full training and confirmation seeds are deliberately unavailable.

The notebook stops unless the persisted checkpoint passes internal QoS/risk gates and then shows a measurable paired FND or packets/J gain against energy-proportional MAC on development seeds 2400–2404.
"""),
        cell("code", f"""# Frozen settings. Change only paths/download preference.
BUNDLE_PATH = '/content/HTA_MAC_Step3_Bounded_Probe_Bundle_20260810.zip'
DRIVE_OUTPUT_DIR = '/content/drive/MyDrive/HTA_MAC_Step3_Bounded_Probe_20260810'
OPTIMIZER_SEED = 5599
TRAINING_SEEDS = [2400]
EVALUATION_SEEDS = [2400, 2401, 2402, 2403, 2404]
EPISODES = 100
HORIZON = 1200
EXPECTED_BUNDLE_SHA256 = '{args.bundle_sha256}'
DOWNLOAD_RESULTS_WHEN_COMPLETE = True
FULL_TRAINING_AUTHORIZED = False
assert OPTIMIZER_SEED == 5599 and EPISODES == 100
assert FULL_TRAINING_AUTHORIZED is False
assert not set(EVALUATION_SEEDS) & set(range(3100, 3105))
assert not set(EVALUATION_SEEDS) & set(range(3400, 3405))
"""),
        cell("code", """# GPU, Drive, checksum, collision-safe extraction, and manifest verification.
import glob, hashlib, json, shutil, subprocess, sys, zipfile
from pathlib import Path, PurePosixPath
import torch
if not torch.cuda.is_available(): raise RuntimeError('Select a GPU runtime before running this notebook.')
from google.colab import drive
if not Path('/content/drive/MyDrive').is_dir():
    try: drive.mount('/content/drive')
    except Exception as exc: raise RuntimeError('Drive authentication failed before training; reconnect and retry.') from exc
bundle = Path(BUNDLE_PATH)
if not bundle.is_file():
    candidates = [Path(p) for p in glob.glob('/content/*Step3*Bounded*Probe*Bundle*.zip')]
    if len(candidates) != 1: raise FileNotFoundError(f'Expected one uploaded bundle, found {candidates}')
    bundle = candidates[0]
digest = hashlib.sha256(bundle.read_bytes()).hexdigest()
assert digest == EXPECTED_BUNDLE_SHA256, (digest, EXPECTED_BUNDLE_SHA256)
WORK = Path('/content/hta_mac_step3_bounded_probe')
if WORK.exists(): shutil.rmtree(WORK)
WORK.mkdir()
with zipfile.ZipFile(bundle) as archive:
    normalized = [(m, m.filename.replace('\\\\', '/').rstrip('/')) for m in archive.infolist()]
    names = {name for _,name in normalized if name}
    directories = {str(parent) for name in names for parent in PurePosixPath(name).parents if str(parent) != '.'}
    for member,name in normalized:
        path = PurePosixPath(name)
        if not name or path.is_absolute() or '..' in path.parts: raise RuntimeError(f'Unsafe ZIP member: {member.filename}')
        target = WORK.joinpath(*path.parts)
        if member.is_dir() or name in directories:
            if target.exists() and not target.is_dir(): raise RuntimeError(f'ZIP collision: {name}')
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.is_dir(): raise RuntimeError(f'ZIP collision: {name}')
            with archive.open(member) as src, target.open('wb') as dst: shutil.copyfileobj(src,dst)
stage2=WORK/'stage2'; repo=stage2/'hta-mac'; upstream=stage2/'final_repo'
manifest_path=stage2/'COLAB_STEP3_BOUNDED_PROBE_MANIFEST.json'
manifest=json.loads(manifest_path.read_text(encoding='utf-8-sig'))
assert manifest['optimizer_seed']==OPTIMIZER_SEED and manifest['episodes']==EPISODES
assert manifest['full_training_authorized'] is False
for entry in manifest['files']:
    target=stage2/entry['path']
    assert target.is_file() and target.stat().st_size==entry['bytes'],entry['path']
    assert hashlib.sha256(target.read_bytes()).hexdigest()==entry['sha256'],entry['path']
DRIVE_ROOT=Path(DRIVE_OUTPUT_DIR); DRIVE_ROOT.mkdir(parents=True,exist_ok=True)
shutil.copy2(manifest_path,DRIVE_ROOT/manifest_path.name)
print('Verified',bundle.name,digest,'| GPU',torch.cuda.get_device_name(0),'| Torch',torch.__version__)
"""),
        cell("code", """# Validation and exact-runtime contract; failure stops before training.
subprocess.run([sys.executable,'-m','pip','install','-q','gymnasium','torch-geometric','scipy','pyyaml','pytest'],check=True)
subprocess.run([sys.executable,'-B','-m','compileall','-q',str(repo),str(upstream)],check=True)
test=subprocess.run([sys.executable,'-B','-m','pytest','validation','-q','-p','no:cacheprovider'],cwd=repo,text=True,capture_output=True)
print(test.stdout); print(test.stderr,file=sys.stderr)
if test.returncode: raise RuntimeError('Validation failed; training is forbidden.')
sys.path.insert(0,str(repo))
from core.runtime_fingerprint import make_runtime_contract
runtime=make_runtime_contract(require_cuda=True)
runtime_path=DRIVE_ROOT/'STEP3_BOUNDED_RUNTIME_CONTRACT.json'
runtime_path.write_text(json.dumps(runtime,indent=2)+'\\n')
"""),
        cell("code", """# Verify the locally completed no-learning headroom decision.
headroom=json.loads((stage2/'headroom_evidence/STEP3_BOUNDED_PROBE_DECISION_20260810.json').read_text())
assert headroom['status']=='one_bounded_100_episode_probe_justified'
assert headroom['overall_pass'] is True and headroom['full_training_authorized'] is False
print('Bounded probe justified by:',headroom['selected_diagnostic_policy'])
print('This authorizes 100 episodes only; it is not a performance claim.')
"""),
        cell("code", """# One frozen risk/QoS configuration; no hyperparameter sweep.
profile=repo/'config/paper_aligned_hasani2025_b16_qos_repaired.json'
risk=repo/'config/step3_v3_risk_weight_5.json'
qos=repo/'config/step3_v3_qos_ema_floor_candidate.json'
scale=DRIVE_ROOT/'STEP3_BOUNDED_RETURN_SCALE.json'
subprocess.run([sys.executable,'-B','experiments/calibrate_step3_v3_return_scale.py','--ch-risk-config',str(risk),
    '--environment-profile',str(profile),'--qos-constraint-config',str(qos),'--development-seeds','2400',
    '--max-steps',str(HORIZON),'--rollouts','20','--output',str(scale)],cwd=repo,check=True)
mechanism=DRIVE_ROOT/'STEP3_BOUNDED_MECHANISM.json'
subprocess.run([sys.executable,'-B','experiments/probe_step3_mechanism.py','--environment-profile',str(profile),
    '--ch-risk-config',str(risk),'--seeds',','.join(map(str,EVALUATION_SEEDS)),'--max-steps',str(HORIZON),
    '--output',str(mechanism)],cwd=repo,check=True)
probe_name='step3_bounded_foundation_probe'
subprocess.run([sys.executable,'-B','experiments/train_step3_v3_probe.py','--ch-risk-config',str(risk),
    '--step3-qos-config',str(qos),'--runtime-contract',str(runtime_path),'--episodes','1','--max-steps','5',
    '--development-seeds','2400','--optimizer-seed','20260810','--run-name',probe_name,
    '--architecture','equivariant_set_branching','--projection-budget','16','--reward-scale-config',str(scale),
    '--environment-profile',str(profile),'--normalize-input-blocks','--learning-rate','1e-5',
    '--trajectory-loss-weight','1.0','--concavity-loss-weight','0.1','--learn-every','4','--precision','fp32','--device','cuda'],cwd=repo,check=True)
preflight=DRIVE_ROOT/'STEP3_BOUNDED_PREFLIGHT.json'
subprocess.run([sys.executable,'-B','validation/step3_v3_preflight.py','--mechanism-report',str(mechanism),
    '--checkpoint',str(repo/'outputs/phase2'/probe_name/'branching_c51.pt'),'--environment-profile',str(profile),
    '--ch-risk-config',str(risk),'--step3-qos-config',str(qos),'--output',str(preflight),
    '--runtime-contract-output',str(runtime_path)],cwd=repo,check=True)
"""),
        cell("code", """# Train once for 100 episodes. Stability checkpoints are copied to Drive immediately.
name='step3_v3_bounded_100ep_seed5599'; local=repo/'outputs/phase2'/name; saved=DRIVE_ROOT/'phase2'/name
saved.mkdir(parents=True,exist_ok=True)
checkpoint=saved/'stability_episode_100.pt'; episodes_file=saved/'episodes.jsonl'
if not (checkpoint.is_file() and episodes_file.is_file()):
    if local.exists(): shutil.rmtree(local)
    cmd=[sys.executable,'-B','experiments/train_step3_v3_complete.py','--ch-risk-config',str(risk),
        '--step3-qos-config',str(qos),'--runtime-contract',str(runtime_path),'--preflight-report',str(preflight),
        '--checkpoint-export-dir',str(saved),'--episodes',str(EPISODES),'--max-steps',str(HORIZON),
        '--development-seeds','2400','--optimizer-seed',str(OPTIMIZER_SEED),'--run-name',name,
        '--architecture','equivariant_set_branching','--projection-budget','16','--reward-scale-config',str(scale),
        '--environment-profile',str(profile),'--normalize-input-blocks','--learning-rate','1e-5',
        '--trajectory-loss-weight','1.0','--concavity-loss-weight','0.1','--learn-every','4','--precision','fp32',
        '--stability-interval','25','--stability-tail-episodes','100','--device','cuda']
    log=saved/f'{name}.log'
    with log.open('a') as handle:
        process=subprocess.Popen(cmd,cwd=repo,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        for line in process.stdout: print(line,end=''); handle.write(line); handle.flush()
        code=process.wait()
    if code not in (0,3): raise RuntimeError(f'Training execution failed: exit {code}')
assert checkpoint.is_file() and episodes_file.is_file(), 'Episode-100 checkpoint was not persisted.'
checkpoint_gate=DRIVE_ROOT/'STEP3_BOUNDED_CHECKPOINT_GATE.json'
gate=subprocess.run([sys.executable,'-B','validation/analyze_step3_bounded_training_checkpoint.py',
    '--checkpoint',str(checkpoint),'--episodes-jsonl',str(episodes_file),'--output',str(checkpoint_gate)],cwd=repo)
if gate.returncode: raise RuntimeError('Bounded checkpoint failed QoS/risk gate. Stop; do not train longer.')
"""),
        cell("code", """# Five paired 1,200-round development evaluations against all baselines.
eval_name='step3_v3_bounded_seed5599_dev1200'; eval_local=repo/'outputs/phase3'/eval_name
run=subprocess.run([sys.executable,'-B','experiments/run_phase3_step3_v3.py','--ch-risk-config',str(risk),
    '--seeds',','.join(map(str,EVALUATION_SEEDS)),'--horizon',str(HORIZON),'--run-name',eval_name,
    '--skip-compatibility','--hta-checkpoint',str(checkpoint),'--hta-budget','16','--environment-profile',str(profile)],cwd=repo)
if run.returncode: raise RuntimeError('Paired development evaluation failed.')
saved_eval=DRIVE_ROOT/'phase3'/eval_name
if saved_eval.exists(): shutil.rmtree(saved_eval)
shutil.copytree(eval_local,saved_eval)
decision=DRIVE_ROOT/'STEP3_BOUNDED_FINAL_DECISION.json'
gate=subprocess.run([sys.executable,'-B','validation/evaluate_step3_bounded_probe.py',
    '--phase3-summary',str(eval_local/'summary.json'),'--output',str(decision)],cwd=repo)
print(json.dumps(json.loads(decision.read_text()),indent=2))
if gate.returncode: print('STOP: the bounded probe did not justify full training. This is the final planned outcome.')
else: print('SIGNAL DETECTED: design a separate frozen full-training contract; do not extend this run automatically.')
"""),
        cell("code", """# Archive the bounded evidence regardless of pass/fail; no full training follows.
archive_base=Path('/content/HTA_MAC_Step3_Bounded_Probe_Results_20260810')
archive=Path(shutil.make_archive(str(archive_base),'zip',root_dir=DRIVE_ROOT))
sha=hashlib.sha256(archive.read_bytes()).hexdigest()
sidecar=archive.with_suffix('.zip.sha256'); sidecar.write_text(f'{sha}  {archive.name}\\n')
shutil.copy2(archive,DRIVE_ROOT/archive.name); shutil.copy2(sidecar,DRIVE_ROOT/sidecar.name)
print('RESULTS',archive,'SHA256',sha)
if DOWNLOAD_RESULTS_WHEN_COMPLETE:
    from google.colab import files
    files.download(str(archive)); files.download(str(sidecar))
"""),
    ]
    notebook={"cells":cells,"metadata":{"accelerator":"GPU","kernelspec":{"display_name":"Python 3","language":"python","name":"python3"}},"nbformat":4,"nbformat_minor":5}
    for index,item in enumerate(cells):
        if item["cell_type"]=="code": compile("".join(item["source"]),f"cell_{index}","exec")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(notebook,indent=1),encoding="utf-8")


if __name__=="__main__":
    main()
