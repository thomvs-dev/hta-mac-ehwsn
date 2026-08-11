"""Generate the locked Step 3 same-runtime Colab notebook."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def cell(kind, source):
    payload = {"cell_type": kind, "metadata": {}, "source": source.splitlines(True)}
    if kind == "code":
        payload.update(execution_count=None, outputs=[])
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cells = [
        cell("markdown", """# HTA-MAC Step 3: scheduled-CH depletion-risk lifetime experiment

Fresh development training only. The CH schedule is frozen; only MAC slot allocation is learned. Run all cells in one uninterrupted GPU runtime. The probe freezes the exact runtime fingerprint and full training refuses any mismatch. Cross-platform tolerance is not claimed or relaxed.

Stop after the preflight cell if any gate fails. No result is a publication claim; seeds 3100–3104 and 3400–3404 remain unused.
"""),
        cell("code", f"""BUNDLE_PATH = '/content/HTA_MAC_Step3_CHRole_Lifetime_Training_Bundle_20260808.zip'
DRIVE_OUTPUT_DIR = '/content/drive/MyDrive/HTA_MAC_Step3_CHRole_Lifetime_20260808'
SEEDS = [5499, 6499, 7499]
DEVELOPMENT_SEEDS = [2400, 2401, 2402, 2403, 2404]
EPISODES = 500
TRAINING_HORIZON = 1200
EVALUATION_HORIZON = 3000
EXPECTED_BUNDLE_SHA256 = '{args.bundle_sha256}'
DOWNLOAD_RESULTS_WHEN_COMPLETE = True
assert SEEDS == [5499, 6499, 7499] and EPISODES == 500
"""),
        cell("code", """import glob, hashlib, json, shutil, subprocess, sys, zipfile
from pathlib import Path, PurePosixPath
import torch
if not torch.cuda.is_available(): raise RuntimeError('Select a GPU runtime.')
from google.colab import drive
drive.mount('/content/drive')
bundle = Path(BUNDLE_PATH)
if not bundle.is_file():
    candidates = [Path(p) for p in glob.glob('/content/*Step3*CHRole*Bundle*.zip')]
    if len(candidates) != 1: raise FileNotFoundError(candidates)
    bundle = candidates[0]
digest = hashlib.sha256(bundle.read_bytes()).hexdigest()
assert digest == EXPECTED_BUNDLE_SHA256, (digest, EXPECTED_BUNDLE_SHA256)
WORK = Path('/content/hta_mac_step3_ch_role_lifetime')
if WORK.exists(): shutil.rmtree(WORK)
WORK.mkdir()
with zipfile.ZipFile(bundle) as archive:
    for member in archive.infolist():
        path = PurePosixPath(member.filename.replace('\\\\', '/'))
        if path.is_absolute() or '..' in path.parts: raise RuntimeError(member.filename)
        target = WORK.joinpath(*path.parts)
        if member.is_dir(): target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as src, target.open('wb') as dst: shutil.copyfileobj(src, dst)
stage2 = WORK / 'stage2'; repo = stage2 / 'hta-mac'; upstream = stage2 / 'final_repo'
manifest_path = stage2 / 'COLAB_STEP3_CH_ROLE_LIFETIME_MANIFEST.json'
manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
assert manifest['optimizer_seeds'] == SEEDS and manifest['episodes'] == EPISODES
assert manifest['training_horizon'] == TRAINING_HORIZON
assert manifest['ch_schedule_modified'] is False and manifest['exact_same_runtime_required'] is True
for entry in manifest['files']:
    target = stage2 / entry['path']
    assert target.is_file() and target.stat().st_size == entry['bytes']
    assert hashlib.sha256(target.read_bytes()).hexdigest() == entry['sha256']
DRIVE_ROOT = Path(DRIVE_OUTPUT_DIR); DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
shutil.copy2(manifest_path, DRIVE_ROOT / manifest_path.name)
print('Verified', bundle.name, digest, '| GPU', torch.cuda.get_device_name(0), '| Torch', torch.__version__)
"""),
        cell("code", """subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'gymnasium', 'torch-geometric', 'scipy', 'pyyaml', 'pytest'], check=True)
subprocess.run([sys.executable, '-B', '-m', 'compileall', '-q', str(repo), str(upstream)], check=True)
test = subprocess.run([sys.executable, '-B', '-m', 'pytest', 'validation', '-q', '-p', 'no:cacheprovider'], cwd=repo, text=True, capture_output=True)
print(test.stdout); print(test.stderr, file=sys.stderr)
if test.returncode: raise RuntimeError('Validation failed; full training is forbidden.')
sys.path.insert(0, str(repo))
from core.runtime_fingerprint import make_runtime_contract
runtime_path = DRIVE_ROOT / 'STEP3_RUNTIME_CONTRACT.json'
runtime = make_runtime_contract(require_cuda=True)
runtime_path.write_text(json.dumps(runtime, indent=2) + '\\n')
print('Frozen exact runtime:', runtime['fingerprint_sha256'])
"""),
        cell("code", """profile = repo / 'config/paper_aligned_hasani2025_b16_qos_repaired.json'
qos = repo / 'config/paper_aligned_hasani2025_qos_constraints_repaired.json'
risk = repo / 'config/step3_ch_role_depletion_risk_v1.json'
scale = DRIVE_ROOT / 'STEP3_RETURN_SCALE.json'
subprocess.run([
    sys.executable, '-B', 'experiments/calibrate_step3_return_scale.py',
    '--ch-risk-config', str(risk), '--environment-profile', str(profile),
    '--qos-constraint-config', str(qos), '--development-seeds', ','.join(map(str, DEVELOPMENT_SEEDS)),
    '--max-steps', str(TRAINING_HORIZON), '--rollouts', '100', '--output', str(scale),
], cwd=repo, check=True)
print(json.dumps(json.loads(scale.read_text()), indent=2))
"""),
        cell("code", """# Bounded five-episode probe. It is not an accepted model lineage.
probe_name = 'step3_preflight_probe_seed20260808'
probe_dir = repo / 'outputs/phase2' / probe_name
if probe_dir.exists(): shutil.rmtree(probe_dir)
probe_cmd = [
    sys.executable, '-B', 'experiments/train_step3_probe.py',
    '--ch-risk-config', str(risk), '--runtime-contract', str(runtime_path),
    '--episodes', '5', '--max-steps', str(TRAINING_HORIZON),
    '--development-seeds', ','.join(map(str, DEVELOPMENT_SEEDS)), '--optimizer-seed', '20260808',
    '--run-name', probe_name, '--architecture', 'equivariant_set_branching', '--projection-budget', '16',
    '--reward-scale-config', str(scale), '--qos-constraint-config', str(qos), '--environment-profile', str(profile),
    '--normalize-input-blocks', '--learning-rate', '1e-5', '--trajectory-loss-weight', '1.0',
    '--concavity-loss-weight', '0.1', '--learn-every', '4', '--precision', 'fp32', '--device', 'cuda',
]
subprocess.run(probe_cmd, cwd=repo, check=True)
preflight = DRIVE_ROOT / 'STEP3_PREFLIGHT.json'
preflight_run = subprocess.run([
    sys.executable, '-B', 'validation/step3_pretraining_preflight.py',
    '--episodes-jsonl', str(probe_dir / 'episodes.jsonl'), '--checkpoint', str(probe_dir / 'branching_c51.pt'),
    '--environment-profile', str(profile), '--ch-risk-config', str(risk), '--output', str(preflight),
    '--runtime-contract-output', str(runtime_path), '--minimum-training-horizon', '1000',
], cwd=repo)
shutil.copytree(probe_dir, DRIVE_ROOT / 'preflight_probe', dirs_exist_ok=True)
if preflight_run.returncode: raise RuntimeError('STEP 3 STOP: preflight failed; do not run the sweep.')
report = json.loads(preflight.read_text())
assert report['overall_pass'] and report['runtime_fingerprint_sha256'] == runtime['fingerprint_sha256']
print(json.dumps(report, indent=2))
"""),
        cell("code", """# Fresh three-lineage sweep. Each subprocess revalidates the exact probe runtime.
def sync(source, destination):
    if destination.exists(): shutil.rmtree(destination)
    shutil.copytree(source, destination)
lineages = []
for seed in SEEDS:
    name = f'step3_ch_role_lifetime_500ep_seed{seed}'
    local = repo / 'outputs/phase2' / name; saved = DRIVE_ROOT / 'phase2' / name
    reusable = (saved / 'summary.json').is_file()
    if reusable:
        prior = json.loads((saved / 'summary.json').read_text())
        reusable = prior.get('phase2_curriculum_gate_pass') is True and prior.get('optimizer_seed') == seed
    if reusable: sync(saved, local)
    else:
        if local.exists(): shutil.rmtree(local)
        cmd = [sys.executable, '-B', 'experiments/train_step3_lifetime.py',
            '--ch-risk-config', str(risk), '--runtime-contract', str(runtime_path), '--preflight-report', str(preflight),
            '--episodes', str(EPISODES), '--max-steps', str(TRAINING_HORIZON),
            '--development-seeds', ','.join(map(str, DEVELOPMENT_SEEDS)), '--optimizer-seed', str(seed),
            '--run-name', name, '--architecture', 'equivariant_set_branching', '--projection-budget', '16',
            '--reward-scale-config', str(scale), '--qos-constraint-config', str(qos), '--environment-profile', str(profile),
            '--normalize-input-blocks', '--learning-rate', '1e-5', '--trajectory-loss-weight', '1.0',
            '--concavity-loss-weight', '0.1', '--learn-every', '4', '--precision', 'fp32',
            '--stability-interval', '50', '--stability-tail-episodes', '100', '--device', 'cuda']
        log = DRIVE_ROOT / f'{name}.log'
        with log.open('w') as handle:
            process = subprocess.Popen(cmd, cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in process.stdout: print(line, end=''); handle.write(line); handle.flush()
            code = process.wait()
        if local.exists(): sync(local, saved)
        if code: raise RuntimeError(f'STOP: lineage {seed} failed; see {log}')
    summary = json.loads((local / 'summary.json').read_text())
    assert summary['phase2_curriculum_gate_pass'] and summary['step3_lifetime']['ch_schedule_modified'] is False
    lineages.append((seed, name, local / 'branching_c51.pt'))
print('Accepted structural lineages:', [seed for seed, _, _ in lineages])
"""),
        cell("code", """# Paired 3,000-round development evaluation of every accepted lineage; no automatic superiority claim.
for seed, name, checkpoint in lineages:
    eval_name = name + '_dev3000'
    local = repo / 'outputs/phase3' / eval_name; saved = DRIVE_ROOT / 'phase3' / eval_name
    if (saved / 'summary.json').is_file(): sync(saved, local); continue
    result = subprocess.run([sys.executable, '-B', 'experiments/run_phase3_pilot.py',
        '--seeds', ','.join(map(str, DEVELOPMENT_SEEDS)), '--horizon', str(EVALUATION_HORIZON),
        '--run-name', eval_name, '--skip-compatibility', '--hta-checkpoint', str(checkpoint),
        '--hta-budget', '16', '--environment-profile', str(profile)], cwd=repo)
    if local.exists(): sync(local, saved)
    if result.returncode: raise RuntimeError(f'Evaluation failed for {seed}')
print('Evaluations complete. Return the result ZIP for preregistered QoS/FND analysis.')
"""),
        cell("code", """archive_base = Path('/content/HTA_MAC_Step3_CHRole_Lifetime_Trained_Results_20260808')
archive = Path(shutil.make_archive(str(archive_base), 'zip', root_dir=DRIVE_ROOT))
sha = hashlib.sha256(archive.read_bytes()).hexdigest()
sidecar = archive.with_suffix('.zip.sha256'); sidecar.write_text(f'{sha}  {archive.name}\\n')
shutil.copy2(archive, DRIVE_ROOT / archive.name); shutil.copy2(sidecar, DRIVE_ROOT / sidecar.name)
print('RESULTS', archive, '| SHA256', sha)
if DOWNLOAD_RESULTS_WHEN_COMPLETE:
    from google.colab import files
    files.download(str(archive)); files.download(str(sidecar))
"""),
    ]
    notebook = {"cells": cells, "metadata": {"accelerator": "GPU", "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 5}
    Path(args.output).write_text(json.dumps(notebook, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
