"""Generate the locked Step 3 v3 screening, training, and evaluation notebook."""

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
        cell("markdown", """# HTA-MAC Step 3 v3 — QoS-feasible CH-aware lifetime training

This notebook uses only development seeds 2400–2404 for screening and evaluation. It never opens confirmation seeds. The CH schedule is frozen; the learned intervention is MAC allocation only. A failed gate stops the run, and no infeasible candidate is promoted.

Drive authentication and bundle verification happen before any training. Episode-500 weights and stability snapshots are copied atomically to Drive before final evaluation, so finalization can recover without retraining.
"""),
        cell("code", f"""# Frozen settings: change paths/download preference only.
BUNDLE_PATH = '/content/HTA_MAC_Step3_V3_Training_Bundle_20260809.zip'
DRIVE_OUTPUT_DIR = '/content/drive/MyDrive/HTA_MAC_Step3_V3_20260809'
FRESH_SEEDS = [5599, 6599, 7599]
DEVELOPMENT_SEEDS = [2400, 2401, 2402, 2403, 2404]
EPISODES = 500
TRAINING_HORIZON = 1200
EVALUATION_HORIZON = 3000
EXPECTED_BUNDLE_SHA256 = '{args.bundle_sha256}'
DOWNLOAD_RESULTS_WHEN_COMPLETE = True
assert FRESH_SEEDS == [5599, 6599, 7599] and EPISODES == 500
assert not set(DEVELOPMENT_SEEDS) & set(range(3100, 3105))
assert not set(DEVELOPMENT_SEEDS) & set(range(3400, 3405))
"""),
        cell("code", """# GPU, Drive auth, checksum, collision-safe extraction, and manifest verification.
import glob, hashlib, json, os, shutil, subprocess, sys, zipfile
from pathlib import Path, PurePosixPath
import torch
if not torch.cuda.is_available(): raise RuntimeError('Select Runtime > Change runtime type > GPU.')
from google.colab import drive
if not Path('/content/drive/MyDrive').is_dir():
    try: drive.mount('/content/drive')
    except Exception as exc: raise RuntimeError('Drive authentication failed before training; reconnect and rerun this cell.') from exc
bundle = Path(BUNDLE_PATH)
if not bundle.is_file():
    candidates = [Path(p) for p in glob.glob('/content/*Step3*V3*Bundle*.zip')]
    if len(candidates) != 1: raise FileNotFoundError(f'Expected one uploaded v3 bundle, found {candidates}')
    bundle = candidates[0]
digest = hashlib.sha256(bundle.read_bytes()).hexdigest()
assert digest == EXPECTED_BUNDLE_SHA256, (digest, EXPECTED_BUNDLE_SHA256)
WORK = Path('/content/hta_mac_step3_v3')
if WORK.exists(): shutil.rmtree(WORK)
WORK.mkdir()
with zipfile.ZipFile(bundle) as archive:
    normalized = [(m, m.filename.replace('\\\\', '/').rstrip('/')) for m in archive.infolist()]
    names = {name for _, name in normalized if name}
    directory_names = {str(PurePosixPath(name).parent) for name in names}
    directory_names |= {str(parent) for name in names for parent in PurePosixPath(name).parents if str(parent) != '.'}
    for member, name in normalized:
        path = PurePosixPath(name)
        if not name or path.is_absolute() or '..' in path.parts: raise RuntimeError(f'Unsafe ZIP member: {member.filename}')
        target = WORK.joinpath(*path.parts)
        is_directory = member.is_dir() or name in directory_names
        if is_directory:
            if target.exists() and not target.is_dir(): raise RuntimeError(f'ZIP file/directory collision: {name}')
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.is_dir(): raise RuntimeError(f'ZIP directory/file collision: {name}')
            with archive.open(member) as src, target.open('wb') as dst: shutil.copyfileobj(src, dst)
stage2 = WORK / 'stage2'; repo = stage2 / 'hta-mac'; upstream = stage2 / 'final_repo'
manifest_path = stage2 / 'COLAB_STEP3_V3_MANIFEST.json'
manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
assert manifest['optimizer_seeds'] == FRESH_SEEDS and manifest['episodes'] == EPISODES
assert manifest['observation_schema'] == 'step3_ch_context_v3'
assert manifest['ch_schedule_modified'] is False and manifest['exact_same_runtime_required'] is True
for entry in manifest['files']:
    target = stage2 / entry['path']
    assert target.is_file() and target.stat().st_size == entry['bytes'], entry['path']
    assert hashlib.sha256(target.read_bytes()).hexdigest() == entry['sha256'], entry['path']
DRIVE_ROOT = Path(DRIVE_OUTPUT_DIR); DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
shutil.copy2(manifest_path, DRIVE_ROOT / manifest_path.name)
print('Verified', bundle.name, digest, '| GPU', torch.cuda.get_device_name(0), '| Torch', torch.__version__)
"""),
        cell("code", """# Tests and exact runtime contract. No GPU training occurs before this passes.
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'gymnasium', 'torch-geometric', 'scipy', 'pyyaml', 'pytest'], check=True)
subprocess.run([sys.executable, '-B', '-m', 'compileall', '-q', str(repo), str(upstream)], check=True)
test = subprocess.run([sys.executable, '-B', '-m', 'pytest', 'validation', '-q', '-p', 'no:cacheprovider'], cwd=repo, text=True, capture_output=True)
print(test.stdout); print(test.stderr, file=sys.stderr)
if test.returncode: raise RuntimeError('Validation failed; training is forbidden.')
sys.path.insert(0, str(repo))
from core.runtime_fingerprint import make_runtime_contract
runtime = make_runtime_contract(require_cuda=True)
runtime_path = DRIVE_ROOT / 'STEP3_V3_RUNTIME_CONTRACT.json'
runtime_path.write_text(json.dumps(runtime, indent=2) + '\\n')
print('Frozen exact runtime', runtime['fingerprint_sha256'])
"""),
        cell("code", """# Helpers: every candidate gets its own scale and same-runtime preflight.
profile = repo / 'config/paper_aligned_hasani2025_b16_qos_repaired.json'
qos_candidates = {
  'episode': repo / 'config/step3_v3_qos_episode_end_candidate.json',
  'ema': repo / 'config/step3_v3_qos_ema_candidate.json',
  'ema_floor': repo / 'config/step3_v3_qos_ema_floor_candidate.json',
}
risk_candidates = {w: repo / f'config/step3_v3_risk_weight_{w}.json' for w in (1,5,10,20)}
screen_root = DRIVE_ROOT / 'screening'; screen_root.mkdir(exist_ok=True)
def sync(source, destination):
    if destination.exists(): shutil.rmtree(destination)
    shutil.copytree(source, destination)
def prepare(candidate_id, risk, qos, horizon):
    root = screen_root / candidate_id; root.mkdir(parents=True, exist_ok=True)
    scale = root / 'return_scale.json'
    subprocess.run([sys.executable, '-B', 'experiments/calibrate_step3_v3_return_scale.py',
        '--ch-risk-config', str(risk), '--environment-profile', str(profile), '--qos-constraint-config', str(qos),
        '--development-seeds', ','.join(map(str, DEVELOPMENT_SEEDS)), '--max-steps', str(horizon),
        '--rollouts', '20', '--output', str(scale)], cwd=repo, check=True)
    mechanism = root / 'mechanism.json'
    subprocess.run([sys.executable, '-B', 'experiments/probe_step3_mechanism.py',
        '--environment-profile', str(profile), '--ch-risk-config', str(risk),
        '--seeds', ','.join(map(str, DEVELOPMENT_SEEDS)), '--max-steps', str(TRAINING_HORIZON),
        '--output', str(mechanism)], cwd=repo, check=True)
    probe_name = 'v3_probe_' + candidate_id
    subprocess.run([sys.executable, '-B', 'experiments/train_step3_v3_probe.py',
        '--ch-risk-config', str(risk), '--step3-qos-config', str(qos), '--runtime-contract', str(runtime_path),
        '--episodes', '1', '--max-steps', '5', '--development-seeds', '2400', '--optimizer-seed', '20260809',
        '--run-name', probe_name, '--architecture', 'equivariant_set_branching', '--projection-budget', '16',
        '--reward-scale-config', str(scale), '--environment-profile', str(profile), '--normalize-input-blocks',
        '--learning-rate', '1e-5', '--trajectory-loss-weight', '1.0', '--concavity-loss-weight', '0.1',
        '--learn-every', '4', '--precision', 'fp32', '--device', 'cuda'], cwd=repo, check=True)
    preflight = root / 'preflight.json'
    subprocess.run([sys.executable, '-B', 'validation/step3_v3_preflight.py',
        '--mechanism-report', str(mechanism), '--checkpoint', str(repo/'outputs/phase2'/probe_name/'branching_c51.pt'),
        '--environment-profile', str(profile), '--ch-risk-config', str(risk), '--step3-qos-config', str(qos),
        '--output', str(preflight), '--runtime-contract-output', str(runtime_path)], cwd=repo, check=True)
    return scale, preflight
def train_screen(candidate_id, risk, qos, episodes, horizon=TRAINING_HORIZON):
    scale, preflight = prepare(candidate_id, risk, qos, horizon)
    name = f'step3_v3_screen_{candidate_id}_{episodes}ep'; local = repo/'outputs/phase2'/name; saved = screen_root/name
    cmd = [sys.executable, '-B', 'experiments/train_step3_v3.py', '--ch-risk-config', str(risk),
        '--step3-qos-config', str(qos), '--runtime-contract', str(runtime_path), '--preflight-report', str(preflight),
        '--checkpoint-export-dir', str(saved), '--episodes', str(episodes), '--max-steps', str(horizon),
        '--development-seeds', ','.join(map(str, DEVELOPMENT_SEEDS)), '--optimizer-seed', '20260809', '--run-name', name,
        '--architecture', 'equivariant_set_branching', '--projection-budget', '16', '--reward-scale-config', str(scale),
        '--environment-profile', str(profile), '--normalize-input-blocks', '--learning-rate', '1e-5',
        '--trajectory-loss-weight', '1.0', '--concavity-loss-weight', '0.1', '--learn-every', '4', '--precision', 'fp32',
        '--stability-interval', str(max(1, episodes//4)), '--stability-tail-episodes', str(max(1, episodes//2)), '--device', 'cuda']
    run = subprocess.run(cmd, cwd=repo)
    if run.returncode not in (0,3): raise RuntimeError(f'Screening execution failed: {candidate_id}, exit={run.returncode}')
    if not (local/'summary.json').is_file(): raise RuntimeError(f'Missing screen summary: {candidate_id}')
    sync(local, saved)
    return local/'summary.json'
def select(rows, stage, maximum):
    output = screen_root / f'selection_{stage}ep.json'
    run = subprocess.run([sys.executable, '-B', 'experiments/select_step3_v3_candidate.py',
        *map(str, rows), '--maximum-survivors', str(maximum), '--stage', str(stage), '--output', str(output)], cwd=repo)
    payload = json.loads(output.read_text())
    if run.returncode: raise RuntimeError(payload['status'])
    return payload
"""),
        cell("code", """# Successive halving: QoS-controller screen, then risk weights 1/5/10/20.
qos_rows = [train_screen(f'qos_{key}', risk_candidates[1], path, 25) for key,path in qos_candidates.items()]
qos_selection = select(qos_rows, 25, 1)
selected_qos_key = next(key for key in qos_candidates if key in Path(qos_selection['survivors'][0]['path']).parent.name)
selected_qos = qos_candidates[selected_qos_key]
risk25 = {w: train_screen(f'risk_{w}', path, selected_qos, 25) for w,path in risk_candidates.items()}
sel25 = select(list(risk25.values()), 25, 2)
surviving_weights = [w for w in risk_candidates if any(f'risk_{w}_' in Path(row['path']).parent.name for row in sel25['survivors'])]
risk100 = {w: train_screen(f'risk_{w}_stage100', risk_candidates[w], selected_qos, 100) for w in surviving_weights}
sel100 = select(list(risk100.values()), 100, min(2, len(risk100)))
surviving_weights = [w for w in surviving_weights if any(f'risk_{w}_stage100' in Path(row['path']).parent.name for row in sel100['survivors'])]
risk250 = {w: train_screen(f'risk_{w}_stage250', risk_candidates[w], selected_qos, 250) for w in surviving_weights[:2]}
sel250 = select(list(risk250.values()), 250, 1)
selected_weight = next(w for w in risk250 if f'risk_{w}_stage250' in Path(sel250['survivors'][0]['path']).parent.name)
selected_risk = risk_candidates[selected_weight]
(DRIVE_ROOT/'FROZEN_STEP3_V3_SELECTION.json').write_text(json.dumps({
    'qos_candidate': selected_qos_key, 'risk_weight': selected_weight,
    'qos_sha256': hashlib.sha256(selected_qos.read_bytes()).hexdigest(),
    'risk_sha256': hashlib.sha256(selected_risk.read_bytes()).hexdigest(),
    'confirmation_seeds_opened': False}, indent=2)+'\\n')
print('Frozen development configuration:', selected_qos_key, 'risk weight', selected_weight)
"""),
        cell("code", """# Fresh 500-episode lineages with immediate Drive checkpoint export and finalization recovery.
full_scale, full_preflight = prepare(f'full_{selected_qos_key}_risk{selected_weight}', selected_risk, selected_qos, TRAINING_HORIZON)
lineages = []
for seed in FRESH_SEEDS:
    name = f'step3_v3_500ep_seed{seed}'; local = repo/'outputs/phase2'/name; saved = DRIVE_ROOT/'phase2'/name
    summary_path = saved/'summary.json'
    reusable = summary_path.is_file() and json.loads(summary_path.read_text()).get('phase2_curriculum_gate_pass') is True
    if not reusable and (saved/'training_complete_weights.pt').is_file() and (saved/'episodes.jsonl').is_file():
        if local.exists(): shutil.rmtree(local)
        sync(saved, local)
        required = [local/f'stability_episode_{ep}.pt' for ep in (400,450,500)]
        if all(path.is_file() for path in required):
            run = subprocess.run([sys.executable, '-B', 'experiments/recover_step3_v3_finalization.py',
                '--run-name', name, '--optimizer-seed', str(seed), '--training-git-hash', 'recovered_same_checkpoint',
                '--ch-risk-config', str(selected_risk), '--step3-qos-config', str(selected_qos),
                '--runtime-contract', str(runtime_path), '--preflight-report', str(full_preflight),
                '--environment-profile', str(profile), '--checkpoint-export-dir', str(saved),
                '--development-seeds', ','.join(map(str, DEVELOPMENT_SEEDS)), '--episodes', str(EPISODES),
                '--max-steps', str(TRAINING_HORIZON), '--device', 'cuda'], cwd=repo)
            reusable = run.returncode == 0
    if not reusable:
        if local.exists(): shutil.rmtree(local)
        saved.mkdir(parents=True, exist_ok=True)
        log = saved/f'{name}.log'
        cmd = [sys.executable, '-B', 'experiments/train_step3_v3_failure_safe.py',
            '--ch-risk-config', str(selected_risk), '--step3-qos-config', str(selected_qos),
            '--runtime-contract', str(runtime_path), '--preflight-report', str(full_preflight), '--checkpoint-export-dir', str(saved),
            '--episodes', str(EPISODES), '--max-steps', str(TRAINING_HORIZON), '--development-seeds', ','.join(map(str, DEVELOPMENT_SEEDS)),
            '--optimizer-seed', str(seed), '--run-name', name, '--architecture', 'equivariant_set_branching',
            '--projection-budget', '16', '--reward-scale-config', str(full_scale), '--environment-profile', str(profile),
            '--normalize-input-blocks', '--learning-rate', '1e-5', '--trajectory-loss-weight', '1.0',
            '--concavity-loss-weight', '0.1', '--learn-every', '4', '--precision', 'fp32', '--stability-interval', '50',
            '--stability-tail-episodes', '100', '--device', 'cuda']
        with log.open('a') as handle:
            process = subprocess.Popen(cmd, cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in process.stdout: print(line, end=''); handle.write(line); handle.flush()
            code = process.wait()
        if local.exists(): sync(local, saved)
        if code: raise RuntimeError(f'STOP: lineage {seed} failed gate/finalization; see {log}')
    summary = json.loads((saved/'summary.json').read_text())
    if summary.get('phase2_curriculum_gate_pass') is not True: raise RuntimeError(f'STOP: lineage {seed} is infeasible')
    lineages.append((seed, saved/'branching_c51.pt'))
print('Accepted fresh lineages', [seed for seed,_ in lineages])
"""),
        cell("code", """# Paired 3,000-round development evaluation plus frozen seed-7399 reference.
reference_checkpoint = stage2/'reference_seed7399'/'branching_c51.pt'
reference_name = 'step2_seed7399_reference_dev3000'
reference_local = repo/'outputs/phase3'/reference_name
subprocess.run([sys.executable, '-B', 'experiments/run_phase3_pilot.py', '--seeds', ','.join(map(str, DEVELOPMENT_SEEDS)),
    '--horizon', str(EVALUATION_HORIZON), '--run-name', reference_name, '--skip-compatibility',
    '--hta-checkpoint', str(reference_checkpoint), '--hta-budget', '16', '--environment-profile', str(profile)], cwd=repo, check=True)
sync(reference_local, DRIVE_ROOT/'phase3'/reference_name)
development_reports = []
for seed, checkpoint in lineages:
    name = f'step3_v3_seed{seed}_dev3000'; local = repo/'outputs/phase3'/name
    subprocess.run([sys.executable, '-B', 'experiments/run_phase3_step3_v3.py', '--ch-risk-config', str(selected_risk),
        '--seeds', ','.join(map(str, DEVELOPMENT_SEEDS)), '--horizon', str(EVALUATION_HORIZON), '--run-name', name,
        '--skip-compatibility', '--hta-checkpoint', str(checkpoint), '--hta-budget', '16',
        '--environment-profile', str(profile)], cwd=repo, check=True)
    decision = DRIVE_ROOT/f'DEVELOPMENT_DECISION_SEED{seed}.json'
    run = subprocess.run([sys.executable, '-B', 'validation/select_step3_v3_development.py',
        '--candidate-summary', str(local/'summary.json'), '--reference-summary', str(reference_local/'summary.json'),
        '--output', str(decision)], cwd=repo)
    sync(local, DRIVE_ROOT/'phase3'/name)
    development_reports.append((seed, run.returncode, decision))
eligible = [seed for seed,code,_ in development_reports if code == 0]
if not eligible: raise RuntimeError('no_candidate_global_qos_feasible; confirmation remains closed')
print('Development-eligible lineages:', eligible, '| confirmation seeds remain unopened')
"""),
        cell("code", """# Archive all evidence. Results are development evidence, not a publication claim.
archive_base = Path('/content/HTA_MAC_Step3_V3_Trained_Results_20260809')
archive = Path(shutil.make_archive(str(archive_base), 'zip', root_dir=DRIVE_ROOT))
sha = hashlib.sha256(archive.read_bytes()).hexdigest()
sidecar = archive.with_suffix('.zip.sha256'); sidecar.write_text(f'{sha}  {archive.name}\\n')
shutil.copy2(archive, DRIVE_ROOT/archive.name); shutil.copy2(sidecar, DRIVE_ROOT/sidecar.name)
print('RESULTS', archive, '| SHA256', sha)
if DOWNLOAD_RESULTS_WHEN_COMPLETE:
    from google.colab import files
    files.download(str(archive)); files.download(str(sidecar))
"""),
    ]
    notebook = {
        "cells": cells,
        "metadata": {"accelerator": "GPU", "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(notebook, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
