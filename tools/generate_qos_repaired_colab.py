"""Generate the locked post-repair paper-aligned Colab notebook."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(True)}


def markdown(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cells = [
        markdown("""# HTA-MAC paper-aligned B16 QoS-repaired training (2026-08-08)

This notebook executes the post-repair development sequence: verify, test, recalibrate, train fresh seeds **5399/6399/7399**, run 300-round global evaluation, select using separately scoped global gates, audit policy-action distinctness on common states, and run a 3,000-round censor-aware development evaluation.

This is a secondary literature-alignment side study; it does not replace the primary idle-on hybrid solar+thermal track and cannot establish C1 or C3. The training QoS ratio is target-backlog service, not end-to-end delivery. Whole-network delivery, stale drops, fairness, allocation pressure, and lifetime are evaluated separately. Registered seeds 3100-3104 are prohibited; confirmation seeds 3400-3404 remain unused. Expected NVIDIA L4 time is approximately 4-9 hours, dominated by three 500-episode lineages and the 3,000-round paired evaluation.
"""),
        code(f"""# Frozen user settings. Change only paths/download preference.
BUNDLE_PATH = '/content/HTA_MAC_PaperAligned_B16_QoSRepaired_Training_Bundle_20260808.zip'
DRIVE_OUTPUT_DIR = '/content/drive/MyDrive/HTA_MAC_PaperAligned_B16_QoSRepaired_20260808'
SEEDS = [5399, 6399, 7399]
EPISODES = 500
DOWNLOAD_RESULTS_WHEN_COMPLETE = True
EXPECTED_BUNDLE_SHA256 = '{args.bundle_sha256}'
assert SEEDS == [5399, 6399, 7399]
assert EPISODES == 500
"""),
        code("""# GPU, Drive, bundle discovery, checksum, safe extraction, and manifest verification.
import csv, glob, hashlib, json, os, shutil, subprocess, sys, zipfile
from pathlib import Path, PurePosixPath
import numpy as np
import torch
if not torch.cuda.is_available():
    raise RuntimeError('Select Runtime > Change runtime type > GPU before training.')
print('GPU:', torch.cuda.get_device_name(0), '| Torch:', torch.__version__)
try:
    from google.colab import drive
    drive.mount('/content/drive')
except ImportError:
    print('Not running in Colab; Drive mount skipped.')
bundle = Path(BUNDLE_PATH)
if not bundle.is_file():
    candidates = [Path(p) for p in glob.glob('/content/*PaperAligned*B16*QoSRepaired*Bundle*.zip')]
    if len(candidates) != 1:
        raise FileNotFoundError(f'Expected one uploaded repaired bundle, found {candidates}')
    bundle = candidates[0]
digest = hashlib.sha256(bundle.read_bytes()).hexdigest()
assert digest == EXPECTED_BUNDLE_SHA256, (digest, EXPECTED_BUNDLE_SHA256)
WORK = Path('/content/hta_mac_paper_aligned_b16_qos_repaired')
if WORK.exists():
    shutil.rmtree(WORK)
WORK.mkdir(parents=True)
with zipfile.ZipFile(bundle) as archive:
    for member in archive.infolist():
        normalized = member.filename.replace('\\\\', '/')
        path = PurePosixPath(normalized)
        if path.is_absolute() or '..' in path.parts:
            raise RuntimeError(f'Unsafe ZIP member: {member.filename}')
        target = WORK.joinpath(*path.parts)
        if member.is_dir() or normalized.endswith('/'):
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(member, 'r') as source, target.open('wb') as destination:
            shutil.copyfileobj(source, destination)
stage2 = WORK / 'stage2'
repo = stage2 / 'hta-mac'
upstream = stage2 / 'final_repo'
manifest_path = stage2 / 'COLAB_PAPER_ALIGNED_B16_QOS_REPAIRED_MANIFEST.json'
manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
DEVELOPMENT_SEEDS = [2400, 2401, 2402, 2403, 2404]
TRAINING_HORIZON = 300
LONG_HORIZON = 3000
DRIVE_ROOT = Path(DRIVE_OUTPUT_DIR)
DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
assert manifest['optimizer_seeds'] == SEEDS
assert manifest['development_seeds'] == DEVELOPMENT_SEEDS
assert manifest['episodes'] == EPISODES
assert manifest['training_horizon'] == TRAINING_HORIZON and manifest['long_horizon'] == LONG_HORIZON
assert set(DEVELOPMENT_SEEDS).isdisjoint(manifest['reserved_confirmation_seeds'])
assert set(DEVELOPMENT_SEEDS).isdisjoint(manifest['prohibited_registered_held_out_seeds'])
assert manifest['architecture'] == 'equivariant_set_branching'
assert manifest['track_role'] == 'secondary_literature_alignment_side_study'
assert manifest['primary_track_replaced'] is False
assert manifest['primary_contributions_evaluated_by_this_profile'] is False
assert manifest['phase3_reports_budget_utilization'] is True
assert manifest['phase3_reports_feasible_demand_contention'] is True
for entry in manifest['files']:
    target = stage2 / entry['path']
    assert target.is_file() and target.stat().st_size == entry['bytes'], entry['path']
    assert hashlib.sha256(target.read_bytes()).hexdigest() == entry['sha256'], entry['path']
shutil.copy2(manifest_path, DRIVE_ROOT / manifest_path.name)
preflight_source = repo / 'outputs/preflight_20260808'
preflight_drive = DRIVE_ROOT / 'preflight_20260808'
if preflight_drive.exists(): shutil.rmtree(preflight_drive)
shutil.copytree(preflight_source, preflight_drive)
print('Verified bundle:', bundle.name, '| files:', len(manifest['files']), '| SHA256:', digest)
"""),
        code("""# Install dependencies, compile, and run the complete validation suite.
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'gymnasium', 'torch-geometric', 'scipy', 'pyyaml', 'pytest'], check=True)
subprocess.run([sys.executable, '-B', '-m', 'compileall', '-q', str(repo), str(upstream)], check=True)
test_result = subprocess.run([sys.executable, '-B', '-m', 'pytest', 'validation', '-q', '-p', 'no:cacheprovider'], cwd=repo, text=True, capture_output=True)
print(test_result.stdout)
if test_result.stderr:
    print(test_result.stderr, file=sys.stderr)
if test_result.returncode:
    raise RuntimeError(f'Validation failed with exit code {test_result.returncode}; see output above.')
print('Validation suite passed.')
profile_contract = json.loads((repo / 'config/paper_aligned_hasani2025_b16_qos_repaired.json').read_text())
architecture_decision = json.loads((repo / 'config/paper_aligned_hasani2025_architecture_decision_repaired.json').read_text())
preflight_foundation = json.loads((preflight_source / 'paper_aligned_b16_current_code_preflight_foundation_seed5299_20260808.json').read_text())
assert profile_contract['track_role'] == 'secondary_literature_alignment_side_study'
assert profile_contract['primary_track_replaced'] is False
assert architecture_decision['selected_architecture_key'] == 'equivariant_set_branching'
assert architecture_decision['rejected_python_class'].endswith('GlobalBranchingDuelingC51')
assert architecture_decision['mechanism_losses']['trajectory_loss_weight'] == 1.0
assert architecture_decision['mechanism_losses']['concavity_loss_weight'] == 0.1
assert architecture_decision['observation_contract']['rank_or_percentile_features_present'] is False
assert preflight_foundation['status'] == 'gate_pass'
"""),
        code("""# Development-only return-scale recalibration under the repaired QoS contract.
profile = repo / 'config/paper_aligned_hasani2025_b16_qos_repaired.json'
qos = repo / 'config/paper_aligned_hasani2025_qos_constraints_repaired.json'
global_gates_path = repo / 'config/paper_aligned_hasani2025_global_evaluation_gates_repaired.json'
scale_drive = DRIVE_ROOT / 'paper_aligned_b16_qos_repaired_return_scale.generated.json'
scale_local = repo / 'config/paper_aligned_b16_qos_repaired_return_scale.generated.json'
subprocess.run([
    sys.executable, '-B', 'experiments/calibrate_paper_aligned_return_scale.py',
    '--environment-profile', str(profile), '--qos-constraint-config', str(qos),
    '--development-seeds', ','.join(map(str, DEVELOPMENT_SEEDS)),
    '--max-steps', str(TRAINING_HORIZON), '--rollouts', '100', '--output', str(scale_drive),
], cwd=repo, check=True)
shutil.copy2(scale_drive, scale_local)
scale = json.loads(scale_local.read_text())
assert scale['status'] == 'frozen_development_scale' and not scale['held_out_seeds_used']
assert scale['development_seeds'] == DEVELOPMENT_SEEDS
print(json.dumps(scale, indent=2))
"""),
        code("""# Train three fresh lineages, run structural audits, and checkpoint each to Drive.
def sync_tree(source, destination):
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)

lineages = []
for seed in SEEDS:
    run_name = f'paper_aligned_b16_qos_repaired_500ep_seed{seed}'
    local_run = repo / 'outputs/phase2' / run_name
    drive_run = DRIVE_ROOT / 'phase2' / run_name
    reusable = (drive_run / 'summary.json').exists() and (drive_run / 'foundation_audit.json').exists()
    if reusable:
        old = json.loads((drive_run / 'summary.json').read_text())
        audit_old = json.loads((drive_run / 'foundation_audit.json').read_text())
        reusable = old.get('optimizer_seed') == seed and old.get('episodes_completed') == EPISODES and old.get('phase2_curriculum_gate_pass') is True and audit_old.get('status') == 'gate_pass'
    if reusable:
        print('Restoring verified completed lineage', seed)
        sync_tree(drive_run, local_run)
    else:
        if local_run.exists():
            shutil.rmtree(local_run)
        command = [
            sys.executable, '-B', 'experiments/train_phase2_dynamic_curriculum.py',
            '--episodes', str(EPISODES), '--max-steps', str(TRAINING_HORIZON),
            '--development-seeds', ','.join(map(str, DEVELOPMENT_SEEDS)),
            '--optimizer-seed', str(seed), '--run-name', run_name,
            '--architecture', 'equivariant_set_branching', '--projection-budget', '16',
            '--reward-scale-config', str(scale_local), '--qos-constraint-config', str(qos),
            '--environment-profile', str(profile), '--normalize-input-blocks',
            '--learning-rate', '1e-5', '--trajectory-loss-weight', '1.0',
            '--concavity-loss-weight', '0.1', '--learn-every', '4',
            '--precision', 'fp32', '--stability-interval', '50', '--stability-tail-episodes', '100',
            '--device', 'cuda',
        ]
        log_dir = DRIVE_ROOT / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f'{run_name}.log'
        print('Starting lineage', seed, '| log:', log_path, flush=True)
        with log_path.open('w', encoding='utf-8') as log_handle:
            process = subprocess.Popen(command, cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in process.stdout:
                print(line, end='', flush=True)
                log_handle.write(line)
                log_handle.flush()
            returncode = process.wait()
        if not local_run.exists():
            raise RuntimeError(f'No output produced for seed {seed}; exit={returncode}; inspect {log_path}')
        sync_tree(local_run, drive_run)
        if returncode:
            raise RuntimeError(f'Training gate failed for seed {seed}; evidence saved at {drive_run}')
        checkpoint = local_run / 'branching_c51.pt'
        audit_local = local_run / 'foundation_audit.json'
        audit_result = subprocess.run([
            sys.executable, '-B', 'experiments/audit_phase2d_foundation.py', str(checkpoint),
            '--output', str(audit_local), '--max-steps', str(TRAINING_HORIZON),
            '--development-seeds', ','.join(map(str, DEVELOPMENT_SEEDS)),
            '--environment-profile', str(profile), '--random-permutations', '20', '--targeted-swaps', '10',
        ], cwd=repo)
        sync_tree(local_run, drive_run)
        if audit_result.returncode:
            raise RuntimeError(f'Foundation audit failed for seed {seed}; evidence saved at {drive_run}')
    summary = json.loads((local_run / 'summary.json').read_text())
    audit = json.loads((local_run / 'foundation_audit.json').read_text())
    assert summary['optimizer_seed'] == seed and summary['phase2_curriculum_gate_pass']
    assert audit['status'] == 'gate_pass'
    lineages.append({'seed': seed, 'run_name': run_name, 'checkpoint': str(local_run / 'branching_c51.pt')})
print('Structurally accepted lineages:', [item['seed'] for item in lineages])
"""),
        code("""# 300-round whole-network development evaluation for every accepted lineage.
for lineage in lineages:
    eval_name = lineage['run_name'] + '_dev300'
    local_eval = repo / 'outputs/phase3' / eval_name
    drive_eval = DRIVE_ROOT / 'phase3' / eval_name
    reusable = (drive_eval / 'summary.json').exists()
    if reusable:
        prior = json.loads((drive_eval / 'summary.json').read_text())
        reusable = prior.get('phase3_structural_gate_pass') is True and prior.get('seeds') == DEVELOPMENT_SEEDS and prior.get('horizon') == TRAINING_HORIZON
    if reusable:
        sync_tree(drive_eval, local_eval)
    else:
        if local_eval.exists():
            shutil.rmtree(local_eval)
        result = subprocess.run([
            sys.executable, '-B', 'experiments/run_phase3_pilot.py',
            '--seeds', ','.join(map(str, DEVELOPMENT_SEEDS)), '--horizon', str(TRAINING_HORIZON),
            '--run-name', eval_name, '--skip-compatibility', '--hta-checkpoint', lineage['checkpoint'],
            '--hta-budget', '16', '--environment-profile', str(profile),
        ], cwd=repo)
        if not local_eval.exists():
            raise RuntimeError(f'No evaluation output for seed {lineage["seed"]}; exit={result.returncode}')
        sync_tree(local_eval, drive_eval)
        if result.returncode:
            raise RuntimeError(f'Network evaluation failed for seed {lineage["seed"]}; evidence saved at {drive_eval}')
    lineage['evaluation'] = str(local_eval)
print('Completed 300-round evaluation for all lineages.')
"""),
        code("""# Select using separately scoped global end-to-end development gates.
gates = json.loads(global_gates_path.read_text())
constraints = {
    'global_network_end_to_end_delivery_ratio_min': gates['minimum_delivery_ratio'],
    'global_network_stale_drop_ratio_max': gates['maximum_stale_drop_ratio'],
    'global_network_service_fairness_min': gates['minimum_global_network_service_fairness'],
}
candidates = []
for lineage in lineages:
    with (Path(lineage['evaluation']) / 'raw_trials.csv').open(newline='', encoding='utf-8') as handle:
        rows = [row for row in csv.DictReader(handle) if row['policy'] == 'hta_mac']
    assert len(rows) == len(DEVELOPMENT_SEEDS)
    trials = []
    for row in rows:
        delivery, stale, fairness = float(row['delivery_ratio']), float(row['stale_drop_ratio']), float(row['queue_fairness'])
        violation = max(0, constraints['global_network_end_to_end_delivery_ratio_min'] - delivery) + max(0, stale - constraints['global_network_stale_drop_ratio_max']) + max(0, constraints['global_network_service_fairness_min'] - fairness)
        trials.append({'seed': int(row['seed']), 'pass': violation == 0.0, 'violation': violation, 'throughput': float(row['throughput']), 'energy_efficiency_packets_per_j': float(row['energy_efficiency_packets_per_j'])})
    candidates.append({
        'optimizer_seed': lineage['seed'], 'run_name': lineage['run_name'],
        'joint_global_qos_pass_count': sum(item['pass'] for item in trials),
        'median_constraint_violation': float(np.median([item['violation'] for item in trials])),
        'median_throughput': float(np.median([item['throughput'] for item in trials])),
        'median_energy_efficiency_packets_per_j': float(np.median([item['energy_efficiency_packets_per_j'] for item in trials])),
        'trials': trials,
    })
candidates.sort(key=lambda item: (-item['joint_global_qos_pass_count'], item['median_constraint_violation'], -item['median_throughput'], -item['median_energy_efficiency_packets_per_j'], item['optimizer_seed']))
best = candidates[0]
selected = best['optimizer_seed'] if best['joint_global_qos_pass_count'] == len(DEVELOPMENT_SEEDS) else None
selection = {
    'status': 'development_candidate_selected' if selected is not None else 'no_candidate_global_qos_feasible',
    'claim_boundary': 'development_selection_only_not_confirmation_or_third_party_reproduction',
    'selected_optimizer_seed': selected, 'thresholds': constraints,
    'ranking_rule': ['joint_global_qos_pass_count_desc', 'median_constraint_violation_asc', 'median_throughput_desc', 'median_energy_efficiency_desc', 'optimizer_seed_asc'],
    'candidates': candidates, 'registered_held_out_seeds_used': False,
}
(DRIVE_ROOT / 'DEVELOPMENT_SELECTION.json').write_text(json.dumps(selection, indent=2))
print(json.dumps(selection, indent=2))
"""),
        code("""# Common-state policy-action distinctness audit for the selected candidate.
if selection['selected_optimizer_seed'] is None:
    print('SKIPPED action audit: no candidate passed all five global development trials.')
else:
    chosen = next(item for item in lineages if item['seed'] == selection['selected_optimizer_seed'])
    action_audit = DRIVE_ROOT / 'ACTION_DISTINCTNESS_AUDIT.json'
    subprocess.run([
        sys.executable, '-B', 'experiments/audit_policy_action_distinctness.py',
        '--checkpoint', chosen['checkpoint'], '--environment-profile', str(profile),
        '--seeds', ','.join(map(str, DEVELOPMENT_SEEDS)), '--horizon', str(TRAINING_HORIZON),
        '--hta-budget', '16', '--output', str(action_audit),
    ], cwd=repo, check=True)
    assert action_audit.exists()
    print(json.dumps(json.loads(action_audit.read_text())['aggregate'], indent=2))
"""),
        code("""# 3,000-round paired development evaluation with censor-aware FND/HND reporting.
if selection['selected_optimizer_seed'] is None:
    print('SKIPPED long-horizon evaluation: no candidate passed the global gate.')
else:
    chosen = next(item for item in lineages if item['seed'] == selection['selected_optimizer_seed'])
    long_name = chosen['run_name'] + '_dev3000_lifetime'
    local_long = repo / 'outputs/phase3' / long_name
    drive_long = DRIVE_ROOT / 'phase3' / long_name
    reusable = (drive_long / 'summary.json').exists()
    if reusable:
        prior = json.loads((drive_long / 'summary.json').read_text())
        reusable = prior.get('phase3_structural_gate_pass') is True and prior.get('seeds') == DEVELOPMENT_SEEDS and prior.get('horizon') == LONG_HORIZON
    if reusable:
        sync_tree(drive_long, local_long)
    else:
        if local_long.exists():
            shutil.rmtree(local_long)
        result = subprocess.run([
            sys.executable, '-B', 'experiments/run_phase3_pilot.py',
            '--seeds', ','.join(map(str, DEVELOPMENT_SEEDS)), '--horizon', str(LONG_HORIZON),
            '--run-name', long_name, '--skip-compatibility', '--hta-checkpoint', chosen['checkpoint'],
            '--hta-budget', '16', '--environment-profile', str(profile),
        ], cwd=repo)
        if not local_long.exists():
            raise RuntimeError(f'No long-horizon output; exit={result.returncode}')
        sync_tree(local_long, drive_long)
        if result.returncode:
            raise RuntimeError(f'Long-horizon evaluation failed; evidence saved at {drive_long}')
    long_summary = json.loads((local_long / 'summary.json').read_text())
    assert long_summary['phase3_structural_gate_pass'] and long_summary['horizon'] == LONG_HORIZON
    print(json.dumps(long_summary['censor_aware_lifetime'], indent=2))
"""),
        code("""# Package all evidence; confirmation seeds remain untouched.
archive_base = Path('/content/HTA_MAC_PaperAligned_B16_QoSRepaired_Trained_Results_20260808')
final_archive = DRIVE_ROOT / (archive_base.name + '.zip')
sidecar = DRIVE_ROOT / (archive_base.name + '.zip.sha256')
for artifact in (final_archive, sidecar):
    if artifact.exists():
        artifact.unlink()
archive = Path(shutil.make_archive(str(archive_base), 'zip', root_dir=DRIVE_ROOT))
shutil.copy2(archive, final_archive)
result_digest = hashlib.sha256(final_archive.read_bytes()).hexdigest()
sidecar.write_text(f'{result_digest}  {final_archive.name}\\n')
print('RESULTS ZIP:', final_archive)
print('SHA256:', result_digest)
print('SELECTION STATUS:', selection['status'])
print('Confirmation seeds 3400-3404 and registered seeds 3100-3104 were not used.')
if DOWNLOAD_RESULTS_WHEN_COMPLETE:
    try:
        from google.colab import files
        files.download(str(final_archive))
    except ImportError:
        print('Not running in Colab; automatic download skipped.')
"""),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"name": "HTA_MAC_PaperAligned_B16_QoSRepaired_Training_Colab_20260808.ipynb", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.x"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
