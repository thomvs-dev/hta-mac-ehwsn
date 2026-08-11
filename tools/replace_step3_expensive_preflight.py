"""Replace the obsolete 100-episode learning preflight with cheap audits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook")
    args = parser.parse_args()
    path = Path(args.notebook)
    notebook = json.loads(path.read_text(encoding="utf-8"))
    replacement = """# Cheap preflight: deterministic mechanism reachability plus one 5-step architecture checkpoint.
mechanism_report = DRIVE_ROOT / 'STEP3_MECHANISM_PROBE.json'
subprocess.run([
    sys.executable, '-B', 'experiments/probe_step3_mechanism.py',
    '--environment-profile', str(profile), '--ch-risk-config', str(risk),
    '--seeds', ','.join(map(str, DEVELOPMENT_SEEDS)), '--max-steps', str(EVALUATION_HORIZON),
    '--output', str(mechanism_report),
], cwd=repo, check=True)
mechanism = json.loads(mechanism_report.read_text())
assert mechanism['status'] == 'step3_mechanism_probe_pass'
print('Mechanism probe passed without learning:', json.dumps(mechanism['gates'], indent=2))

probe_name = 'step3_architecture_probe_seed20260809'
probe_dir = repo / 'outputs/phase2' / probe_name
if probe_dir.exists(): shutil.rmtree(probe_dir)
subprocess.run([
    sys.executable, '-B', 'experiments/train_step3_probe.py',
    '--ch-risk-config', str(risk), '--runtime-contract', str(runtime_path),
    '--episodes', '1', '--max-steps', '5',
    '--development-seeds', ','.join(map(str, DEVELOPMENT_SEEDS)), '--optimizer-seed', '20260809',
    '--run-name', probe_name, '--architecture', 'equivariant_set_branching', '--projection-budget', '16',
    '--reward-scale-config', str(scale), '--qos-constraint-config', str(qos), '--environment-profile', str(profile),
    '--normalize-input-blocks', '--learning-rate', '1e-5', '--trajectory-loss-weight', '1.0',
    '--concavity-loss-weight', '0.1', '--learn-every', '4', '--precision', 'fp32', '--device', 'cuda',
], cwd=repo, check=True)

preflight = DRIVE_ROOT / 'STEP3_PREFLIGHT.json'
preflight_run = subprocess.run([
    sys.executable, '-B', 'validation/step3_pretraining_preflight_v2.py',
    '--mechanism-report', str(mechanism_report), '--checkpoint', str(probe_dir / 'branching_c51.pt'),
    '--environment-profile', str(profile), '--ch-risk-config', str(risk), '--output', str(preflight),
    '--runtime-contract-output', str(runtime_path),
], cwd=repo)
shutil.copytree(probe_dir, DRIVE_ROOT / 'preflight_architecture_probe', dirs_exist_ok=True)
if preflight_run.returncode:
    raise RuntimeError('STEP 3 STOP: cheap preflight failed; evidence is saved in Drive.')
report = json.loads(preflight.read_text())
assert report['overall_pass'] and report['runtime_fingerprint_sha256'] == runtime['fingerprint_sha256']
print(json.dumps(report, indent=2))
"""
    matches = 0
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        if "train_step3_activation_probe.py" in source:
            cell["source"] = replacement.splitlines(True)
            matches += 1
    if matches != 1:
        raise RuntimeError(f"expected one obsolete activation-probe cell, found {matches}")
    path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
