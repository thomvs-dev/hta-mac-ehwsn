"""Finalize the currently running replication using its frozen reporting tools."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/paid_loss_replication_20260912'
deadline=time.monotonic()+10800
while not (OUT/'results.json').exists():
    if (OUT/'STOP_EVIDENCE.json').exists():
        print('STOP: evaluator wrote integration-failure evidence.',flush=True)
        sys.exit(2)
    if time.monotonic()>deadline:
        print('STOP: finalizer timeout; inspect evaluator, retain all files.',flush=True)
        sys.exit(3)
    time.sleep(5)
# results.json is created exclusively and then written. Wait for the complete
# JSON document instead of racing the evaluator's final file write.
while True:
    try:
        json.loads((OUT/'results.json').read_text())
        break
    except json.JSONDecodeError:
        if time.monotonic()>deadline:raise
        time.sleep(1)
try:
    for path in ['tools/report_paid_loss_replication_20260912.py','tools/verify_paid_loss_replication_20260912.py','tools/paid_loss_replication_sensitivity_20260912.py']:
        subprocess.run([sys.executable,'-B',path],cwd=ROOT,check=True)
    screen=json.loads((OUT/'screen.json').read_text())
    passed=all(screen['checks'].values())
    def sha(path):
        with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
    paths=[p for p in OUT.iterdir() if p.is_file()]+[ROOT/'reports/PAID_LOSS_REPLICATION_RESULTS_20260912.md',ROOT/'reports/PAID_LOSS_REPLICATION_METHOD_20260912.md',Path(__file__)]
    record=dict(status='paid_loss_replication_pass_no_training_authorization' if passed else 'STOP_paid_loss_replication_failed_or_inconclusive',checks=screen['checks'],effects=screen['effects'],trials=screen['trials'],frames=screen['frames'],historical_gates_unchanged=True,neural_training_started=False,paid_control_evaluated=True,sha256={str(p.relative_to(ROOT)):sha(p) for p in paths})
    with (OUT/'PIPELINE_COMPLETE.json').open('x') as f:json.dump(record,f,indent=2)
    print(record['status'],flush=True)
    sys.exit(0 if passed else 2)
except Exception as exc:
    with (OUT/'STOP_FINALIZATION.json').open('x') as f:json.dump(dict(status='finalization_failure',error=repr(exc),no_promotion=True),f,indent=2)
    raise
