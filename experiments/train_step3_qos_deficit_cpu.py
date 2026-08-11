"""High-utilization CPU launcher for the bounded QoS-deficit probe."""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--cpu-threads", type=int, required=True)
    known, remaining = parser.parse_known_args()
    threads = int(known.cpu_threads)
    if not 1 <= threads <= (os.cpu_count() or 1):
        raise ValueError("invalid CPU thread count")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(threads)
    import numpy as np
    import torch
    import experiments.train_step3_qos_deficit_complete as controlled

    def threaded_set_seeds(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.set_num_threads(threads)
        try:
            torch.set_num_interop_threads(max(1, min(4, threads // 4)))
        except RuntimeError:
            pass
        torch.use_deterministic_algorithms(True)
        print(
            f"CPU_THREAD_CONTRACT intraop={torch.get_num_threads()} "
            f"interop={torch.get_num_interop_threads()} logical={os.cpu_count()}",
            flush=True,
        )

    controlled.complete.v3.trainer.set_seeds = threaded_set_seeds
    if "--device" not in remaining:
        remaining.extend(["--device", "cpu"])
    elif remaining[remaining.index("--device") + 1] != "cpu":
        raise ValueError("CPU launcher requires --device cpu")
    sys.argv = [sys.argv[0], *remaining]
    return controlled.main()


if __name__ == "__main__":
    raise SystemExit(main())
