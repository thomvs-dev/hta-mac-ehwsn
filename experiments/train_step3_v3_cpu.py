"""CPU-threaded launcher for the bounded Step 3 v3 development probe."""

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
        raise ValueError("CPU thread count exceeds available logical processors")
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(threads)

    import numpy as np
    import torch
    import experiments.train_step3_v3_complete as complete

    def threaded_set_seeds(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.set_num_threads(threads)
        try:
            torch.set_num_interop_threads(max(1, min(4, threads // 4)))
        except RuntimeError:
            pass
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True)
        print(
            f"CPU_THREAD_CONTRACT intraop={torch.get_num_threads()} "
            f"interop={torch.get_num_interop_threads()} logical={os.cpu_count()}",
            flush=True,
        )

    complete.v3.trainer.set_seeds = threaded_set_seeds
    if "--device" not in remaining:
        remaining.extend(["--device", "cpu"])
    elif remaining[remaining.index("--device") + 1] != "cpu":
        raise ValueError("CPU launcher requires --device cpu")
    sys.argv = [sys.argv[0], *remaining]
    return complete.main()


if __name__ == "__main__":
    raise SystemExit(main())
