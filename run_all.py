"""Run the whole pipeline in order.

    python run_all.py            # every step
    python run_all.py 02 03      # only the named steps

Step 0 is the slow one (it walks ~3.25 M raw events); the rest read its
output. Steps 6 and 7 are optional diagnostics and are skipped automatically
if shap or torch is missing.
"""
import runpy
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = ["00_extract.py", "01_benchmarks.py", "02_corners.py", "03_throwins.py",
         "04_matches.py", "05_external.py", "06_predictive.py",
         "07_sequence_model.py", "08_figures.py", "09_audit.py"]


def main(argv):
    wanted = argv or [s[:2] for s in STEPS]
    selected = [s for s in STEPS if s[:2] in wanted]
    if not selected:
        raise SystemExit(f"no steps matched {wanted}; available: {[s[:2] for s in STEPS]}")
    for step in selected:
        print(f"\n=== {step} ===", flush=True)
        started = time.time()
        runpy.run_path(str(ROOT / step), run_name="__main__")
        print(f"--- {step} finished in {time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
