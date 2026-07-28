import os
import time
import subprocess
import sys

BASE = os.path.dirname(os.path.abspath(__file__))


def run_step(name, module_path):
    print("\n" + "=" * 70)
    print(f"{name}")
    print("=" * 70)

    start = time.time()

    # Run as `python -m <module_path>` from the project root so that the
    # `from src....` imports inside each script resolve correctly.
    # Invoking the .py file path directly would only put that file's own
    # folder on sys.path, not the project root, and raise
    # ModuleNotFoundError: src.
    result = subprocess.run(
        [sys.executable, "-m", module_path],
        cwd=BASE,
    )

    if result.returncode != 0:
        print(f"ERROR in {name}")
        exit(1)

    print(f"✔ Done in {round(time.time() - start, 2)} sec")


def main():
    print("\n=== FRAUD DETECTION PIPELINE ===\n")

    run_step("Data Pipeline", "scripts.pipeline")
    run_step("Training", "src.models.train_models")
    run_step("Cross Validation", "src.models.cross_validate")
    run_step("Evaluation", "src.models.evaluate_models")

    print("\nPIPELINE COMPLETE")


if __name__ == "__main__":
    main()
