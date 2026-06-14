import os
import time
import subprocess
import sys


BASE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(BASE, "src", "models")  # correct path


def run_step(name, script):
    script_path = os.path.join(SRC, script)

    print("\n" + "=" * 70)
    print(f"{name}")
    print("=" * 70)

    start = time.time()

    # IMPORTANT FIX: use SAME python as current environment
    result = subprocess.run([sys.executable, script_path])

    if result.returncode != 0:
        print(f"ERROR in {name}")
        exit(1)

    print(f"✔ Done in {round(time.time() - start, 2)} sec")


def main():
    print("\n=== FRAUD DETECTION TASK 2 ===\n")

    run_step("Training", "train_models.py")
    run_step("Cross Validation", "cross_validate.py")
    run_step("Evaluation", "evaluate_models.py")

    print("\nPIPELINE COMPLETE")


if __name__ == "__main__":
    main()