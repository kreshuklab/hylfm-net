import argparse
import subprocess
import sys
import os
import warnings
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task_nr", type=int)
    parser.add_argument("tasks_path", type=Path)

    args = parser.parse_args()

    tasks = [task.strip() for task in args.tasks_path.read_text().split("\n") if task]
    task_nr = args.task_nr
    if task_nr >= len(tasks):
        warnings.warn(f"task_nr {task_nr} out if range")

    task = tasks[task_nr]

    subprocess.call(f"run_lnet.sh {task}")
#     try:
#         ret = subprocess.run(f"{sys.executable} -m lnet {task}".split(), shell=True, check=True, stderr=subprocess.STDOUT)
#     except subprocess.CalledProcessError as e:
#         print(e.output.decode())
#         raise e
#     else:
#         print(ret.stdout.decode())
