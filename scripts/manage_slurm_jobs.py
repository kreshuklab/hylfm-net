import argparse
import subprocess
import sys
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

    subprocess.check_output(f"{sys.executable} -m lnet {task}", stderr=subprocess.STDOUT, shell=True)
