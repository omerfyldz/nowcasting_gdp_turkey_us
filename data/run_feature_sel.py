"""Wrapper: runs feature_selection_ensemble.py as subprocess and logs output."""
import subprocess, sys, os, time

script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "feature_selection_ensemble.py")
log = os.path.join(os.path.dirname(os.path.abspath(__file__)), "feature_sel_run.log")

print(f"Starting: {script}", flush=True)
print(f"Log: {log}", flush=True)

with open(log, "w", buffering=1) as f:
    proc = subprocess.Popen(
        [sys.executable, "-u", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
        cwd=os.path.dirname(script),
    )
    for line in proc.stdout:
        f.write(line)
        f.flush()
        print(line, end="", flush=True)
    proc.wait()

print(f"\nExit code: {proc.returncode}", flush=True)
