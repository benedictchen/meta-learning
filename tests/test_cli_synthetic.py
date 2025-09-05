import json, subprocess, sys, os, pathlib

def test_cli_eval_runs():
    # Run the CLI in-process via python -m to ensure entrypoint works
    cmd = [sys.executable, "-m", "meta_learning.cli", "eval", "--episodes", "50"]
    out = subprocess.check_output(cmd, text=True)
    data = json.loads(out)
    assert "mean_acc" in data and data["episodes"]==50
