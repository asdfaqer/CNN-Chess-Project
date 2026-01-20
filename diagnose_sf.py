import subprocess
import os
from utils import STOCKFISH_PATH

test_file = "sf_test_result.txt"
with open(test_file, "w") as f:
    f.write(f"Testing path: {STOCKFISH_PATH}\n")
    if not os.path.exists(STOCKFISH_PATH):
        f.write("Path does NOT exist on disk.\n")
        # Try to list the desktop to help the user
        desktop = os.path.join(os.path.expanduser("~"), "Desktop")
        f.write(f"Desktop path: {desktop}\n")
        if os.path.exists(desktop):
            f.write(f"Desktop contents: {os.listdir(desktop)}\n")
    else:
        f.write("Path exists on disk.\n")
        try:
            # Try to run it
            res = subprocess.run([STOCKFISH_PATH, "--version"], capture_output=True, text=True, timeout=5)
            f.write(f"Return code: {res.returncode}\n")
            f.write(f"Stdout: {res.stdout}\n")
            f.write(f"Stderr: {res.stderr}\n")
        except Exception as e:
            f.write(f"Failed to run: {e}\n")
