import os
from utils import STOCKFISH_PATH

with open("path_check.txt", "w") as f:
    f.write(f"STOCKFISH_PATH: {STOCKFISH_PATH}\n")
    f.write(f"Exists: {os.path.exists(STOCKFISH_PATH)}\n")
    if os.path.dirname(STOCKFISH_PATH):
        dir_path = os.path.dirname(STOCKFISH_PATH)
        f.write(f"Directory {dir_path} exists: {os.path.exists(dir_path)}\n")
        if os.path.exists(dir_path):
            f.write(f"Contents of {dir_path}: {os.listdir(dir_path)}\n")
