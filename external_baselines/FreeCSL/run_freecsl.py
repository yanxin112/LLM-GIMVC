import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from official_wrapper import run_official_wrapper


if __name__ == "__main__":
    run_official_wrapper("freecsl", "FreeCSL")
