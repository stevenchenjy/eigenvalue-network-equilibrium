"""Run the current local source without depending on an editable installation."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from network_models.experiments import main


if __name__ == "__main__":
    raise SystemExit(main(command=[sys.executable, *sys.argv]))
