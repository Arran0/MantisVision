"""Puts ml/ on sys.path so tests can `import config` and `from src...` the
same way every script under ml/ already does (see e.g. src/train.py's own
sys.path.insert)."""
import sys
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parents[1]
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))
