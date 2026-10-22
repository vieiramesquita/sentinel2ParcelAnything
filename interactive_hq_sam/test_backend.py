import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.chdir(PROJECT_ROOT)
print("Testing import...")
from interactive_hq_sam.backend.app import app, health_check, get_presets, engine
print("Health check:", health_check())
presets = get_presets()
print(f"Presets loaded: {len(presets)}")
if presets:
    print(f"Sample preset: {presets[0]['title']} ({presets[0]['id']})")
print("ALL TESTS PASSED!")
