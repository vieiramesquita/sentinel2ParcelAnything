#!/usr/bin/env python3
"""
Launcher for Interactive Parcel Segmentation Web Application with HQ-SAM
LAPIG - Image Processing and Geoprocessing Laboratory (UFG)

Usage:
  python run_interactive_app.py [--port 8088] [--host 0.0.0.0]
"""

import os
import sys
import argparse
import uvicorn

# Add interactive directory and project root to sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def main():
    parser = argparse.ArgumentParser(description="LAPIG HQ-SAM Interactive Parcel Segmenter")
    parser.add_argument("--host", default="0.0.0.0", help="Host IP to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8088, help="TCP Port (default: 8088)")
    parser.add_argument("--reload", action="store_true", help="Enable code auto-reload")
    args = parser.parse_args()

    print("=" * 70)
    print("LAPIG HQ-SAM - Interactive Prompt-Based Parcel Segmenter")
    print("=" * 70)
    print(f"Server starting at:  http://127.0.0.1:{args.port}")
    print(f"Local network access: http://localhost:{args.port}")
    print("=" * 70)

    # Launch FastAPI server pointing to backend app
    os.chdir(PROJECT_ROOT)  # Allows root access to results/ and local GeoPackages
    uvicorn.run("interactive_hq_sam.backend.app:app", host=args.host, port=args.port, reload=args.reload)

if __name__ == "__main__":
    main()
