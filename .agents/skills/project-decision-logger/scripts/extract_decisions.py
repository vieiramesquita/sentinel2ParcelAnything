#!/usr/bin/env python3
"""
extract_decisions.py
Helper script to parse conversation logs and assist in synchronizing PROJECT_DECISIONS.md.
Reads transcript.jsonl from Antigravity IDE brain logs if available, or analyzes recent conversation history.
"""

import os
import sys
import json
import glob
from pathlib import Path
from datetime import datetime

def find_latest_transcript(app_data_dir=None):
    if not app_data_dir:
        home = Path.home()
        app_data_dir = home / ".gemini" / "antigravity-ide" / "brain"
    else:
        app_data_dir = Path(app_data_dir)

    if not app_data_dir.exists():
        return None

    # Search for transcript.jsonl
    transcripts = list(app_data_dir.glob("*/.system_generated/logs/transcript.jsonl"))
    if not transcripts:
        return None
    # Pick the most recently modified transcript
    transcripts.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return transcripts[0]

def extract_user_requests(transcript_path, max_entries=50):
    user_requests = []
    if not transcript_path or not os.path.exists(transcript_path):
        return user_requests

    with open(transcript_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if data.get("type") == "USER_INPUT":
                    content = data.get("content", "").strip()
                    if content and "<USER_REQUEST>" in content:
                        # Extract between <USER_REQUEST> tags
                        start = content.find("<USER_REQUEST>") + len("<USER_REQUEST>")
                        end = content.find("</USER_REQUEST>")
                        if end > start:
                            req = content[start:end].strip()
                        else:
                            req = content[start:].strip()
                        if req:
                            user_requests.append({
                                "step": data.get("step_index"),
                                "request": req,
                                "timestamp": data.get("timestamp")
                            })
                    elif content:
                        user_requests.append({
                            "step": data.get("step_index"),
                            "request": content[:300],
                            "timestamp": data.get("timestamp")
                        })
            except Exception:
                continue

    return user_requests[-max_entries:]

def main():
    transcript = find_latest_transcript()
    print(f"Latest transcript: {transcript}")
    if transcript:
        requests = extract_user_requests(transcript)
        print(f"\nExtracted {len(requests)} recent user interactions:")
        for r in requests[-20:]:
            step = r.get('step', '?')
            req = r.get('request', '').replace('\n', ' ')
            print(f"[{step}] {req[:120]}")

if __name__ == "__main__":
    main()
