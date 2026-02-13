#!/usr/bin/env python3
"""Debug script to examine W&B run structure."""

import wandb
from pathlib import Path

ENTITY = "cesar-valdez-mcgill-university"
PROJECT = "geomancer-phate-labeled"

def debug_runs():
    """Examine the structure of W&B runs."""
    print(f"🔗 Connecting to W&B project: {ENTITY}/{PROJECT}")

    try:
        api = wandb.Api()
        runs = list(api.runs(f"{ENTITY}/{PROJECT}", per_page=5))  # Just get first 5
        print(f"📊 Found {len(runs)} runs for debugging")
    except Exception as e:
        print(f"❌ Error connecting to W&B: {e}")
        return

    for i, run in enumerate(runs):
        print(f"\n--- Run {i}: {run.id} ---")
        print(f"Name: {run.name}")
        print(f"State: {run.state}")
        print(f"Created: {run.created_at}")
        print(f"URL: {run.url}")

        print(f"Config type: {type(run.config)}")
        print(f"Config value: {run.config}")

        print(f"Summary type: {type(run.summary)}")
        print(f"Summary value: {run.summary}")

        # Try to access files
        try:
            files = list(run.files())
            print(f"Files: {len(files)} files")
            for f in files[:3]:  # Just first 3
                print(f"  - {f.name}")
        except Exception as e:
            print(f"Files error: {e}")

        if i >= 2:  # Only debug first 3 runs
            break

if __name__ == "__main__":
    debug_runs()