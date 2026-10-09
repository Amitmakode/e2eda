"""
Main pipeline runner — runs the full data pipeline in order:
1. Test Snowflake connection
2. Run EDA (saves charts to outputs/)
3. Train ML model (saves model to models/)

Usage:
    python main.py
"""

import subprocess
import sys

STEPS = [
    ("Connecting to Snowflake & verifying data", "src/connect_snowflake.py"),
    ("Running EDA", "src/eda.py"),
    ("Training ML model", "src/train_model.py"),
]

def run_step(title, script_path):
    print("\n" + "=" * 60)
    print(f"STEP: {title}")
    print("=" * 60)
    result = subprocess.run([sys.executable, script_path])
    if result.returncode != 0:
        print(f"\n❌ Failed at: {title} (exit code {result.returncode})")
        sys.exit(1)
    print(f"✅ Done: {title}")

if __name__ == "__main__":
    for title, script in STEPS:
        run_step(title, script)
    print("\n🎉 Pipeline complete — model trained, charts saved, ready for app.py")