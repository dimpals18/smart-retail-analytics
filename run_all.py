"""Run the whole pipeline in order.  Usage:  python run_all.py [--dashboard]"""
import subprocess
import sys

STEPS = [
    "create_data.py",
    "src/data_collection.py",
    "src/data_cleaning.py",
    "src/database.py",
    "src/analysis.py",
    "src/forecasting.py",
]

for number, script in enumerate(STEPS):
    print(f"\n=== Step {number}: {script} ===")
    if subprocess.run([sys.executable, script]).returncode != 0:
        sys.exit(f"\nPipeline stopped: {script} failed.")

print("\nPipeline finished successfully.")
if "--dashboard" in sys.argv:
    subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard.py"])
else:
    print("Start the dashboard with:  python -m streamlit run dashboard.py")
