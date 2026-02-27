"""
RUN_ME.py - Start the Resume Screening System
Copy & run this script to start the application
"""

import subprocess
import webbrowser
import time
import os
import sys

print("=" * 70)
print("RESUME SCREENING SYSTEM - AUTO LAUNCHER")
print("=" * 70)

# Change to project directory
project_dir = r"c:\Users\hp\Desktop\NLPpro\resume_screening_project"
os.chdir(project_dir)

print(f"\nProject Directory: {project_dir}")
print(f"Current Directory: {os.getcwd()}")

# Start Streamlit app
print("\n[1/3] Starting Streamlit application...")
print("      This may take 10-15 seconds as BERT model loads...\n")

streamlit_process = subprocess.Popen(
    [sys.executable, "-m", "streamlit", "run", "app.py"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)

print("      Waiting for app to initialize...")
time.sleep(5)

print("\n[2/3] Opening browser...")
time.sleep(2)

try:
    webbrowser.open("http://localhost:8501")
    print("      Browser opened: http://localhost:8501")
except Exception as e:
    print(f"      Could not auto-open browser: {e}")
    print("      Manual URL: http://localhost:8501")

print("\n[3/3] Application ready!")
print("\nApplication is running in the background.")
print("Visit: http://localhost:8501\n")

print("=" * 70)
print("INSTRUCTIONS:")
print("=" * 70)
print("1. Your browser should open automatically")
print("2. If not, go to: http://localhost:8501")
print("3. Upload PDF resumes")
print("4. Paste job description")
print("5. Click 'Analyze Resumes'")
print("6. View rankings and export results")
print("\nPress Ctrl+C to stop the server")
print("=" * 70)

# Keep the script running
try:
    streamlit_process.wait()
except KeyboardInterrupt:
    print("\n\nShutting down...")
    streamlit_process.terminate()
    streamlit_process.wait()
    print("Application stopped.")
