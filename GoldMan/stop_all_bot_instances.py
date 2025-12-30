"""
Script to stop all Python processes that might be running the bot
WARNING: This will stop ALL Python processes. Use with caution.
"""
import subprocess
import sys
import os

def stop_python_processes():
    try:
        result = subprocess.run(
            ['tasklist', '/FI', 'IMAGENAME eq python.exe', '/FO', 'CSV'],
            capture_output=True,
            text=True
        )
        
        lines = result.stdout.strip().split('\n')
        if len(lines) <= 1:
            print("No Python processes found.")
            return
        
        print("Found Python processes:")
        pids = []
        for line in lines[1:]:  # Skip header
            if line.strip():
                parts = line.split(',')
                if len(parts) >= 2:
                    pid = parts[1].strip('"')
                    try:
                        pids.append(int(pid))
                        print(f"  PID: {pid}")
                    except:
                        pass
        
        if not pids:
            print("No valid PIDs found.")
            return
        
        response = input(f"\nDo you want to stop {len(pids)} Python process(es)? (yes/no): ")
        if response.lower() != 'yes':
            print("Cancelled.")
            return
        
        for pid in pids:
            try:
                subprocess.run(['taskkill', '/PID', str(pid), '/F'], check=False)
                print(f"Stopped process {pid}")
            except Exception as e:
                print(f"Error stopping process {pid}: {e}")
        
        print("\nDone. Please wait 10-20 seconds before starting the bot again.")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    stop_python_processes()

