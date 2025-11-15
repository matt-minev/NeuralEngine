#!/usr/bin/env python3
"""
Neural Engine - Launch All Applications
Starts all four Flask applications simultaneously
"""

import subprocess
import sys
import time
import signal
import os
from pathlib import Path

# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'

# Application configurations
APPS = [
    {
        'name': 'Landing Page',
        'path': 'apps/landing/app.py',
        'port': 8000,
        'color': Colors.CYAN,
        'description': 'Main entry point'
    },
    {
        'name': 'Digit Recognizer',
        'path': 'apps/digit_recognizer_web/app.py',
        'port': 8001,
        'color': Colors.BLUE,
        'description': 'Handwritten digit recognition'
    },
    {
        'name': 'Quadratic Neural Network',
        'path': 'apps/quadratic_web/app.py',
        'port': 8002,
        'color': Colors.GREEN,
        'description': 'Quadratic equation solver'
    },
    {
        'name': 'Universal Character Recognizer',
        'path': 'apps/universal_recognizer_web/app.py',
        'port': 8003,
        'color': Colors.YELLOW,
        'description': 'Universal character recognition'
    }
]

processes = []

def print_header():
    """Print startup header"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*70}")
    print("🧠 NEURAL ENGINE - LAUNCHING ALL APPLICATIONS")
    print(f"{'='*70}{Colors.END}\n")

def print_app_info(app):
    """Print application information"""
    print(f"{app['color']}▶ {app['name']}{Colors.END}")
    print(f"   Port: {Colors.BOLD}{app['port']}{Colors.END}")
    print(f"   URL: {Colors.BOLD}http://localhost:{app['port']}{Colors.END}")
    print(f"   {app['description']}\n")

def start_app(app):
    """Start a Flask application"""
    try:
        # Get the absolute path
        script_path = Path(__file__).parent / app['path']
        
        if not script_path.exists():
            print(f"{Colors.RED}✗ Error: {app['path']} not found{Colors.END}")
            return None
        
        print(f"{app['color']}Starting {app['name']}...{Colors.END}")
        print(f"{app['color']}  Script: {script_path}{Colors.END}")
        print(f"{app['color']}  Command: {sys.executable} {script_path}{Colors.END}")
        
        # Start the Flask app with unbuffered output
        process = subprocess.Popen(
            [sys.executable, '-u', str(script_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Combine stderr into stdout
            text=True,
            bufsize=1,
            universal_newlines=True,
            cwd=str(Path(__file__).parent)  # Set working directory
        )
        
        print(f"{app['color']}  Process ID: {process.pid}{Colors.END}")
        print(f"{app['color']}  Status: Started{Colors.END}\n")
        
        return process
    except Exception as e:
        print(f"{Colors.RED}✗ Error starting {app['name']}: {e}{Colors.END}")
        import traceback
        traceback.print_exc()
        return None

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print(f"\n\n{Colors.YELLOW}Shutting down all applications...{Colors.END}")
    
    for i, process in enumerate(processes):
        if process and process.poll() is None:
            app = APPS[i]
            print(f"{app['color']}Stopping {app['name']}...{Colors.END}")
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
    
    print(f"{Colors.GREEN}All applications stopped.{Colors.END}\n")
    sys.exit(0)

def main():
    """Main function to launch all applications"""
    print_header()
    
    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start all applications
    print(f"{Colors.BOLD}Starting applications...{Colors.END}\n")
    
    for app in APPS:
        print_app_info(app)
        process = start_app(app)
        if process:
            processes.append(process)
            time.sleep(1)  # Small delay between starts
        else:
            processes.append(None)
    
    # Print summary
    print(f"\n{Colors.GREEN}{Colors.BOLD}{'='*70}")
    print("✓ All applications started successfully!")
    print(f"{'='*70}{Colors.END}\n")
    
    print(f"{Colors.BOLD}Access the applications at:{Colors.END}")
    for app in APPS:
        print(f"  {app['color']}• {app['name']}:{Colors.END} {Colors.BOLD}http://localhost:{app['port']}{Colors.END}")
    
    print(f"\n{Colors.YELLOW}Press Ctrl+C to stop all applications{Colors.END}\n")
    
    # Start output readers for each process
    import threading
    
    def read_output(process, app_name, color):
        """Read output from a process and display it"""
        if not process:
            return
        try:
            for line in iter(process.stdout.readline, ''):
                if line:
                    print(f"{color}[{app_name}]{Colors.END} {line.rstrip()}")
        except Exception as e:
            print(f"{Colors.RED}Error reading output from {app_name}: {e}{Colors.END}")
    
    # Start output reader threads
    output_threads = []
    for i, process in enumerate(processes):
        if process:
            app = APPS[i]
            thread = threading.Thread(
                target=read_output,
                args=(process, app['name'], app['color']),
                daemon=True
            )
            thread.start()
            output_threads.append(thread)
    
    # Monitor processes
    try:
        while True:
            time.sleep(2)
            # Check if any process has died
            for i, process in enumerate(processes):
                if process and process.poll() is not None:
                    app = APPS[i]
                    return_code = process.returncode
                    print(f"\n{Colors.RED}⚠ {app['name']} has stopped unexpectedly{Colors.END}")
                    print(f"{Colors.RED}  Return code: {return_code}{Colors.END}")
                    print(f"{Colors.RED}  Process ID: {process.pid}{Colors.END}")
                    
                    # Try to read any remaining output
                    try:
                        remaining_output = process.stdout.read()
                        if remaining_output:
                            print(f"{Colors.RED}  Last output:{Colors.END}")
                            print(remaining_output)
                    except:
                        pass
                    
                    # Optionally restart it
                    # processes[i] = start_app(app)
    except KeyboardInterrupt:
        signal_handler(None, None)

if __name__ == '__main__':
    main()

